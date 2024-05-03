from __future__ import annotations

import email
import json
import os
import random
import shutil
from functools import partial
from pathlib import Path

import torch.utils.data
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, LlamaTokenizer

import spaces as sp
from mgz.ds import DataState
from mgz.ds.sentence_datasets.datasets_base.enron_emails import EnronEmails
from mgz.ds.sentence_datasets.datasets_base.sentence_datasets import \
    SentenceDataset, \
    SampleType, MetaLearningMixIn
from mgz.ds.sentence_datasets.datasets_metalearning_responsiveness.responsive_batch import \
    TagQAMetaTaskBatch
from mgz.ds.sentence_datasets.gpt_input_augments import PromptConfig, \
    BatchChatInput, DocumentRequestChat
from mgz.ds.sentence_datasets.heuristic_matching import DocumentRuleEvaluator
from mgz.model_running.run_ops import prompt_lm_logits_controller
from mgz.models.nlp.base_transformer import InferenceContext
from mgz.typing import *


class EnronEmailsTagQA(EnronEmails, MetaLearningMixIn):

    def __init__(self, tokenizer: Union[
        LlamaTokenizer, LlamaTokenizer, PreTrainedTokenizerBase],
                 max_src_len: SrcSeqLen,
                 prompt_config: PromptConfig,
                 n_query_per_cls: List[int] = 5,
                 n_support_per_cls: List[int] = 5,
                 n_episodes: int = 100,
                 training_ratio: float = 0.75,
                 dataset_dir: str = None):
        assert isinstance(tokenizer, LlamaTokenizer) or isinstance(
            tokenizer, LlamaTokenizer), \
            f'This dataset requires the LLamaTokenizer found {type(tokenizer)}'
        super(EnronEmailsTagQA, self).__init__(tokenizer=tokenizer,
                                               max_src_len=max_src_len,
                                               dataset_dir=dataset_dir,
                                               training_ratio=training_ratio)
        self._prompt_config = prompt_config
        self._tag_to_sample_idx_map: OrderedDict[str, List[int]] = OrderedDict()

        # Meta Learning Sampling Parameters
        self._n_query_per_cls = n_query_per_cls  # Will be roughly n_shot per class, not exact
        self._n_support_per_cls = n_support_per_cls  # Will be roughly n_shot per class, not exact
        self.n_episodes = n_episodes

    def _load(self, train: bool = False, val: bool = False, test: bool = False):
        super()._load(train, val, test)  # Only loads into self.data
        self.loaded = False
        # keys_to_keep = [
        #     SampleType.MESSAGE_ID,
        #     SampleType.DATE,
        #     SampleType.FROM,
        #     SampleType.TO,
        #     SampleType.SUBJECT,
        #     SampleType.MIME_VERSION,
        #     SampleType.CONTENT_TYPE,
        #     SampleType.CONTENT_TRANSFER_ENCODING,
        #     SampleType.X_FROM,
        #     SampleType.X_TO,
        #     SampleType.X_CC,
        #     SampleType.X_BCC,
        #     SampleType.X_FOLDER,
        #     SampleType.X_ORIGIN,
        #     SampleType.X_FILENAME,
        #     # SampleType.FILE_NAME,
        #     SampleType.CATCHPHRASES,
        #     SampleType.FULL_AS_TEXT,
        # ]
        # for i in range(len(self.data)):
        #     doc_filtered = {key: self.data[i][key] for key in keys_to_keep}
        #     self.data[i] = doc_filtered
        self._tag_to_sample_idx_map: Dict[
            str, List[int]] = self.create_tag_to_sample_idx_map()
        self.loaded = True

    @property
    @overrides(SentenceDataset)
    def input_space(self) -> sp.Sentence:
        return sp.Sentence(
            len((self.vocab_src)), sequence_len=self.max_src_len)

    def __len__(self):
        'Denotes the total number of samples'
        assert self.loaded, "Dataset not loaded"
        if self.n_episodes is not None:
            return self.n_episodes
        return len(self._tag_to_sample_idx_map)

    @property
    @overrides(SentenceDataset)
    def target_space(self) -> sp.Tagging:
        return sp.BinaryTagging()

    def get_collate_fn(self, device: Union[int, torch.device]):
        assert self.loaded, "Dataset not loaded"
        return partial(TagQAMetaTaskBatch.default_collate_fn, self, device)

    @overrides(EnronEmails)
    def __getitem__(self, idx) -> Tuple[
        List[Tuple[BatchChatInput, int]], List[Tuple[BatchChatInput, int]]]:
        tags = list(self._tag_to_sample_idx_map.keys())
        tag_idx = idx % len(self._tag_to_sample_idx_map)
        selected_tag: TgtStringT = tags[tag_idx]
        rand_sample_for_tag: SrcStringT = self.data[random.choice(
            self._tag_to_sample_idx_map[selected_tag])][
            SampleType.FULL_AS_STRING]
        batch = rand_sample_for_tag, selected_tag
        assert self.data_state != DataState.NOT_LOADED, "Dataset not loaded"
        assert len(batch) == 1, "Batch size must be 1 for meta-learning for now"
        src_text, pos_tag = batch[0]

        # self.task_sizes_per_cls can be a single value
        n_query_per_cls: int = random.choice(self.n_query_per_cls)
        n_support_per_cls: int = random.choice(self.n_support_per_cls)
        task_size_per_cls: int = n_query_per_cls + n_support_per_cls

        # Select the samples for the task based on the tag.
        pos_sample_idxs: List[int] = self.tag_to_sample_idx_map[
            pos_tag]
        random.shuffle(pos_sample_idxs)
        positive_examples = pos_sample_idxs[:task_size_per_cls]

        # If we're having trouble finding negative examples, we'll timeout,
        # we currently randomly check, we don't keep an index that maps from
        # negative to the sample indices.
        timeout = 5 * task_size_per_cls
        neg_sampling_tries = 0
        negative_examples: set[int] = set()
        while (len(negative_examples) < task_size_per_cls) and \
                neg_sampling_tries < timeout:
            # TODO: experiment with pulling negative samples that have very
            #  different embeddings
            if len(negative_examples) < task_size_per_cls:
                neg_sample_idx = random.randint(0, len(self.data) - 1)
                neg_tags = self.data[neg_sample_idx][
                    SampleType.CATCHPHRASES]
                if pos_tag not in neg_tags:
                    negative_examples.add(neg_sample_idx)
            neg_sampling_tries += 1
        pos_batch: List[Tuple[BatchChatInput, LabelT]] = \
            [(DocumentRequestChat(
                prompt_config=self.prompt_config,
                # document_text=ds.data[i][SampleType.FILE_NAME] + ds.data[i][
                #     SampleType.FULL_AS_STRING],
                document_text=self.data[i][SampleType.FULL_AS_STRING],
                document_requests=pos_tag, ),
              1) for i in positive_examples]
        neg_batch: List[Tuple[BatchChatInput, LabelT]] = \
            [(DocumentRequestChat(
                prompt_config=self.prompt_config,
                # document_text=ds.data[i][SampleType.FILE_NAME] + ds.data[i][
                #     SampleType.FULL_AS_STRING],
                document_text=self.data[i][SampleType.FULL_AS_STRING],
                document_requests=pos_tag, ),
              0) for i in negative_examples]

        # TODO fix so we catch this earlier
        min_to_have_1_query = n_support_per_cls + 1
        if len(neg_batch) < min_to_have_1_query or len(
                pos_batch) < min_to_have_1_query:
            return None
        return neg_batch, pos_batch


def dump_n_examples(model_name: str, n: int = 100000000):
    from mgz.version_control import ModelDatabase, ModelNode
    logging.basicConfig(level=logging.WARNING)

    tag_to_sample = 'all documents or communications between enron employees discussing government inquiries and investigations into enron'
    system_context = (
        "Given this as the only background: The FERC's investigating enron for market manipulation. The FERC investigation primarily focused on Enron's role in the California energy crisis of 2000-2001, "
        "along with its trading practices and their impact on electricity markets across the United States. Determine if the email should be produced as evidence based on the document request.")
    export_dir = os.path.join(
        Path(__file__).resolve().parent.parent.parent.parent,
        f"datasets/enron_export_investigations_{model_name.replace('/', '_')}/").replace(
        "\\", "/")
    model_node: ModelNode = ModelDatabase.mistral_openchat(model_name)
    model_node.model.eval()

    rule_set = ("or",
                ("inquir", 1),
                ("investigat", 1),
                ("government", 1),
                ("FERC", 2),
                ("and", "government", "investigat", 3),
                ("and", "government", "inquir", 3),
                ("and", "FERC", "investigat", 4),
                ("and", "FERC", "inquir", 4),
                ("and", "FERC", "government", 4),
                )
    heuristic_evaluator = DocumentRuleEvaluator(rule_set)

    bsz = 2
    max_src_len = 8191
    sample_negative = False
    if sample_negative:
        p_should_include_maybes = 0.25
        p_random = 0.09
    else:
        p_should_include_maybes = 1.0
        p_random = 1.0
    # mgz.settings.print_parameters(model_node.model)
    # model_node.tokenizer = LlamaTokenizer.from_pretrained(
    #     "openchat/openchat-3.5-0106")
    # model_node.tokenizer.padding_side = 'left'
    # model_node.tokenizer.add_special_tokens(
    #     {'pad_token': model_node.tokenizer.eos_token})

    ds = EnronEmails(model_node.tokenizer,
                     training_ratio=1.0,
                     max_src_len=max_src_len,
                     max_tgt_len=-1,
                     dataset_dir='/datasets/enron_with_categories').load_training_data()

    #     answers = tag_questions_controller(
    #         model=model_node.model, texts=[r'''Message-ID: <7361482.1075847628266.JavaMail.evans@thyme>
    # Date: Wed, 28 Feb 2001 06:18:00 -0800 (PST)
    # From: steven.kean@enron.com
    # To: jeffrey.shankman@enron.com
    # Subject: Re:
    # Mime-Version: 1.0
    # Content-Type: text/plain; charset=us-ascii
    # Content-Transfer-Encoding: 7bit
    # X-From: Steven J Kean
    # X-To: Jeffrey A Shankman
    # X-cc:
    # X-bcc:
    # X-Folder: \Steven_Kean_June2001_1\Notes Folders\All documents
    # X-Origin: KEAN-S
    # X-FileName: skean.nsf
    #
    # help is on the way.  We are going to establish some senior executive
    # oversight as well as an executive director (probably someone who came through
    # the program).  Charlene is going to be moving into a commercial role (which
    # was the original plan when she was brought into the organization).  I'd be
    # happy to talk to you in more detail when I get back to Houston.
    #
    #
    #
    # 	Jeffrey A Shankman@ECT
    # 	02/26/2001 10:21 AM
    #
    # 		 To: Jeff Skilling/Corp/Enron@ENRON, Steven J Kean/NA/Enron@Enron
    # 		 cc:
    # 		 Subject:
    #
    # Hi guys.
    #
    # I wanted to let you know that I am extremely concerned about the
    # Associate/Analyst program, so much so that I feel all the work I have done,
    # and all the time I have spent on the program has had little impact outside
    # Wharton/Penn recruiting.  In fact we won't get more than 1 associate from
    # Wharton this year for a variety of internal and external reasons.  This
    # program has brought incredible talent into the organization, but we have lost
    # a lot of momentum over the last two years.
    #
    # In as much as I would like to continue to support the program, I can't  in
    # its current form, and don't have time to fix what I thought we had been
    # addressing.  The entire program is disfunctional, and the commercial teams
    # are not lending support to the program.  I'd be very happy to spend a few
    # minutes of your time (rather than blather on in an email) to give you both my
    # overview of the program, and suggest changes and improvements.
    #
    # You know you have my support, but the current state of affairs down there has
    # gotten me to my rope's end with the program.'''] * 2,
    #         tags=[tag_to_sample] * 2, tokenizer=model_node.tokenizer,
    #         augment_function=ContextInputAugment(),
    #         max_new_tokens=1, max_src_len=max_src_len)
    #     print('starting answers', answers)
    # exit(3)
    keys_to_keep = [
        SampleType.MESSAGE_ID,
        SampleType.DATE,
        SampleType.FROM,
        SampleType.TO,
        SampleType.SUBJECT,
        SampleType.MIME_VERSION,
        SampleType.CONTENT_TYPE,
        SampleType.CONTENT_TRANSFER_ENCODING,
        SampleType.X_FROM,
        SampleType.X_TO,
        SampleType.X_CC,
        SampleType.X_BCC,
        SampleType.X_FOLDER,
        SampleType.X_ORIGIN,
        SampleType.X_FILENAME,
        SampleType.FILE_NAME,
        # SampleType.CATCHPHRASES,
        SampleType.BODY,
        SampleType.FULL_AS_STRING,
        # SampleType.FULL_AS_BYTES
    ]
    docs_formatted = []

    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.mkdir(export_dir)
    os.mkdir(os.path.join(export_dir, 'yes'))
    os.mkdir(os.path.join(export_dir, 'no'))
    os.mkdir(os.path.join(export_dir, 'maybe'))
    with open(os.path.join(export_dir, 'categories_map.json'), 'w') as file:
        file.write(json.dumps({1: tag_to_sample}))

    for i in tqdm(range(0, len(ds), bsz)):
        batch = ds[i:i + bsz]
        doc_batch_filtered = [{key: doc[key] for key
                               in keys_to_keep} for doc in batch]
        full_texts = [doc[SampleType.FULL_AS_STRING] for doc in batch]
        tags: List[str] = [tag_to_sample] * len(full_texts)
        lm_logits = prompt_lm_logits_controller(model=model_node.model,
                                                texts=full_texts,
                                                tags=tags,
                                                tokenizer=model_node.tokenizer,
                                                max_src_len=max_src_len,
                                                prompt_config=PromptConfig(
                                                    system_context=system_context,
                                                    model=model_node.model))
        no_yes_scores = InferenceContext(
            model_node.tokenizer).get_word_scores_from_logits(
            lm_logits)
        answers = model_node.tokenizer.batch_decode(lm_logits.argmax(-1),
                                                    skip_special_tokens=True)
        assert len(answers) == len(doc_batch_filtered)
        for i, (answer, doc) in enumerate(zip(answers, doc_batch_filtered)):
            print('--------------------------------------------')
            # print(full_texts[i])
            # print('--------------------------------------------')

            print('answer', answer)
            print('lm_logits', lm_logits[i])
            print('no_yes_scores', no_yes_scores[i])
            print('doc', doc[SampleType.FILE_NAME])
            random_p = random.random()
            no_inclusion = no_yes_scores[i].argmax(
                0) == 0 and random_p < p_random

            doc_content_lower = doc[SampleType.FULL_AS_STRING].lower()
            doc_heuristic_score = heuristic_evaluator.check_document(
                doc_content_lower)
            maybe_inclusion = doc_heuristic_score > 0 and random.random() < p_should_include_maybes

            cls_probs = torch.softmax(no_yes_scores[i], 0)
            yes_inclusion = cls_probs.argmax(0) == 1 or (
                    maybe_inclusion and cls_probs[1] > (
                    0.6 - (doc_heuristic_score * 0.025)))
            if yes_inclusion:
                maybe_inclusion = False

            if sample_negative and (
                    not (
                            no_inclusion or maybe_inclusion or yes_inclusion)):
                continue
            docs_formatted.append(
                {str(key) + ".value": doc[key] for key in keys_to_keep})

            if yes_inclusion:
                shutil.copy(doc[SampleType.FILE_NAME], os.path.join(
                    export_dir, 'yes'))
            elif maybe_inclusion:
                shutil.copy(doc[SampleType.FILE_NAME], os.path.join(
                    export_dir, 'maybe'))
            elif no_inclusion:
                shutil.copy(doc[SampleType.FILE_NAME], os.path.join(
                    export_dir, 'no'))

            email_msg: email.message.Message = email.message_from_string(
                doc[SampleType.FULL_AS_STRING])
            assert email_msg.as_string() == doc[SampleType.FULL_AS_STRING]
            # if "135886.txt" in doc[SampleType.FILE_NAME]:
            #     print('email_msg', email_msg)
            #     print('doc', doc)
            #     exit(3)
            # email_msg: email.message.Message = email.message_from_bytes(
            #     doc[SampleType.FULL_AS_BYTES])
            # assert email_msg.as_bytes() == doc[SampleType.FULL_AS_BYTES]
        if len(docs_formatted) > n:
            break
        print(f'collected {len(docs_formatted)} examples so far')

    as_dict_file = os.path.join(export_dir, 'formatted_docs.json')
    # Writing to the file
    with open(as_dict_file, 'w') as file:
        file.write('[\n')
        for entry in docs_formatted:
            file.write('\t{\n')
            for key, value in entry.items():
                cleaned_key = key.replace('\"', '')
                file.write(f"\t\t{cleaned_key}: {repr(value)},\n")
            file.write('\t},\n')
        file.write(']\n')

    # Reading from the file and printing its contents
    # with open(as_dict_file, 'r') as file:
    #     content = file.read()
    #     print(content)


class CustomTqdm(tqdm):
    @staticmethod
    def get_for_job(job_id: int):
        return partial(CustomTqdm, job_id=job_id)

    def __init__(self, iterable, desc=None, leave=True, file=None,
                 job_id: int = None):
        super().__init__(iterable, desc, leave, file, mininterval=2.0)
        self.job_id = job_id

    @overrides(tqdm)
    def update(self, n=30):
        super().update(n)
        print('total', len(self.iterable))
        print('update', n)
        print(self.job_id)


def main():
    with torch.no_grad():
        # ds = EnronEmails(LlamaTokenizer.from_pretrained("AdaptLLM/law-chat"),
        #                  training_ratio=0.1,
        #                  max_src_len=4096,
        #                  max_tgt_len=-1,
        #                  dataset_dir='/Users/ceyer/Documents/Projects/Maghz/datasets/enron_with_categories').load_training_data()
        # print(ds[0].get(SampleType.FULL_AS_STRING))
        # exit(4)

        # model_name = 'openchat/openchat-3.5-0106'
        # model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
        # model_name = 'jan-hq/Mistral-7B-Instruct-v0.2-SLERP'
        dump_n_examples('mistralai/Mistral-7B-Instruct-v0.2')


if __name__ == '__main__':
    main()
