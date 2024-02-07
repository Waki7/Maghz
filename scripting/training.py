from __future__ import annotations

import os

from mgz.ds.sentence_datasets.gpt_input_augments import PromptConfig

os.putenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# from mgz.models.nlp.bart_interface import BARTHubInterface

from mgz.model_running.nlp_routines.model_routine_tagging import DistanceMeasure
from mgz.ds.sentence_datasets.enron_emails import EnronEmailsTagging, \
    EnronEmailsTagQA
from mgz.typing import *

import torch
import mgz.settings as settings
from mgz.ds.sentence_datasets.synthetic_memorization import \
    SyntheticMemorization
import logging
from mgz.ds.sentence_datasets.aus_legal_case_reports import \
    AusCaseReportsToTagGrouped
from mgz.model_running.nlp_routines.model_routine_tagging import TaggingRoutine
from mgz.version_control import ModelNode, ModelDatabase
from mgz.version_control.model_edge import ModelTransitionEdge
from mgz.ds.sentence_datasets.sentence_datasets import SentenceDataset

'''
input_ids tensor([[    0, 31414,   232,  1437,     2,     2]])
attention_mask None
attn_weights torch.Size([16, 6, 6])
attn_probs torch.Size([16, 6, 6])
'''


def dataset_memorization():
    return SyntheticMemorization(128, 128, 128, 1000, 1), SyntheticMemorization(
        128, 128, 128, 1000, 1)


def dataset_select(model_node: ModelNode, aus: bool = False,
                   enron: bool = False, old_enron=False) -> Tuple[
    SentenceDataset, SentenceDataset]:
    assert aus or enron or old_enron, 'must select a dataset'
    ds, val_ds = None, None
    cfg = model_node.model.config
    if aus:
        ds = AusCaseReportsToTagGrouped(model_node.tokenizer,
                                        max_src_len=3000,
                                        n_episodes=1000, n_queries_per_cls=[2],
                                        n_supports_per_cls=[3, 4])
        val_ds = AusCaseReportsToTagGrouped(model_node.tokenizer,
                                            max_src_len=3000,
                                            n_episodes=100,
                                            n_queries_per_cls=[2],
                                            n_supports_per_cls=[3, 4])
    if enron:
        system_context = (
            "Given this as the only background: The FERC's investigating enron for market manipulation. The FERC investigation primarily focused on Enron's role in the California energy crisis of 2000-2001, "
            "along with its trading practices and their impact on electricity markets across the United States. Determine if the email should be produced as evidence based on the document request.")
        prompt_config = PromptConfig(model=model_node.model,
                                     system_context=system_context)
        ds = EnronEmailsTagQA(model_node.tokenizer,
                              prompt_config=prompt_config,
                              max_src_len=7000,
                              n_episodes=100,
                              n_query_per_cls=[1],
                              n_support_per_cls=[1, 2, 3, 4, 5, 6, 7, 8],
                              dataset_dir="/home/ceyer/Documents/Projects/Maghz/datasets/enron_export_investigations_mistral_labeled")
        val_ds = EnronEmailsTagQA(model_node.tokenizer,
                                  prompt_config=prompt_config,
                                  max_src_len=7000,
                                  n_episodes=25,
                                  n_query_per_cls=[1],
                                  n_support_per_cls=[1, 2, 3, 4, 5, 6, 7, 8],
                                  dataset_dir="/home/ceyer/Documents/Projects/Maghz/datasets/enron_export_investigations_mistral_labeled")
    if old_enron:
        ds = EnronEmailsTagging(model_node.tokenizer,
                                max_src_len=3000,
                                n_episodes=2000,
                                n_query_per_cls=[1],
                                n_support_per_cls=[1, 2, 3])
        val_ds = EnronEmailsTagging(model_node.tokenizer,
                                    max_src_len=3000,
                                    n_episodes=25,
                                    n_query_per_cls=[1],
                                    n_support_per_cls=[1, 2, 3])
    return ds, val_ds


# 174178


def led_main_train():
    # import transformers
    # tokenizer = transformers.LlamaTokenizer.from_pretrained(
    #     "openchat/openchat_3.5")
    # print(tokenizer.chat_template)
    # print(tokenizer.chat_template_ids)
    # print(type(tokenizer))
    # print(tokenizer)
    # exit(4)

    logging.basicConfig(level=logging.WARNING)

    # Bart Models
    # model_name = "facebook/bart-base"
    # model_name = 'allenai/bart-large-multi_lexsum-long-tiny'
    # model_name = 'allenai/bart-large-multi_lexsum-long-short'
    # model_name = 'allenai/primera-multi_lexsum-source-long'
    # model_name = 'allenai/led-base-16384-multi_lexsum-source-tiny'

    # LED Models
    # model_name = 'allenai/led-base-16384'
    # model_name = 'allenai/led-large-16384'
    # model_name = 'allenai/led-base-16384-multi_lexsum-source-long'

    # Mistral Models
    # model_name = 'mistralai/Mistral-7B-v0.1'
    model_name = 'mistralai/Mistral-7B-Instruct-v0.1'
    # model_name = 'mistralai/Mistral-cont-exp'
    # model_name = 'openchat/openchat_3.5'
    # model_name = 'openchat/openchat-3.5-0106'
    # model_name = 'AdaptLLM/law-chat'
    # model_name = 'openchat/openchat_3.5/data_EnronEmailsTagQA_2/BEST'
    # model_name = 'facebook/bart-large-cnn'
    # model_cls = BartForBinaryTagging
    print('... loading model and tokenizer')

    with torch.cuda.amp.autocast(enabled=True):
        model_node: ModelNode = ModelDatabase.mistral_openchat(model_name)

        # ModelNode.load_from_id(LEDForConditionalGeneration, model_name,
        model_node.model.train()
        settings.print_gpu_usage()
        ds, val_ds = dataset_select(model_node, aus=False, enron=True)
        loss_fn = torch.nn.NLLLoss()
        # loss_fn = torch.nn.CrossEntropyLoss()

        training_cfg = {}
        lr = 0.001
        weight_decay = 0.0000
        betas = (0.9, 0.999)
        eps = 1e-4

        model_node.freeze_parameters_except_for([
            'embedding_head',
            # 'lm_head',
            # 'model.layers.31.mlp'
        ])

        # embed_params = model_node.get_parameters_by_string_in_name(
        #     'embedding_head')
        # lm_head_params = model_node.get_parameters_by_string_in_name('lm_head')
        # mlp_params = model_node.get_parameters_by_string_in_name(
        #     'model.layers.31.mlp')
        # trainable_params = [
        #     {'params': embed_params, 'lr': 0.0005},
        #     {'params': lm_head_params, 'lr': 0.00001},
        #     {'params': mlp_params, 'lr': 0.000001},
        # ]

        trainable_params = [p for n, p in model_node.model.named_parameters() if
                            p.requires_grad]

        settings.print_trainable_parameters(model_node.model)
        optimizer = model_node.get_optimizer(quantize=True, lr=lr,
                                             params=trainable_params,
                                             weight_decay=weight_decay,
                                             eps=eps, betas=betas)

        # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
        train_transition_edge = ModelTransitionEdge(model_node, loss_fn,
                                                    optimizer, ds)
        routine = TaggingRoutine(
            distance_measure=DistanceMeasure.COSINE,
            tokenizer=model_node.tokenizer, debug=False, gpu_max_batch_size=2)

        # routine.evaluate(model_node=model_node, val_ds=val_ds)
        routine.train(
            model_node=model_node, ds=ds, val_ds=val_ds,
            model_edge=train_transition_edge,
            device=settings.DEVICE, distributed=False,
            turn_off_shuffle=True, n_epochs=50, )
        torch.cuda.empty_cache()


if __name__ == '__main__':
    led_main_train()
