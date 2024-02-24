import logging

from mgz import settings
from mgz.ds.sentence_datasets.gpt_input_augments import \
    ContextPromptingInput, PromptConfig
from mgz.ds.sentence_datasets.sentence_datasets import \
    prompts_to_padded_id_tensor_w_mask
from mgz.model_running.run_ops import prompt_lm_logits_controller
from mgz.models.nlp.base_transformer import InferenceContext
from mgz.version_control import ModelNode, ModelDatabase

logging.basicConfig(level=logging.WARNING)
model_node: ModelNode = ModelDatabase.mistral_openchat(
    # "AdaptLLM/law-chat")
    # "mistralai/Mistral-7B-Instruct-v0.1")
    "mistralai/Mistral-cont-exp/data_EnronEmailsTagQA_1e-5_lbl_fix1/BEST")
    # "mistralai/Mistral-cont-exp/data_EnronEmailsTagQA_1e-7")
# "openchat/openchat-3.5-0106")
# "teknium/OpenHermes-2.5-Mistral-7B")
# AdaptLLM/law-chat
model_node.model.eval()
print(model_node.tokenizer.special_tokens_map)

email = """Message-ID: <5140735.1075840386341.JavaMail.evans@thyme>
Date: Thu, 13 Dec 2001 15:21:00 -0800 (PST)
From: richardson@copn.com
To: cliff.baxter@enron.com, rick.buy@enron.com, richard.causey@enron.com, 
	mark.frevert@enron.com, joe.hirko@enron.com, 
	stanley.horton@enron.com, j..kean@enron.com, mark.koenig@enron.com, 
	mike.mcconnell@enron.com, jeffrey.mcmahon@enron.com, 
	mark.metts@enron.com, cindy.olson@enron.com, lou.pai@enron.com, 
	ken.rice@enron.com, joe.sutton@enron.com, c..williams@enron.com
Subject: Confidential communications
Cc: c..williams@enron.com, gail.brownfeld@enron.com
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
Bcc: c..williams@enron.com, gail.brownfeld@enron.com
X-From: Richardson, Terri <richardson@copn.com>
X-To: Baxter, Cliff <jcbax1@aol.com>, Buy, Rick </O=ENRON/OU=NA/CN=RECIPIENTS/CN=RBUY>, Causey, Richard </O=ENRON/OU=NA/CN=RECIPIENTS/CN=RCAUSEY>, Frevert, Mark </O=ENRON/OU=NA/CN=RECIPIENTS/CN=MFREVERT>, Hirko, Joe <joehirko@aol.com>, Horton, Stanley </O=ENRON/OU=NA/CN=RECIPIENTS/CN=SHORTON>, Kean, Steven J. </O=ENRON/OU=NA/CN=RECIPIENTS/CN=SKEAN>, Koenig, Mark </O=ENRON/OU=NA/CN=RECIPIENTS/CN=MKOENIG>, McConnell, Mike <mike_s_mcconnell@hotmail.com>, McMahon, Jeffrey </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JMCMAHO>, Metts, Mark </O=ENRON/OU=NA/CN=RECIPIENTS/CN=MMETTS>, Olson, Cindy </O=ENRON/OU=NA/CN=RECIPIENTS/CN=COLSON>, Pai, Lou <kowens@loupai.com>, Rice, Ken <krice@rvtcapital.com>, Sutton, Joe <jsutton@suttonventures.com>, Williams, Robert C. </O=ENRON/OU=NA/CN=RECIPIENTS/CN=RWILLIA2>
X-cc: Williams, Robert C. </O=ENRON/OU=NA/CN=RECIPIENTS/CN=RWILLIA2>, Brownfeld, Gail </O=ENRON/OU=NA/CN=RECIPIENTS/CN=GBROWNF>
X-bcc: 
X-Folder: \rbuy\Inbox
X-Origin: BUY-R
X-FileName: richard buy 1-30-02..pst

J.C. asked that I send you the following message:

In light of rumors that investigations into Enron's financial difficulties may be launched or expanded, including investigations by the civil plaintiffs or by the FBI on behalf of the SEC or Congress, we believe it prudent to advise you that if anyone contacts you and seeks to question you (even someone with a badge and an authoritative demeanor), don't answer their questions. Refer them to us for the purpose of arranging an interview.

Second, if you know of any individuals who you believe might be interviewed in connection with these investigations, please call us so we can discuss how such information should be forwarded to the appropriate people.

Please call if you have any questions.

 



This e-mail and any attached files may be confidential and subject to attorney/client privilege. If you received it in error, please immediately notify the sender by return e-mail or by calling (713)654-7600. 
"""
# tag = 'document or communication between enron employees discussing government inquiries and investigations into enron'
tag = 'all documents or communications between enron employees discussing government inquiries and investigations into enron'
system_context = (
    "Given this as some of the background: The FERC's investigating enron for market manipulation. The FERC investigation primarily focused on Enron's role in the California energy crisis of 2000-2001, "
    "along with its trading practices and their impact on electricity markets across the United States. Determine if the email is related to the document request.")

# 12818
print('--------------------------------------------')
# print(email)
# print('--------------------------------------------')
max_src_len = 3000

lm_logits = prompt_lm_logits_controller(model=model_node.model,
                                        texts=[email],
                                        tags=[tag],
                                        system_context=system_context,
                                        tokenizer=model_node.tokenizer,
                                        max_src_len=max_src_len, )

no_yes_scores = InferenceContext(
    model_node.tokenizer, ).get_word_scores_from_logits(
    lm_logits)
print('lm_logits', lm_logits)
print('no_yes_scores', no_yes_scores)
print('decoded lm_logits ',
      model_node.tokenizer.batch_decode(lm_logits.argmax(-1),
                                        skip_special_tokens=True))
src_ids, src_mask = prompts_to_padded_id_tensor_w_mask(
    ContextPromptingInput.from_list(
        PromptConfig(model=model_node.model, system_context=system_context),
        document_texts=[email],
        document_requests_list=[tag]),
    tokenizer=model_node.tokenizer, max_len=max_src_len,
    device=settings.DEVICE)

result = model_node.model.generate(src_ids=src_ids, src_mask=src_mask,
                                   max_new_tokens=200)

answer_start = int(src_ids.shape[-1])
answer = model_node.tokenizer.batch_decode(result[:, answer_start:],
                                           skip_special_tokens=True)

print('answer', answer)

print('original', model_node.tokenizer.batch_decode(result[:, :answer_start],
                                                    skip_special_tokens=True))
