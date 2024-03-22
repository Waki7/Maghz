import logging

from mgz import settings
from mgz.ds.sentence_datasets.gpt_input_augments import \
    FreePromptInput, ContextPromptingInput, PromptConfig
from mgz.ds.sentence_datasets.sentence_datasets import \
    prompts_to_padded_id_tensor_w_mask
from mgz.model_running.run_ops import prompt_lm_logits_controller
from mgz.models.nlp.base_transformer import InferenceContext
from mgz.version_control import ModelNode, ModelDatabase

logging.basicConfig(level=logging.WARNING)
model_node: ModelNode = ModelDatabase.mistral_openchat(
    # "AdaptLLM/law-chat")
    "/home/ceyer/Documents/Projects/Maghz/index_dir/main/mistralai/Mistral-7B-Instruct-v0.2")
    # "mistralai/Mistral-cont-exp/data_EnronEmailsTagQA_1e-5_lbl_fix1/BEST")
    # "mistralai/Mistral-cont-exp/data_EnronEmailsTagQA_1e-7")
# "openchat/openchat-3.5-0106")
# "teknium/OpenHermes-2.5-Mistral-7B")
# AdaptLLM/law-chat
model_node.model.eval()
print(model_node.tokenizer.special_tokens_map)

email = """Message-ID: <28574048.1075852650572.JavaMail.evans@thyme>
Date: Fri, 20 Jul 2001 10:59:25 -0700 (PDT)
From: kevinscott@onlinemailbox.net
To: skean@enron.com
Subject: Moving foward at a good clip
Cc: jeff.skilling@enron.com
Mime-Version: 1.0
Content-Type: text/plain; charset=ANSI_X3.4-1968
Content-Transfer-Encoding: 7bit
Bcc: jeff.skilling@enron.com
X-From: Kevin Scott <kevinscott@onlinemailbox.net>@ENRON <IMCEANOTES-Kevin+20Scott+20+3Ckevinscott+40onlinemailbox+2Enet+3E+40ENRON@ENRON.com>
X-To: Steve Kean <skean@enron.com>
X-cc: Skilling, Jeff </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JSKILLIN>
X-bcc: 
X-Folder: \JSKILLIN (Non-Privileged)\Deleted Items
X-Origin: Skilling-J
X-FileName: JSKILLIN (Non-Privileged).pst


Steve
 
Good news.  As you indicated would happen, Kalen Pieper called me mid-week.  We had a very good conversation about EES and Dave Delainey's leadership.  She explored my views about doing business with government.  Shortly thereafter, Dave's office called to invite me to Houston on Thursday July 26.  
 
Kay Chapman explained that Dave's schedule would keep him out of pocket until August 16.  In order to move forward, Janet Dietrich will be meeting with me when I go to Houston next Thursday.  (I must confess that after all the great things I have heard about the man, I do look forward to meeting Dave himself.)
 
Is there a time next week that I can speak with you by phone to fine-tune my thinking / preparation for Thursday's meeting with Janet?
 
Thank you for your all of your help.  I am pleased and appreciative that things are moving forward at a good clip.  
 
Kevin
___________________________________
Contact Information
E-mail
kevinscott@onlinemailbox.net 
Phone
(213) 926-2626
Fax
(707) 516-0019
Traditional Mail
PO Box 21074 ?Los Angeles, CA 90021
___________________________________
   
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
