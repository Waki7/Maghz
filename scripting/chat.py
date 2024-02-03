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
    "mistralai/Mistral-7B-Instruct-v0.1")
    # "openchat/openchat-3.5-0106")
# "teknium/OpenHermes-2.5-Mistral-7B")
# AdaptLLM/law-chat
model_node.model.eval()
print(model_node.tokenizer.special_tokens_map)

email = """Message-ID: <21112352.1075851644449.JavaMail.evans@thyme>
Date: Thu, 20 Sep 2001 06:30:51 -0700 (PDT)
From: robert.frank@enron.com
To: ray.alvarez@enron.com, alan.comnes@enron.com, steve.walton@enron.com, 
	susan.mara@enron.com, leslie.lawner@enron.com, w..cantrell@enron.com, 
	donna.fulton@enron.com, jeff.dasovich@enron.com, 
	l..nicolay@enron.com, d..steffes@enron.com, j..noske@enron.com, 
	dave.perrino@enron.com, don.black@enron.com, 
	stephanie.miller@enron.com, barry.tycholiz@enron.com, 
	sarah.novosel@enron.com, jennifer.thome@enron.com, 
	legal <.hall@enron.com>, susan.lindberg@enron.com
Subject: RE: Western Wholesale Activities - Gas & Power Conf. Call
 Privileged & Confidential Communication Attorney-Client Communication and
 Attorney Work Product Privileges Asserted
Mime-Version: 1.0
Content-Type: text/plain; charset=us-ascii
Content-Transfer-Encoding: 7bit
X-From: Frank, Robert </O=ENRON/OU=NA/CN=RECIPIENTS/CN=RFRANK>
X-To: Alvarez, Ray </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Ralvare2>, Comnes, Alan </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Acomnes>, Walton, Steve </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Swalto2>, Mara, Susan </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Smara>, Lawner, Leslie </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Llawner>, Cantrell, Rebecca W. </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Bcantre>, Fulton, Donna </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Dfulton>, Dasovich, Jeff </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Jdasovic>, Nicolay, Christi L. </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Cnicola>, Steffes, James D. </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Jsteffe>, 'jalexander@gibbs-bruns.com', Allen, Phillip K. </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Pallen>, Noske, Linda J. </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Lnoske>, Perrino, Dave </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Dperrino>, Black, Don </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Dblack>, Miller, Stephanie </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Smiller2>, Tycholiz, Barry </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Btychol>, Novosel, Sarah </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Snovose>, Thome, Jennifer </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Jthome>, Hall, Steve C. (Legal) </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Shall4>, Lindberg, Susan </O=ENRON/OU=NA/CN=RECIPIENTS/CN=Slindber>
X-cc: 
X-bcc: 
X-Folder: \Dasovich, Jeff (Non-Privileged)\Dasovich, Jeff\Inbox
X-Origin: DASOVICH-J
X-FileName: Dasovich, Jeff (Non-Privileged).pst

FYI, Attached is memo I prepared for Jim and Rick on the PNW refund case.

 

 -----Original Message-----
From: 	Alvarez, Ray  
Sent:	Thursday, September 20, 2001 7:22 AM
To:	Comnes, Alan; Walton, Steve; Mara, Susan; Lawner, Leslie; Cantrell, Rebecca W.; Fulton, Donna; Dasovich, Jeff; Nicolay, Christi L.; Steffes, James D.; 'jalexander@gibbs-bruns.com'; Allen, Phillip K.; Noske, Linda J.; Perrino, Dave; Black, Don; Frank, Robert; Miller, Stephanie; Tycholiz, Barry; Novosel, Sarah; Thome, Jennifer; Hall, Steve C. (Legal); Lindberg, Susan
Subject:	RE: Western Wholesale Activities - Gas & Power Conf. Call Privileged & Confidential Communication Attorney-Client Communication and Attorney Work Product Privileges Asserted

[Alvarez, Ray]  I will be unable to host the call this morning since I must attend a hearing in the CA refund proceeding.  Please proceed with the call in my absence.  Thanks!  Ray 
PLEASE MARK YOUR CALENDAR
Date:			Every Thursday
Time: 		7:30 am Pacific, 9:30 am Central, and 10:30 am Eastern time
 Number: 		1-888-271-0949 
 Host Code:		661877 (for Ray only)
 Participant Code:	936022 (for everyone else)

The table of the on-going FERC issues and proceedings is available to all team members on the O drive. Please feel free to revise/add to/ update this table as appropriate.

Proposed agenda for today :

ISO settlement redesign and imbalance energy agreement
FERC sponsored reliability meeting at CAISO offices
CA and PAC NW refund proceeding status

Please feel free to communicate any additional agenda items to the group ."""
# tag = 'document or communication between enron employees discussing government inquiries and investigations into enron'
tag = 'all documents or communications between enron employees discussing government inquiries and investigations into enron'
system_context = (
    "Given this as the only background: The FERC's investigating enron for market manipulation. The FERC investigation primarily focused on Enron's role in the California energy crisis of 2000-2001, "
    "along with its trading practices and their impact on electricity markets across the United States. Determine if the email should be produced as evidence based on the document request.")

# 12818
print('--------------------------------------------')
# print(email)
# print('--------------------------------------------')
max_src_len = 8191

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
print('decoded lm_logits ', model_node.tokenizer.batch_decode(lm_logits.argmax(-1),
                                                              skip_special_tokens=True))
src_ids, src_mask = prompts_to_padded_id_tensor_w_mask(
    ContextPromptingInput.from_list(
        PromptConfig(model=model_node.model, system_context=system_context),
        document_texts=[email],
        document_requests_list=[tag]),
    tokenizer=model_node.tokenizer, max_len=max_src_len,
    device=settings.DEVICE)

result = model_node.model.generate(src_ids=src_ids, src_mask=src_mask,
                                   max_new_tokens=96)

answer_start = int(src_ids.shape[-1])
answer = model_node.tokenizer.batch_decode(result[:, answer_start:],
                                           skip_special_tokens=True)

print('answer', answer)

print('original', model_node.tokenizer.batch_decode(result[:, :answer_start],
                                                    skip_special_tokens=True))
