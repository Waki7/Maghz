import logging

import torch.cuda.amp
import transformers as hug

from mgz import settings
from mgz.ds.sentence_datasets.datasets_base.sentence_datasets import \
    prompts_to_padded_id_tensor_w_mask
from mgz.ds.sentence_datasets.gpt_input_augments import \
    DocumentRequestChat
from mgz.model_running.run_ops import prompt_lm_logits_controller
from mgz.models.nlp.base_transformer import InferenceContext, ModelType
from mgz.version_control import ModelDatabase, ModelNode

with torch.cuda.amp.autocast(enabled=True):
    logging.basicConfig(level=logging.WARNING)
    model_id = 'mistralai/Mistral-7B-Instruct-v0.2'
    model_node = ModelDatabase.get_quantized_model(model_id)
    model = model_node.model
    tokenizer = model_node.tokenizer

    # model = hug.LlamaForCausalLM.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', device_map="auto").half()
    #
    # model.config._flash_attn_2_enabled = True
    # model.config._attn_implementation = "flash_attention_2"
    #
    # tokenizer = hug.PreTrainedTokenizerFast.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', )
    # tokenizer.pad_token = tokenizer.eos_token
    # model.MODEL_TYPE = ModelType.DecoderTransformer
    # model_node = ModelNode(model=model,
    #                         tokenizer=tokenizer, model_id='meta-llama/Meta-Llama-3-8B-Instruct')

    settings.print_gpu_usage()

    # model_node: ModelNode = ModelDatabase.get_quantized_model(
    #     # "AdaptLLM/law-chat")
    #     "meta-llama/Meta-Llama-3-8B-Instruct")
    #     # "mistralai/Mistral-cont-exp/data_EnronEmailsTagQA_1e-5_lbl_fix1/BEST")
    #     # "mistralai/Mistral-cont-exp/data_EnronEmailsTagQA_1e-7")
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
        "Given this as the only background: The FERC's investigating enron for market manipulation. The FERC investigation primarily focused on Enron's role in the California energy crisis of 2000-2001, "
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
        DocumentRequestChat(
            document_texts=[email],
            document_requests=[[tag]]),
        tokenizer=model_node.tokenizer, max_len=max_src_len,
        device=settings.DEVICE)

    generation_config: hug.GenerationConfig = hug.GenerationConfig()
    generation_config.use_cache = True
    generation_config.max_new_tokens = 100
    result = model_node.model.generate(input_ids=src_ids,
                                       attention_mask=src_mask)

    answer_start = int(src_ids.shape[-1])
    answer = model_node.tokenizer.batch_decode(result[:, answer_start:],
                                               skip_special_tokens=True)

    print('answer', answer)

    print('original',
          model_node.tokenizer.batch_decode(result[:, :answer_start],
                                            skip_special_tokens=True))
