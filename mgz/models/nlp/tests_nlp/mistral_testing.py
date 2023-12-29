from __future__ import annotations

import math
import os
import time

from mgz import settings
from mgz.ds.sentence_datasets.enron_emails import EnronEmailsTagQA
from mgz.models.nlp.mistral import MistralForCausalLM
from mgz.models.nlp.mistral_hug import MistralForCausalLMHug
from mgz.typing import *
from mgz.version_control import ModelNode, Metrics


@torch.no_grad()
def mistral_original():
    import os
    import mgz.settings as settings
    os.putenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import transformers as hug

    logging.basicConfig(level=logging.WARNING)
    model_name = 'openchat/openchat_3.5'
    print('... loading model and tokenizer')
    with torch.cuda.amp.autocast(enabled=True):
        quantize = True
        quantization_cfg = None
        if quantize:
            try:
                from accelerate.utils import BnbQuantizationConfig
                import bitsandbytes

                quantization_cfg = BnbQuantizationConfig(
                    load_in_8bit=quantize, )
            except ImportError:
                print("Module 'some_module' is not installed.")
                quantization_cfg = None
                quantize = False

        # model_node: ModelNode = \
        #     ModelNode.load_from_id(MistralForCausalLM, model_name,
        #                            model_name,
        #                            quantization_config=quantization_cfg)
        # tokenizer = model_node.tokenizer
        # model_node.model.eval()

        tokenizer = hug.LlamaTokenizerFast.from_pretrained(
            model_name)
        model_hug = hug.MistralForCausalLM.from_pretrained(model_name,
                                                           device_map={
                                                               "": settings.DEVICE},
                                                           load_in_8bit=True)
        model_hug.eval()

    with (torch.cuda.amp.autocast(enabled=True)):
        tag_qa_text = (
            f"GPT4 Correct User: Is this e-mail about company business or strategy?: "
            """
            Message-ID: <23743848.1075863311776.JavaMail.evans@thyme>
            Date: Wed, 11 Jul 2001 08:29:22 -0700 (PDT)
            From: legalonline-compliance@enron.com
            To: williams@mailman.enron.com, bwillia5@enron.com
            Subject: Confidential Information and Securities Trading
            Mime-Version: 1.0
            Content-Type: text/plain; charset=us-ascii
            Content-Transfer-Encoding: 7bit
            X-From: Office of the Chairman - Enron Wholesale Services <legalonline-compliance@enron.com>@ENRON <IMCEANOTES-Office+20of+20the+20Chairman+20-+20Enron+20Wholesale+20Services+20+3Clegalonline-compliance+40enron+2Ecom+3E+40ENRON@ENRON.com>
            X-To: WILLIAMS@mailman.enron.com, WILLIAM <bwillia5@enron.com>
            X-cc: 
            X-bcc: 
            X-Folder: \Williams III, Bill (Non-Privileged)\Bill Williams III
            X-Origin: Williams-B
            X-FileName: Williams III, Bill (Non-Privileged).pst
    
            To:WILLIAMS, WILLIAM
            Email:bwillia5@enron.com - 503-464-3730
    
            Enron Wholesale Services - Office of the Chairman
    
            From:  Mark Frevert, Chairman & CEO
            Mark Haedicke, Managing Director & General Counsel
    
            Subject:  Confidential Information and Securities Trading
    
            To keep pace with the fluid and fast-changing demands of our equity trading activities, Enron Wholesale Services ("EWS") has recently revised its official Policies and Procedures Regarding Confidential Information and Securities Trading ("Policies and Procedures").  These revisions reflect two major developments: (1) our equity trading activities have been extended into the United Kingdom, and (2) in an effort to streamline the information flow process, the "Review Team" will play a more centralized role, so that the role of the "Resource Group" is no longer necessary.You are required to become familiar with, and to comply with, the Policies and Procedures.  The newly revised Policies and Procedures are available for your review on LegalOnline, the new intranet website maintained by the Enron Wholesale Services Legal Department.  Please click on the attached link to access LegalOnline:
            http://legalonline.corp.enron.com/chinesewall.asp 
    
            If you have already certified compliance with the Policies and Procedures during the 2001 calendar year, you need not re-certify at this time, although you are still required to to review and become familiar with the revised Policies and Procedures.  If you have not certified compliance with the Policies and Procedures during the 2001 calendar year, then you must do so within two weeks of your receipt of this message.  The LegalOnline site will allow you to quickly and conveniently certify your compliance on-line with your SAP Personal ID number.  If you have any questions concerning the Policies or Procedures, please call Bob Bruce at extension 5-7780 or Donna Lowry at extension 3-1939. 
            """
            f"<|end_of_turn|>GPT4 Correct Assistant: ")
        text = [tag_qa_text]

        # tokens = generate_controller(model_node.model, text,
        #                              model_node.tokenizer)
        # summary: List[str] = model_node.tokenizer.batch_decode(tokens,
        #                                                        skip_special_tokens=True)
        # print('mgz decoded', summary)

        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenized = tokenizer(text, return_tensors='pt',
                              padding=True)
        input_ids = tokenized.input_ids.to(settings.DEVICE)
        attention_mask = tokenized.attention_mask.to(settings.DEVICE)
        generated_ids = model_hug.generate(input_ids,
                                           attention_mask=attention_mask,
                                           max_new_tokens=1000)
        # print('generated_ids', generated_ids)
        summary: List[str] = tokenizer.batch_decode(generated_ids,
                                                    skip_special_tokens=True)
        print('hug decoded', summary)


@torch.no_grad()
def mistral_mgz():
    import os
    os.putenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    logging.basicConfig(level=logging.WARNING)
    model_name = 'openchat/openchat_3.5'
    print('... loading model and tokenizer')
    with torch.cuda.amp.autocast(enabled=True):
        quantize = True
        quantization_cfg = None
        if quantize:
            try:
                from accelerate.utils import BnbQuantizationConfig
                import bitsandbytes

                quantization_cfg = BnbQuantizationConfig(
                    load_in_8bit=quantize, )
            except ImportError:
                print("Module 'some_module' is not installed.")
                quantization_cfg = None
                quantize = False

        model_node: ModelNode = \
            ModelNode.load_from_id(MistralForCausalLM, model_name,
                                   model_name,
                                   quantization_config=quantization_cfg)
        tokenizer = model_node.tokenizer
        model_node.model.eval()

    with (torch.cuda.amp.autocast(enabled=True)):
        tag_qa_text = (
            f"GPT4 Correct User: Is this e-mail about company business or strategy?: "
            """
            Message-ID: <23743848.1075863311776.JavaMail.evans@thyme>
            Date: Wed, 11 Jul 2001 08:29:22 -0700 (PDT)
            From: legalonline-compliance@enron.com
            To: williams@mailman.enron.com, bwillia5@enron.com
            Subject: Confidential Information and Securities Trading
            Mime-Version: 1.0
            Content-Type: text/plain; charset=us-ascii
            Content-Transfer-Encoding: 7bit
            X-From: Office of the Chairman - Enron Wholesale Services <legalonline-compliance@enron.com>@ENRON <IMCEANOTES-Office+20of+20the+20Chairman+20-+20Enron+20Wholesale+20Services+20+3Clegalonline-compliance+40enron+2Ecom+3E+40ENRON@ENRON.com>
            X-To: WILLIAMS@mailman.enron.com, WILLIAM <bwillia5@enron.com>
            X-cc: 
            X-bcc: 
            X-Folder: \Williams III, Bill (Non-Privileged)\Bill Williams III
            X-Origin: Williams-B
            X-FileName: Williams III, Bill (Non-Privileged).pst

            To:WILLIAMS, WILLIAM
            Email:bwillia5@enron.com - 503-464-3730

            Enron Wholesale Services - Office of the Chairman

            From:  Mark Frevert, Chairman & CEO
            Mark Haedicke, Managing Director & General Counsel

            Subject:  Confidential Information and Securities Trading

            To keep pace with the fluid and fast-changing demands of our equity trading activities, Enron Wholesale Services ("EWS") has recently revised its official Policies and Procedures Regarding Confidential Information and Securities Trading ("Policies and Procedures").  These revisions reflect two major developments: (1) our equity trading activities have been extended into the United Kingdom, and (2) in an effort to streamline the information flow process, the "Review Team" will play a more centralized role, so that the role of the "Resource Group" is no longer necessary.You are required to become familiar with, and to comply with, the Policies and Procedures.  The newly revised Policies and Procedures are available for your review on LegalOnline, the new intranet website maintained by the Enron Wholesale Services Legal Department.  Please click on the attached link to access LegalOnline:
            http://legalonline.corp.enron.com/chinesewall.asp 

            If you have already certified compliance with the Policies and Procedures during the 2001 calendar year, you need not re-certify at this time, although you are still required to to review and become familiar with the revised Policies and Procedures.  If you have not certified compliance with the Policies and Procedures during the 2001 calendar year, then you must do so within two weeks of your receipt of this message.  The LegalOnline site will allow you to quickly and conveniently certify your compliance on-line with your SAP Personal ID number.  If you have any questions concerning the Policies or Procedures, please call Bob Bruce at extension 5-7780 or Donna Lowry at extension 3-1939. 
            """
            f"<|end_of_turn|>GPT4 Correct Assistant: 1. Is this e-mail about company business or strategy?: \n\n")
        text = [tag_qa_text]

        model_inputs = tokenizer(text, return_tensors="pt").to(
            settings.DEVICE)
        generated_ids = model_node.model.generate(
            src_ids=model_inputs.input_ids,
            src_mask=model_inputs.attention_mask,
            tgt_ids=model_inputs.input_ids,
            max_new_tokens=2000)

        print(model_inputs.input_ids[:, -5:],
              tokenizer.batch_decode(model_inputs.input_ids[:, -5:]))
        print(generated_ids[:, -5:],
              tokenizer.batch_decode(generated_ids[:, -5:]))
        print(tokenizer.batch_decode(generated_ids))


@torch.no_grad()
def mistral_mgz_hug():
    import os
    os.putenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    logging.basicConfig(level=logging.WARNING)
    model_name = 'openchat/openchat_3.5'
    print('... loading model and tokenizer')
    with torch.cuda.amp.autocast(enabled=True):
        quantize = True
        quantization_cfg = None
        if quantize:
            try:
                from accelerate.utils import BnbQuantizationConfig
                import bitsandbytes

                quantization_cfg = BnbQuantizationConfig(
                    load_in_8bit=quantize, )
            except ImportError:
                print("Module 'some_module' is not installed.")
                quantization_cfg = None
                quantize = False

        model_node: ModelNode = \
            ModelNode.load_from_id(MistralForCausalLMHug, model_name,
                                   model_name,
                                   quantization_config=quantization_cfg)
        tokenizer = model_node.tokenizer
        model_node.model.eval()

    with (torch.cuda.amp.autocast(enabled=True)):
        tag_qa_text = (
            f"GPT4 Correct User: Is this e-mail about company business or strategy?: "
            """
            Message-ID: <23743848.1075863311776.JavaMail.evans@thyme>
            Date: Wed, 11 Jul 2001 08:29:22 -0700 (PDT)
            From: legalonline-compliance@enron.com
            To: williams@mailman.enron.com, bwillia5@enron.com
            Subject: Confidential Information and Securities Trading
            Mime-Version: 1.0
            Content-Type: text/plain; charset=us-ascii
            Content-Transfer-Encoding: 7bit
            X-From: Office of the Chairman - Enron Wholesale Services <legalonline-compliance@enron.com>@ENRON <IMCEANOTES-Office+20of+20the+20Chairman+20-+20Enron+20Wholesale+20Services+20+3Clegalonline-compliance+40enron+2Ecom+3E+40ENRON@ENRON.com>
            X-To: WILLIAMS@mailman.enron.com, WILLIAM <bwillia5@enron.com>
            X-cc: 
            X-bcc: 
            X-Folder: \Williams III, Bill (Non-Privileged)\Bill Williams III
            X-Origin: Williams-B
            X-FileName: Williams III, Bill (Non-Privileged).pst

            To:WILLIAMS, WILLIAM
            Email:bwillia5@enron.com - 503-464-3730

            Enron Wholesale Services - Office of the Chairman

            From:  Mark Frevert, Chairman & CEO
            Mark Haedicke, Managing Director & General Counsel

            Subject:  Confidential Information and Securities Trading

            To keep pace with the fluid and fast-changing demands of our equity trading activities, Enron Wholesale Services ("EWS") has recently revised its official Policies and Procedures Regarding Confidential Information and Securities Trading ("Policies and Procedures").  These revisions reflect two major developments: (1) our equity trading activities have been extended into the United Kingdom, and (2) in an effort to streamline the information flow process, the "Review Team" will play a more centralized role, so that the role of the "Resource Group" is no longer necessary.You are required to become familiar with, and to comply with, the Policies and Procedures.  The newly revised Policies and Procedures are available for your review on LegalOnline, the new intranet website maintained by the Enron Wholesale Services Legal Department.  Please click on the attached link to access LegalOnline:
            http://legalonline.corp.enron.com/chinesewall.asp 

            If you have already certified compliance with the Policies and Procedures during the 2001 calendar year, you need not re-certify at this time, although you are still required to to review and become familiar with the revised Policies and Procedures.  If you have not certified compliance with the Policies and Procedures during the 2001 calendar year, then you must do so within two weeks of your receipt of this message.  The LegalOnline site will allow you to quickly and conveniently certify your compliance on-line with your SAP Personal ID number.  If you have any questions concerning the Policies or Procedures, please call Bob Bruce at extension 5-7780 or Donna Lowry at extension 3-1939. 
            """
            f"<|end_of_turn|>GPT4 Correct Assistant: ")
        text = [tag_qa_text]

        model_inputs = tokenizer(text, return_tensors="pt").to(
            settings.DEVICE)

        # model.config.max_length = 7
        generated_ids = model_node.model.generate(
            src_ids=model_inputs.input_ids,
            src_mask=model_inputs.attention_mask,
            tgt_ids=model_inputs.input_ids,
            max_new_tokens=2000)
        print(tokenizer.batch_decode(generated_ids))


def verify_quantize_save_load():
    from mgz.models.nlp.mistral import MistralForCausalLM

    from mgz.model_running.nlp_routines.model_routine_tagging import \
        DistanceMeasure, TaggingRoutine

    model_name = 'openchat/openchat_3.5'

    with torch.cuda.amp.autocast(enabled=True):
        quantize = True
        quantization_cfg = None
        if quantize:
            try:
                from accelerate.utils import BnbQuantizationConfig
                import bitsandbytes
                quantization_cfg = BnbQuantizationConfig(
                    load_in_8bit=quantize, )
            except ImportError:
                print("Module 'some_module' is not installed.")
                quantization_cfg = None
                quantize = False
                print('verify_quantize_save_load test is skipped')
                return

        model_node: ModelNode = \
            ModelNode.load_from_id(MistralForCausalLM, model_name,
                                   model_name,
                                   quantization_config=quantization_cfg)
        val_ds = EnronEmailsTagQA(model_node.tokenizer,
                                  max_src_len=4096,
                                  n_episodes=1,
                                  n_query_per_cls=[1],
                                  n_support_per_cls=[2, 3, 4, 4, 5, 5])
        routine = TaggingRoutine(
            distance_measure=DistanceMeasure.L2,
            tokenizer=model_node.tokenizer, )
        original_metrics: Dict[
            Metrics, Union[float, List[float]]] = routine.evaluate(
            model_node=model_node, val_ds=val_ds)
        path = model_node.get_path()
        path = os.path.join(path, "test_save_load")
        model_node.store_model_node(path=path)

        model_node: ModelNode = \
            ModelNode.load_from_id(MistralForCausalLM,
                                   model_name + '/test_save_load',
                                   model_name,
                                   quantization_config=quantization_cfg)
        new_metrics: Dict[
            Metrics, Union[float, List[float]]] = routine.evaluate(
            model_node=model_node, val_ds=val_ds)
        print('original_metrics', original_metrics)
        print('new_metrics', new_metrics)
        assert (math.abs(
            original_metrics[Metrics.VAL_ACC_MEAN] - new_metrics[
                Metrics.VAL_ACC_MEAN]) < 0.1)


if __name__ == '__main__':
    start_time = time.time()
    verify_quantize_save_load()  # 16.888160467147827
    print('time taken', time.time() - start_time)
