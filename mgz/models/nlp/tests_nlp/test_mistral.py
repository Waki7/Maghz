from __future__ import annotations

import os
import unittest

import torch.testing

import mgz.settings as settings
from mgz.ds.sentence_datasets.enron_emails import EnronEmailsTagQA
from mgz.models.nlp.mistral import MistralForCausalLM
from mgz.version_control import Metrics, ModelNode

os.putenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from mgz.typing import *
import transformers as hug


class TestLED(unittest.TestCase):
    def recursive_check(self, obj1, obj2):
        for key, val in vars(obj1).items():
            if hasattr(val, "__dict__"):
                self.assertTrue(hasattr(obj2, key))
                self.recursive_check(val, getattr(obj2, key))
            else:
                self.assertTrue(hasattr(obj2, key))
                self.assertEqual(val, getattr(obj2, key), msg=val)

    def setUp(self):
        pass

    # ------------------ TESTING LOADING ------------------
    def get_quantized_model(self, model_id: str, tokenizer_id: str = None):

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
                self.assertTrue(True)
        model_node: ModelNode = \
            ModelNode.load_from_id(MistralForCausalLM, model_id,
                                   tokenizer_id,
                                   quantization_config=quantization_cfg)
        model_node.model.eval()
        return model_node

    def val_routine(self, model_node,
                    ) -> Dict[Metrics, Union[float, List[float]]]:

        from mgz.model_running.nlp_routines.model_routine_tagging import \
            DistanceMeasure, TaggingRoutine
        val_ds = EnronEmailsTagQA(model_node.tokenizer,
                                  max_src_len=4096,
                                  n_episodes=1,
                                  n_query_per_cls=[1],
                                  n_support_per_cls=[2, 3, 4, 4, 5, 5])
        routine = TaggingRoutine(
            distance_measure=DistanceMeasure.L2,
            tokenizer=model_node.tokenizer, )
        metrics: Dict[Metrics, Union[float, List[float]]] = routine.evaluate(
            model_node=model_node, val_ds=val_ds)
        return metrics

    @torch.no_grad()
    def test_quantize_save_load(self):
        import gc
        model_name = 'allenai/led-base-16384-multi_lexsum-source-tiny'
        test_ids = torch.ones((1, 10), dtype=torch.long).to(settings.DEVICE)
        test_mask = torch.ones((1, 10), dtype=torch.long).to(settings.DEVICE)

        with torch.cuda.amp.autocast(enabled=True):
            model_node = self.get_quantized_model(model_name, model_name)
            original_metrics = self.val_routine(model_node)
            original_embedding = model_node.model.decoder_embedding(test_ids,
                                                                    test_mask).cpu()

            path = model_node.get_path()
            path = os.path.join(path, "test_save_load")
            model_node.store_model_node(path=path)

        model_node.model = model_node.model.cpu()
        del model_node
        settings.empty_cache()
        gc.collect()
        settings.print_gpu_usage()
        with torch.cuda.amp.autocast(enabled=True):
            new_model_node = self.get_quantized_model(
                model_name + '/test_save_load', model_name)
            new_metrics = self.val_routine(new_model_node)
            new_embedding = new_model_node.model.decoder_embedding(test_ids,
                                                                   test_mask).cpu()

            print('original_embedding', original_embedding)
            print('new_embedding', new_embedding)
            print('original_metrics', original_metrics)
            print('new_metrics', new_metrics)
            self.assertTrue(abs(
                original_metrics[Metrics.VAL_ACC_MEAN] - new_metrics[
                    Metrics.VAL_ACC_MEAN]) < 0.1)

    def test_mistral_tokenizer(self):
        test_string = ['This is a test',
                       'This is another test, adding words.']

        tex_len = 10
        bsz = 2
        assert len(test_string) == bsz

        tokenizer = hug.LlamaTokenizerFast.from_pretrained(
            "mistralai/Mistral-7B-v0.1")
        config = hug.MistralConfig.from_pretrained(
            "mistralai/Mistral-7B-v0.1")
        print('pad_token_id', config.pad_token_id)
        print('sep_token_id', config.sep_token_id)
        print('sep_token_id', config.eos_token_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.sep_token_id = tokenizer.eos_token_id
        print('pad_token_id', tokenizer.pad_token_id)
        print('sep_token_id', tokenizer.sep_token_id)
        print('eos_token_id', tokenizer.eos_token_id)
        print('bos_token_id', tokenizer.bos_token_id)
        print('unk_token_id', tokenizer.unk_token_id)
        input_ids = tokenizer.__call__(test_string, return_tensors="pt",
                                       padding=True, max_length=tex_len,
                                       truncation=False).input_ids
        print(input_ids)
        if config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids,
                                             config.pad_token_id).long().argmax(
                    -1) - 1)
            else:
                sequence_lengths = -1
        print('seq len', (torch.eq(input_ids,
                                   tokenizer.pad_token_id).long().argmax(
            -1) - 1))
        print('seq len', sequence_lengths)
        logits = torch.randn(2, 10, 2)
        print(logits)
        pooled_logits = logits[
            torch.arange(2, device=logits.device), sequence_lengths]
        print(pooled_logits)

    # def test_hug_mistral(self):
    #     from transformers import LlamaTokenizerFast
    #     from mgz.models.nlp.mistral_orig import MistralForCausalLM
    #     model = MistralForCausalLM.from_pretrained(
    #         "mistralai/Mistral-7B-v0.1", use_flash_attention_2=True,
    #         torch_dtype=torch.float16, device_map={"": settings.DEVICE}, )
    #     tokenizer = LlamaTokenizerFast.from_pretrained(
    #         "mistralai/Mistral-7B-v0.1")
    #
    #     prompt = "My favourite condiment is"
    #
    #     model_inputs = tokenizer([prompt], return_tensors="pt").to(
    #         settings.DEVICE)
    #     # model.config.max_length = 3
    #     print(model.config.max_length)
    #     generated_ids = model.generate(**model_inputs, max_new_tokens=100)
    #     print(tokenizer.batch_decode(generated_ids)[0])

    @torch.no_grad()
    def test_my_mistral(self):
        from mgz.models.nlp.mistral import MistralForCausalLM
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

        with torch.cuda.amp.autocast(enabled=True):
            model_name = 'openchat/openchat_3.5'
            model_node: ModelNode = \
                ModelNode.load_from_id(MistralForCausalLM, model_name,
                                       model_name,
                                       quantization_config=quantization_cfg)
            model_node.model.eval()
            model = model_node.model
            tokenizer = model_node.tokenizer

            prompt = (
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

            model_inputs = tokenizer([prompt], return_tensors="pt").to(
                settings.DEVICE)

            # model.config.max_length = 7
            generated_ids = model.generate(src_ids=model_inputs.input_ids,
                                           src_mask=model_inputs.attention_mask,
                                           tgt_ids=model_inputs.input_ids,
                                           max_new_tokens=1000)
            print(tokenizer.batch_decode(generated_ids)[0])

    def test_hug_mistral(self):
        import os

        import mgz.settings as settings

        os.putenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        # from mgz.models.nlp.bart_interface import BARTHubInterface

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
