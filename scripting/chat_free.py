import logging

import torch.cuda.amp
import transformers as hug

from mgz import settings
from mgz.ds.sentence_datasets.datasets_base.sentence_datasets import \
    prompts_to_padded_id_tensor_w_mask
from mgz.ds.sentence_datasets.gpt_input_augments import \
    FreePromptInput
from mgz.models.nlp.base_transformer import ModelType
from mgz.version_control import ModelNode

with torch.cuda.amp.autocast(enabled=True):
    with torch.no_grad():
        logging.basicConfig(level=logging.WARNING)
        model_id = 'mistralai/Mistral-7B-Instruct-v0.2'
        # model_id = "mistralai/Mistral-cont-exp/data_EnronEmailsTagQA_1e-5_lbl_fix1/BEST"
        # model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'

        # model_node = ModelDatabase.get_quantized_model(model_id)
        # model= model_node.model
        # tokenizer = model_node.tokenizer

        model = hug.MistralForCausalLM.from_pretrained(
            model_id, device_map="auto").half()
        tokenizer = hug.AutoTokenizer.from_pretrained(
            model_id, )
        tokenizer.pad_token = tokenizer.eos_token
        model.MODEL_TYPE = ModelType.DecoderTransformer
        model_node = ModelNode(model=model,
                               tokenizer=tokenizer,
                               model_id=model_id)

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

        # 12818
        print('--------------------------------------------')
        # print(email)
        # print('--------------------------------------------')
        max_src_len = 1000

        src_ids, src_mask = prompts_to_padded_id_tensor_w_mask(
            FreePromptInput(
                raw_chats=[
                    'What happens to clothes when water spills on it? D'], ),
            tokenizer=model_node.tokenizer, max_len=max_src_len,
            device=settings.DEVICE)
        print(src_ids.shape)
        print(src_mask.shape)
        generation_config: hug.GenerationConfig = model.generation_config
        generation_config.use_cache = True
        generation_config.max_new_tokens = 1000
        result = model_node.model.generate(input_ids=src_ids,
                                           attention_mask=src_mask)

        answer_start = int(src_ids.shape[-1])
        answer = model_node.tokenizer.batch_decode(result[:, answer_start:],
                                                   skip_special_tokens=True)

        print('answer', answer)

        print('original',
              model_node.tokenizer.batch_decode(result[:, :answer_start],
                                                skip_special_tokens=True))
