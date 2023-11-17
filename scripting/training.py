from __future__ import annotations

import os

from mgz.models.nlp.mistral import MistralForCausalLM

os.putenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# from mgz.models.nlp.bart_interface import BARTHubInterface


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
from mgz.version_control import ModelNode
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
                                        n_supports_per_cls=[2])
        val_ds = AusCaseReportsToTagGrouped(model_node.tokenizer,
                                            max_src_len=3000,
                                            n_episodes=100,
                                            n_queries_per_cls=[2],
                                            n_supports_per_cls=[2])
    if enron:
        ds = EnronEmailsTagQA(model_node.tokenizer,
                              max_src_len=4096,
                              n_episodes=2000,
                              n_query_per_cls=[1],
                              n_support_per_cls=[1, 2])
        val_ds = EnronEmailsTagQA(model_node.tokenizer,
                                  max_src_len=4096,
                                  n_episodes=25,
                                  n_query_per_cls=[1],
                                  n_support_per_cls=[1, 2])
    if old_enron:
        ds = EnronEmailsTagging(model_node.tokenizer,
                                max_src_len=3000,
                                n_episodes=2000,
                                n_query_per_cls=[2],
                                n_support_per_cls=[1, 2, 3])
        val_ds = EnronEmailsTagging(model_node.tokenizer,
                                    max_src_len=3000,
                                    n_episodes=25,
                                    n_query_per_cls=[2],
                                    n_support_per_cls=[1, 2, 3])
    return ds, val_ds


def led_main_train():
    logging.basicConfig(level=logging.WARNING)
    batch_size = 1
    # Initializing a BART facebook/bart-large style configuration
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
    model_name = 'openchat/openchat_3.5'
    # model_name = 'facebook/bart-large-cnn'
    # model_cls = BartForBinaryTagging
    print('... loading model and tokenizer')

    with ((torch.cuda.amp.autocast(enabled=True))):
        quantize = True
        quantization_cfg = None
        if quantize:
            try:
                from accelerate.utils import BnbQuantizationConfig
                import bitsandbytes
                # quantization_cfg = BitsAndBytesConfig(load_in_8bit=quantize)
                quantization_cfg = BnbQuantizationConfig(
                    load_in_8bit=quantize, )
            except ImportError:
                print("Module 'some_module' is not installed.")
                quantization_cfg = None

        model_node: ModelNode = \
            ModelNode.load_from_id(MistralForCausalLM, model_name,
                                   model_name,
                                   quantization_config=quantization_cfg)
        # model_node.model.to(settings.DEVICE)
        model_node.model.train()
        settings.print_gpu_usage()
        ds, val_ds = dataset_select(model_node, aus=False, enron=True)
        loss_fn = torch.nn.CrossEntropyLoss()
        # loss_fn = LabelSmoothing(
        #     n_cls=2,
        #     padding_idx=model_node.tokenizer.pad_token_id,
        #     smoothing=0.1
        # ).to(settings.DEVICE)

        lr = 0.0001
        weight_decay = 0.0001
        betas = (0.9, 0.98)
        eps = 1e-4
        if quantize:
            # settings.print_trainable_parameters(model_node.model)
            # exit(3)
            optimizer = \
                bitsandbytes.optim.Adam8bit(
                    [p for n, p in model_node.model.named_parameters() if
                     p.requires_grad],
                    lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        else:
            optimizer = \
                torch.optim.Adam(model_node.model.parameters(), lr=lr,
                                 weight_decay=weight_decay, betas=betas,
                                 eps=eps)
        # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
        train_transition_edge = ModelTransitionEdge(model_node, loss_fn,
                                                    optimizer, ds)
        TaggingRoutine().train(model_node=model_node, ds=ds, val_ds=val_ds,
                               model_edge=train_transition_edge,
                               device=settings.DEVICE, distributed=False,
                               turn_off_shuffle=False, n_epochs=15, )
        torch.cuda.empty_cache()


if __name__ == '__main__':
    led_main_train()
