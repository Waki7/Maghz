from __future__ import annotations

# from mgz.models.nlp.bart_interface import BARTHubInterface
import os

from mgz.ds.sentence_datasets.enron_emails import EnronEmailsTagging
from mgz.typing import *

os.putenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
import settings
from mgz.ds.sentence_datasets.synthetic_memorization import \
    SyntheticMemorization
from mgz.models.nlp.led import LEDForBinaryTagging
import logging

from mgz.ds.sentence_datasets.aus_legal_case_reports import \
    AusCaseReportsToTagGrouped
from mgz.model_running.learning_ops import LabelSmoothing
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
                   enron: bool = False) -> Tuple[
    SentenceDataset, SentenceDataset]:
    assert aus or enron, 'must select a dataset'
    ds, val_ds = None, None
    cfg = model_node.model.config
    print(cfg.max_encoder_position_embeddings)
    if aus:
        ds = AusCaseReportsToTagGrouped(model_node.tokenizer,
                                        max_src_len=3000,
                                        n_episodes=1000, n_shot=2)
        val_ds = AusCaseReportsToTagGrouped(model_node.tokenizer,
                                            max_src_len=3000,
                                            n_episodes=100, n_shot=2)
    if enron:
        ds = EnronEmailsTagging(model_node.tokenizer,
                                max_src_len=2000,
                                n_episodes=2000,
                                n_shot=2)
        val_ds = EnronEmailsTagging(model_node.tokenizer,
                                    max_src_len=2000,
                                    n_episodes=50,
                                    n_shot=2)
    return ds, val_ds


def led_main_train():
    logging.basicConfig(level=logging.WARNING)
    batch_size = 1
    # Initializing a BART facebook/bart-large style configuration
    # model_name = "facebook/bart-base"
    # model_name = 'allenai/bart-large-multi_lexsum-long-tiny'
    # model_name = 'allenai/bart-large-multi_lexsum-long-short'
    # model_name = 'allenai/led-base-16384-multi_lexsum-source-long'
    # model_name = 'allenai/led-base-16384-multi_lexsum-source-tiny'
    model_name = 'allenai/primera-multi_lexsum-source-long/epoch4'

    # model_name = 'allenai/led-base-16384'
    # model_name = 'allenai/led-large-16384'
    # model_name = 'allenai/led-base-16384-multi_lexsum-source-long'


    # model_name = 'facebook/bart-large-cnn'
    # model_cls = BartForBinaryTagging
    print('... loading model and tokenizer')
    with torch.cuda.amp.autocast():
        model_node: ModelNode = ModelNode.load_from_id(LEDForBinaryTagging,
                                                       model_name,
                                                       model_name)
        model_node.model.to(settings.DEVICE)
        model_node.model.train()
        ds, val_ds = dataset_select(model_node, aus=False, enron=True)
        routine = TaggingRoutine()
        loss_fn = LabelSmoothing(
            n_cls=ds.tgt_vocab_len(),
            padding_idx=model_node.tokenizer.pad_token_id,
            smoothing=0.1
        ).to(settings.DEVICE)
        optimizer = torch.optim.Adam(
            model_node.model.parameters(), lr=0.0001,
            # weight_decay=0.0001,
            betas=(0.9, 0.98),
            eps=1e-4
        )
        # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
        train_transition_edge = ModelTransitionEdge(model_node, loss_fn,
                                                    optimizer, ds)
        routine.train(model_node=model_node, ds=ds, val_ds=val_ds,
                      model_edge=train_transition_edge, batch_size=batch_size,
                      device=settings.DEVICE, distributed=False,
                      turn_off_shuffle=False, n_epochs=10, )
        torch.cuda.empty_cache()


if __name__ == '__main__':
    led_main_train()
