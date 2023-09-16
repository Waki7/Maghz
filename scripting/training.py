from __future__ import annotations

# from mgz.models.nlp.bart_interface import BARTHubInterface
import os

os.putenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
import transformers as hug
import settings
from mgz.ds.sentence_datasets.synthetic_memorization import \
    SyntheticMemorization
from mgz.models.nlp.led import LEDForBinaryTagging
import logging

from mgz.ds.sentence_datasets.aus_legal_case_reports import \
    AusCaseReportsToTagGrouped
from mgz.model_running.learning_ops import LabelSmoothing
from mgz.model_running import TaggingRoutine
from mgz.version_control import ModelEdge, ModelNode

'''
input_ids tensor([[    0, 31414,   232,  1437,     2,     2]])
attention_mask None
attn_weights torch.Size([16, 6, 6])
attn_probs torch.Size([16, 6, 6])
'''


def dataset_memorization():
    return SyntheticMemorization(128, 128, 128, 1000, 1), SyntheticMemorization(
        128, 128, 128, 1000, 1)


def led_main_train():
    logging.basicConfig(level=logging.INFO)
    batch_size = 4
    # Initializing a BART facebook/bart-large style configuration
    # model_name = "facebook/bart-base"
    # model_name = 'allenai/bart-large-multi_lexsum-long-tiny'
    # model_name = 'allenai/bart-large-multi_lexsum-long-short'
    model_name = 'allenai/led-base-16384-multi_lexsum-source-long'
    print('... loading model and tokenizer')
    with torch.cuda.amp.autocast():
        model_node: ModelNode = LEDForBinaryTagging.from_pretrained(
            model_name,
            model_name)
        model_node.model.to(settings.DEVICE)
        cfg: hug.LEDConfig = model_node.model.config
        print(cfg.attention_window)
        model_node.model.train()

        dataset = AusCaseReportsToTagGrouped(model_node.tokenizer,
                                             max_src_len=cfg.max_encoder_position_embeddings)
        routine = TaggingRoutine()
        model_node = ModelNode(model_node.model, model_node.tokenizer)
        loss_fn = LabelSmoothing(
            n_cls=dataset.tgt_vocab_len(),
            padding_idx=model_node.tokenizer.pad_token_id,
            smoothing=0.1
        ).to(settings.DEVICE)
        optimizer = torch.optim.Adam(
            model_node.model.parameters(), lr=.00001, betas=(0.9, 0.98),
            eps=1e-4
        )
        train_transition_edge = ModelEdge(model_node, loss_fn, optimizer)

        routine.train(model_node=model_node, ds=dataset,
                      model_edge=train_transition_edge, batch_size=batch_size,
                      device=settings.DEVICE, distributed=False,
                      turn_off_shuffle=True)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    led_main_train()
