# from mgz.models.nlp.bart_interface import BARTHubInterface
import os

os.putenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
from transformers import BartConfig
import settings
from mgz.ds.sentence_datasets.synthetic_memorization import \
    SyntheticMemorization
from mgz.models.nlp.bart import BartForConditionalGeneration
import logging
from mgz.ds.sentence_datasets.multi_lex_sum import \
    MultiLexSumLongToShort
from mgz.model_running.learning_ops import LabelSmoothing
from mgz.model_running import SummarizationRoutine
from mgz.model_vc import ModelEdge, ModelNode

'''
input_ids tensor([[    0, 31414,   232,  1437,     2,     2]])
attention_mask None
attn_weights torch.Size([16, 6, 6])
attn_probs torch.Size([16, 6, 6])
'''


def dataset_memorization():
    return SyntheticMemorization(128, 128, 128, 1000, 1), SyntheticMemorization(
        128, 128, 128, 1000, 1)


def main2():
    logging.basicConfig(level=logging.INFO)
    batch_size = 4
    # Initializing a BART facebook/bart-large style configuration
    # model_name = "facebook/bart-base"
    # model_name = 'allenai/bart-large-multi_lexsum-long-tiny'
    model_name = 'allenai/bart-large-multi_lexsum-long-short'
    print('... loading model and tokenizer')
    with torch.cuda.amp.autocast():
        model, tokenizer = BartForConditionalGeneration.from_pretrained(
            model_name,
            model_name)
        model.to(settings.DEVICE)
        cfg: BartConfig = model.config
        model.train()

        dataset = MultiLexSumLongToShort(tokenizer,
                                         cfg.max_position_embeddings)
        routine = SummarizationRoutine()
        model_node = ModelNode(model, tokenizer)
        loss_fn = LabelSmoothing(
            n_cls=dataset.tgt_vocab_len(), padding_idx=tokenizer.pad_token_id,
            smoothing=0.1
        ).to(settings.DEVICE)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=.00001, betas=(0.9, 0.98),
            eps=1e-4
        )
        train_transition_edge = ModelEdge(model_node, loss_fn, optimizer)

        routine.train(model_node=model_node, ds=dataset,
                      edge=train_transition_edge, batch_size=batch_size,
                      device=settings.DEVICE, distributed=False,
                      turn_off_shuffle=True)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main2()
