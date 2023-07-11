# from mgz.models.nlp.bart_interface import BARTHubInterface
import os

os.putenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import BartConfig
import settings
from mgz.ds.sentence_datasets.sentence_datasets import SentenceBatch
from mgz.ds.sentence_datasets.synthetic_memorization import \
    SyntheticMemorization
from mgz.models.nlp.bart import BartForConditionalGeneration
import logging
from mgz.ds.sentence_datasets.multi_lex_sum import \
    MultiLexSumLongToTiny
from mgz.run_ops.learning_ops import LabelSmoothing, SimpleLossCompute, rate
from mgz.run_ops.run_ops import TrainState
from mgz.run_ops.run_ops import run_epoch
from transformers import BartTokenizer, PreTrainedTokenizer

'''
input_ids tensor([[    0, 31414,   232,  1437,     2,     2]])
attention_mask None
attn_weights torch.Size([16, 6, 6])
attn_probs torch.Size([16, 6, 6])
'''


def dataset_memorization():
    return SyntheticMemorization(128, 128, 128, 1000, 1), SyntheticMemorization(
        128, 128, 128, 1000, 1)


def dataset_legal_summary(tokenizer: PreTrainedTokenizer,
                          max_position_embeddings):
    return MultiLexSumLongToTiny(tokenizer,
                                 max_position_embeddings).load_training_data(), \
        MultiLexSumLongToTiny(tokenizer,
                              max_position_embeddings).load_validation_data()


def main2():
    logging.basicConfig(level=logging.INFO)
    batch_size = 4
    # Initializing a BART facebook/bart-large style configuration
    # model_name = "facebook/bart-base"
    model_name = 'allenai/bart-large-multi_lexsum-long-tiny'
    print('... loading model and tokenizer')
    with torch.cuda.amp.autocast():
        model, tokenizer = BartForConditionalGeneration.from_pretrained(
            model_name,
            model_name)
        model.to(settings.DEVICE)
        cfg: BartConfig = model.config
        print('max position embeddings: ', cfg.max_position_embeddings)
        print('vocab size: ', cfg.vocab_size)
        print('d_moel: ', cfg.d_model)
        model.train()
        print('loaded')
        # Initializing a model (with random weights) from the facebook/bart-large style configuration
        train_ds, valid_ds = dataset_legal_summary(tokenizer,
                                                   cfg.max_position_embeddings)

        train_dl = train_ds.create_dataloaders(settings.DEVICE, batch_size,
                                               False, turn_off_shuffle=True)
        val_dl = valid_ds.create_dataloaders(settings.DEVICE, batch_size, False,
                                             turn_off_shuffle=True)
        print('loaded data')
        pad_idx = train_ds.pad_idx()

        cfg.vocab_size = max(train_ds.src_vocab_len(),
                             train_ds.tgt_vocab_len())
        train_state = TrainState()
        criterion = LabelSmoothing(
            n_cls=train_ds.tgt_vocab_len(), padding_idx=pad_idx,
            smoothing=0.1
        )
        # criterion.cuda()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=.00001, betas=(0.9, 0.98),
            eps=1e-4
        )
        lr_scheduler = None
        #     LambdaLR(
        #     optimizer=optimizer,
        #     lr_lambda=lambda step: rate(
        #         step, cfg.d_model, factor=1, warmup=2
        #     ),
        # )
        print(len(train_ds))
        _, train_state = run_epoch(
            train_dl,
            val_dl,
            model, tokenizer,
            SimpleLossCompute(model.lm_head, criterion),
            optimizer,
            lr_scheduler,
            accum_iter=1,
            train_state=train_state,
        )

        torch.cuda.empty_cache()

        # print(model)
        # Accessing the model configuration
        cfg = model.config


if __name__ == '__main__':
    main2()
