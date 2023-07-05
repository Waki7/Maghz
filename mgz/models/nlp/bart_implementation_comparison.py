# from mgz.models.nlp.bart_interface import BARTHubInterface
import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import BartConfig

import settings
from mgz.ds.sentence_datasets.sentence_datasets import SentenceBatch
from mgz.ds.sentence_datasets.synthetic_memorization import \
    SyntheticMemorization
from mgz.ds.sentence_datasets.multi_lex_sum import \
    MultiLexSum
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
    return MultiLexSum(tokenizer,
                       max_position_embeddings).load_training_data(), MultiLexSum(
        tokenizer,
        max_position_embeddings).load_validation_data()


def main2():
    from mgz.models.nlp.bart import BartForConditionalGeneration

    # Initializing a BART facebook/bart-large style configuration
    cfg = BartConfig()
    # model_name = "facebook/bart-base"
    model_name = 'allenai/bart-large-multi_lexsum-long-tiny'
    print('... loading tokenizer')
    tokenizer = BartTokenizer.from_pretrained(model_name)

    print('... loading model')
    model = BartForConditionalGeneration.from_pretrained(model_name)
    model.to(settings.DEVICE)
    model.train()
    print('loaded')
    with torch.cuda.amp.autocast():
        # Initializing a model (with random weights) from the facebook/bart-large style configuration
        train_ds, valid_ds = dataset_legal_summary(tokenizer,
                                                   cfg.max_position_embeddings)

        train_dl = train_ds.create_dataloaders(settings.DEVICE, 8, False)
        val_dl = valid_ds.create_dataloaders(settings.DEVICE, 8, False)
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
            model.parameters(), lr=.01, betas=(0.9, 0.98),
            eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: rate(
                step, cfg.d_model, factor=1, warmup=2
            ),
        )
        _, train_state = run_epoch(
            (SentenceBatch(b[0], b[1], pad_idx) for b in train_dl),
            model,
            SimpleLossCompute(model.lm_head, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=1,
            train_state=train_state,
        )

        torch.cuda.empty_cache()

        # print(model)
        # Accessing the model configuration
        cfg = model.config

        # import torchtext
        # from torchtext.data import get_tokenizer
        # tokenizer: Language = get_tokenizer("basic_english")

        # print(model.forward(input_ids))

        # def main():
        #     from mgz.models.nlp.bart import BARTModel
        #     bart_large = "C:/Users/ceyer/.cache/torch/pytorch_fairseq/40858f8de84f479771b2807266d806749e9ad0f8cb547921c35a76ae9c3ed0f6.099ef973524a5edb31b1211569b67bcc2863bc6d00781b79bac752acf8e48991/model.pt"
        #     import torch
        #     bart = BARTModel().load_state_dict(torch.load(bart_large))
        #     print(bart)
        #     # bart: BARTHubInterface = torch.hub.load('pytorch/fairseq', 'bart.large')
        #     print(type(bart))
        #     bart.eval()  # disable dropout (or leave in train mode to finetune)
        #     tokens = bart.encode('Hello world!')
        #     assert tokens.tolist() == [0, 31414, 232, 328, 2]
        #     bart.decode(tokens)  # 'Hello world!'
        #     print(bart.decode(tokens))


if __name__ == '__main__':
    main2()
