# from mgz.models.nlp.bart_interface import BARTHubInterface
import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import BartConfig

import settings
from mgz.ds.sentence_datasets.multi_lex_sum import \
    MultiLexSum
from mgz.ds.sentence_datasets.sentence_datasets import SentenceBatch
from mgz.run_ops.learning_ops import LabelSmoothing, SimpleLossCompute, rate
from mgz.run_ops.run_ops import TrainState

'''
input_ids tensor([[    0, 31414,   232,  1437,     2,     2]])
attention_mask None
attn_weights torch.Size([16, 6, 6])
attn_probs torch.Size([16, 6, 6])
'''

from mgz.run_ops.run_ops import create_dataloaders


def main2():
    from transformers import BartTokenizer
    from mgz.run_ops.run_ops import run_epoch
    use_mgz = True
    # Initializing a BART facebook/bart-large style configuration
    cfg = BartConfig()
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    input_ids = tokenizer("Hello world </s>",
                          return_tensors="pt").input_ids  # Batch size 1

    output_ids = tokenizer("Bye world </s>",
                           return_tensors="pt").input_ids  # Batch size 1
    if not use_mgz:
        # from transformers import BartModel
        from bart_orig import BartModel, BartForConditionalGeneration
        input_ids = tokenizer("Hello world </s>",
                              return_tensors="pt").input_ids  # Batch size 1
        # Initializing a model (with random weights) from the facebook/bart-large style configuration
        model = BartForConditionalGeneration(cfg)
        # model.forward(input_ids)
        model.generate(input_ids)
    else:
        from mgz.models.nlp.bart import BartForConditionalGeneration
        input_ids = tokenizer("Hello world </s>",
                              return_tensors="pt").input_ids  # Batch size 1
        # Initializing a model (with random weights) from the facebook/bart-large style configuration

        valid_ds = MultiLexSum(
            cfg.max_position_embeddings).load_validation_data()

        # train_ds = MultiLexSum(
        #     cfg.max_position_embeddings).load_training_data()
        train_ds = valid_ds

        train_dl, val_dl = create_dataloaders(
            train_ds, valid_ds, torch.device('cpu'),
            batch_size=10,
            is_distributed=False,
        )
        pad_idx = train_ds.vocab_tgt["<blank>"]

        cfg.vocab_size = max(len(train_ds.vocab_src), len(train_ds.vocab_tgt))
        model = BartForConditionalGeneration(cfg)
        # model.to(settings.DEVICE)
        model.train()
        train_state = TrainState()
        criterion = LabelSmoothing(
            n_cls=len(train_ds.vocab_tgt), padding_idx=pad_idx, smoothing=0.1
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

    print(tokenizer.__call__(" Hello world"))
    print(tokenizer.__call__("Hello world"))

    print(tokenizer.tokenize("Hello world </s>"))
    print(tokenizer.__call__("Hello world </s>"))
    print(tokenizer.__call__("Hello world asdf"))
    print(tokenizer.__call__("Hello world <pad>>"))

    input_ids = tokenizer("Hello world </s>",
                          return_tensors="pt").input_ids  # Batch size 1
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
