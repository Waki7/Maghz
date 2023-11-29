from __future__ import annotations

import multiprocessing as mp
import os

import GPUtil
import torch.distributed as dist
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from transformers import PreTrainedTokenizerBase

import mgz.settings as settings
from mgz.ds.sentence_datasets.sentence_datasets import Sent2SentBatch, \
    SentenceDataset
from mgz.ds.sentence_datasets.sentence_datasets import subsequent_mask
from mgz.model_running.base_routine import run_epoch, TrainState
from mgz.model_running.learning_ops import LabelSmoothing, DummyOptimizer, \
    DummyScheduler, SimpleLossCompute, rate
from mgz.models.nlp.base_transformer import BaseTransformer
from archive.models.bert_basic import make_model
from mgz.models.nlp.led import LEDForConditionalGeneration
from mgz.typing import *

from mgz.models.nlp.base_transformer import BaseTransformer, \
    EncoderDecoderTransformer, DecoderTransformer


@torch.no_grad()
def embedding_controller(model: BaseTransformer, texts: List[str],
                         tokenizer: PreTrainedTokenizerBase) -> FloatTensorT[
    'B,EmbedLen']:
    batch_encoding = tokenizer.__call__(texts, padding=True, truncation=True,
                                        return_tensors='pt')
    src_ids = batch_encoding.input_ids.to(settings.DEVICE)
    src_mask = batch_encoding.attention_mask.to(settings.DEVICE)

    batch_size = len(texts)
    tgt_ids = torch.LongTensor([tokenizer.sep_token_id]).unsqueeze(0).to(
        settings.DEVICE).repeat(batch_size, 1)
    tgt_mask = (tgt_ids != tokenizer.pad_token_id).unsqueeze(-2)

    if isinstance(model, EncoderDecoderTransformer):
        embedding: FloatTensorT[
            'TaskSize,EmbedLen'] = model.encoder_decoder_embedding(
            src_ids=src_ids,
            tgt_ids=tgt_ids,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
    elif isinstance(model, DecoderTransformer):
        embedding: FloatTensorT[
            'TaskSize,EmbedLen'] = model.decoder_embedding(
            src_ids=src_ids,
            src_mask=src_mask,
        )
    else:
        raise NotImplementedError
    return embedding


def forward_controller(model: BaseTransformer, texts: List[str],
                       tokenizer: PreTrainedTokenizerBase):
    batch_encoding = tokenizer(texts, return_tensors="pt")
    src_ids = batch_encoding.input_ids.to(settings.DEVICE)
    batch_size = len(texts)
    tgt_ids = torch.LongTensor([tokenizer.sep_token_id]).unsqueeze(0).to(
        settings.DEVICE).repeat(batch_size, 1)
    src_mask = batch_encoding.attention_mask.to(settings.DEVICE)
    tgt_mask = (tgt_ids != tokenizer.pad_token_id).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt_ids.size(-1)).type_as(
        tgt_mask.data
    )
    # don't need tgt_mask because you are generating one token at a time
    return model.forward(src_ids=src_ids, tgt_ids=tgt_ids,
                         src_mask=src_mask, tgt_mask=tgt_mask)


@torch.no_grad()
def generate_controller(model: BaseTransformer, texts: List[str],
                        tokenizer: PreTrainedTokenizerBase,
                        ):
    batch_size = len(texts)
    batch_encoding = tokenizer.__call__(texts, padding=True, truncation=True,
                                        return_tensors='pt')
    src_ids: LongTensorT['B,SrcSeqLen'] = batch_encoding.input_ids.to(
        settings.DEVICE)
    # bart for summarization uses sep token for the first token in the decoder,
    # I'm not sure this is true for all models
    tgt_ids = torch.LongTensor([tokenizer.sep_token_id]).unsqueeze(0).to(
        settings.DEVICE).repeat(batch_size, 1)
    src_mask = batch_encoding.attention_mask.to(settings.DEVICE)
    src_ids, src_mask = model._pre_encode_pad_if_needed(src_ids, src_mask,
                                                        tokenizer.pad_token_id)
    # don't need tgt_mask because you are generating one token at a time
    return model.generate(src_ids=src_ids, tgt_ids=tgt_ids,
                          src_mask=src_mask)


def tagging_embedding_controller(model: LEDForConditionalGeneration,
                                 text: List[str],
                                 tag_text: List[str],
                                 tokenizer: PreTrainedTokenizerBase,
                                 max_src_len: int = None,
                                 max_tgt_len: int = None
                                 ) -> FloatTensorT['B,EmbedLen']:
    if max_src_len is None:
        max_src_len = model.get_max_encoder_positions()
    if max_tgt_len is None:
        max_tgt_len = model.get_max_decoder_positions()

    src_encodings = tokenizer(text, return_tensors="pt",
                              max_length=max_src_len, truncation=True)
    src_ids: LongTensorT['B,SrcSeqLen'] = src_encodings.input_ids.to(
        settings.DEVICE)
    src_mask = src_encodings.attention_mask.to(settings.DEVICE)

    tgt_encodings = tokenizer(tag_text, return_tensors="pt",
                              max_length=max_tgt_len, truncation=True)
    tgt_ids: LongTensorT['EmbedLen'] = tgt_encodings.input_ids.to(
        settings.DEVICE)
    tgt_mask = tgt_encodings.attention_mask.to(settings.DEVICE)

    # don't need tgt_mask because you are generating one token at a time
    return model.forward(src_ids=src_ids, tgt_ids=tgt_ids,
                         src_mask=src_mask, tgt_mask=tgt_mask)


def train_worker(
        gpu: Union[torch.device, int],
        ngpus_per_node,
        config,
        is_distributed=False,
):
    from mgz.ds.sentence_datasets.multi_lex_sum import MultiLexSum
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    train_ds: SentenceDataset = MultiLexSum(
        config["max_padding"]).load_training_data()
    valid_ds: SentenceDataset = MultiLexSum(
        config["max_padding"]).load_validation_data()

    train_dataloader = train_ds.create_dataloaders(gpu, config[
        "batch_size"] // ngpus_per_node, is_distributed)
    valid_dataloader = valid_ds.create_dataloaders(gpu, config[
        "batch_size"] // ngpus_per_node, is_distributed)

    pad_idx = train_ds.vocab_tgt["<blank>"]
    d_model = 512
    model = make_model(len(train_ds.vocab_src), len(train_ds.vocab_tgt), N=6,
                       max_seq_len=config["max_padding"])
    model.cuda(gpu).half()

    module = model
    is_main_process = True
    if is_distributed:
        dist.init_process_group(
            "nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node
        )
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(
        n_cls=len(train_ds.vocab_tgt), padding_idx=pad_idx, smoothing=0.1
    )
    criterion.cuda(gpu)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, d_model, factor=1, warmup=config["warmup"]
        ),
    )
    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        print(f"[GPU{gpu}] Epoch {epoch} Training ====", flush=True)
        _, train_state = run_epoch(
            (Sent2SentBatch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        GPUtil.showUtilization()
        if is_main_process:
            file_path = "%s%.2d.pt" % (config["file_prefix"], epoch)
            torch.save(module.state_dict(), file_path)
        torch.cuda.empty_cache()

        print(f"[GPU{gpu}] Epoch {epoch} Validation ====", flush=True)
        model.eval()
        sloss = run_epoch(
            (Sent2SentBatch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )
        print(sloss)
        torch.cuda.empty_cache()

    if is_main_process:
        file_path = "%sfinal.pt" % config["file_prefix"]
        torch.save(module.state_dict(), file_path)


def train_distributed_model(config):
    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    print(f"Number of GPUs detected: {ngpus}")
    print("Spawning training processes ...")
    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, config, True),
    )


def train_model(config: Dict):
    if config["distributed"]:
        train_distributed_model(config)
    else:
        train_worker(0, 1, config, False)


def main():
    # temp config
    config = {'batch_size': 1, 'max_padding': 4096, 'base_lr': .01,
              'warmup': 2,
              'num_epochs': 10, 'accum_iter': 1,
              'file_prefix': 'C:/Users/ceyer/OneDrive/Documents/Projects/Maghz/index_dir/model_storage/',
              'distributed': False}
    with torch.cuda.amp.autocast():
        train_model(config)


if __name__ == '__main__':
    main()
