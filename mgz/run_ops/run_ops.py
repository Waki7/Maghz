from __future__ import annotations

import multiprocessing as mp
import os
import time

import GPUtil
import torch.distributed as dist
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

import mgz.models.metrics as metrics
import settings
from mgz.ds.sentence_datasets.multi_lex_sum import MultiLexSum
from mgz.ds.sentence_datasets.sentence_datasets import Sent2SentBatch
from mgz.ds.sentence_datasets.sentence_datasets import subsequent_mask
from mgz.models.nlp.base_transformer import BaseTransformer
from mgz.models.nlp.bert_basic import make_model
from mgz.run_ops.learning_ops import LabelSmoothing, DummyOptimizer, \
    DummyScheduler, SimpleLossCompute, rate
from mgz.typing import *


class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed


def run_epoch(
        data_loader: torch.utils.data.DataLoader[Sent2SentBatch],
        val_data_loader: torch.utils.data.DataLoader[Sent2SentBatch],
        model: BaseTransformer,
        tokenizer: PreTrainedTokenizerBase,
        loss_compute,
        optimizer,
        scheduler=None,
        accum_iter=1,
        train_state=TrainState(),
        log_interval=40,
        accuracy_interval=120,
) -> (torch.Tensor, TrainState):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    i: int
    batch: Sent2SentBatch
    with torch.no_grad():
        val_model(val_data_loader, model, tokenizer)

    for i, batch in tqdm(enumerate(data_loader)):
        logits = model.forward(
            src_ids=batch.src, tgt_ids=batch.tgt, src_mask=batch.src_mask,
            tgt_mask=batch.tgt_mask
        )

        loss, loss_node = loss_compute(logits, batch.tgt, batch.ntokens)
        # loss_node = loss_node / accum_iter
        loss_node.backward()
        train_state.step += 1
        train_state.samples += batch.src.shape[0]
        train_state.tokens += batch.ntokens
        if i % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_accum += 1
            train_state.accum_step += 1
        if scheduler is not None: scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % log_interval == 1:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        if i % accuracy_interval == 1:
            with torch.no_grad():
                settings.empty_cache()
                val_model(val_data_loader, model, tokenizer)
        # cuda / mps / gpu usage and managing
        settings.empty_cache()
        settings.print_gpu_usage()
    return total_loss / total_tokens, train_state


def val_model(
        val_data_loader: torch.utils.data.DataLoader[Sent2SentBatch],
        model: BaseTransformer,
        tokenizer: PreTrainedTokenizerBase
) -> (torch.Tensor, TrainState):
    started_training = model.training
    model.eval()

    if started_training:
        model.train()
    i: int
    batch: Sent2SentBatch
    n_samples = 0
    # TODO, MAKE SURE THAT BLEU SCORES ADJACENTLY, CHECK BY EITHER VARYING THE BATCH SIZE AND MAKIGN SURE SAME SCORE, OR SHUFFLYING
    for i, batch in enumerate(val_data_loader):
        # TgtSeqLen is max padded length
        predictions: LongTensorT['B,TgtSeqLen'] = model.generate(
            src_ids=batch.src, src_mask=batch.src_mask)
        prediction_sentences: List[str] = tokenizer.batch_decode(predictions,
                                                                 skip_special_tokens=True)
        print(
            (
                "Validation BLEU Score: %6d"
            )
            % (metrics.bleu_from_batch_decode(prediction_sentences, batch.tgt)
               )
        )


def embedding_controller(model: BaseTransformer, text: List[str],
                         tokenizer: PreTrainedTokenizerBase) -> FloatTensorT[
    'B,EmbeddingDim']:
    batch_encoding = tokenizer(text, return_tensors="pt")
    src_ids = batch_encoding.input_ids.to(settings.DEVICE)
    src_mask = batch_encoding.attention_mask.to(settings.DEVICE)
    return model.encode(src_ids=src_ids, src_mask=src_mask).mean(1)


def forward_controller(model: BaseTransformer, text: List[str],
                       tokenizer: PreTrainedTokenizerBase):
    batch_encoding = tokenizer(text, return_tensors="pt")
    src_ids = batch_encoding.input_ids.to(settings.DEVICE)
    batch_size = len(text)
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


def generate_controller(model: BaseTransformer, text: List[str],
                        tokenizer: PreTrainedTokenizerBase,
                        ):
    batch_size = len(text)
    batch_encoding = tokenizer(text, return_tensors="pt")
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


def train_worker(
        gpu: Union[torch.device, int],
        ngpus_per_node,
        config,
        is_distributed=False,
):
    print(f"Train worker process using GPU: {gpu} for training", flush=True)
    torch.cuda.set_device(gpu)

    train_ds = MultiLexSum(config["max_padding"]).load_training_data()
    valid_ds = MultiLexSum(config["max_padding"]).load_validation_data()

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


def load_trained_model(config):
    # config = {
    #     "batch_size": 32,
    #     "distributed": False,
    #     "num_epochs": 8,
    #     "accum_iter": 10,
    #     "base_lr": 1.0,
    #     "max_padding": 72,
    #     "warmup": 3000,
    #     "file_prefix": "multi30k_model_",
    # }
    model_path = "C:/Users/ceyer/OneDrive/Documents/Projects/Maghz/index_dir/model_storage/multi30k_model_final.pt"
    # if not exists(model_path):
    #     train_model(config)

    ds = MultiLexSum(config["max_padding"])
    model = make_model(len(ds.vocab_src), len(ds.vocab_tgt), N=6)
    print(model)
    model.load_state_dict(torch.load(model_path))
    return model


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
        # load_trained_model(config)


if __name__ == '__main__':
    main()
