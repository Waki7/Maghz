from __future__ import annotations

import multiprocessing as mp
import os
import time

import GPUtil
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mgz.ds.sentence_datasets.multi_lex_sum import MultiLexSum
from mgz.ds.sentence_datasets.sentence_datasets import SentenceBatch, \
    SentenceDataset
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
        data_generator: Generator[SentenceBatch, None, None],
        model: BaseTransformer,
        loss_compute,
        optimizer,
        scheduler,
        mode="train",
        accum_iter=1,
        train_state=TrainState(),
) -> (torch.Tensor, TrainState):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    i: int
    batch: SentenceBatch
    for i, batch in enumerate(data_generator):
        print('i: ', i)
        logits = model.forward(
            src_ids=batch.src, tgt_ids=batch.tgt, src_mask=batch.src_mask,
            tgt_mask=batch.tgt_mask
        )
        loss, loss_node = loss_compute(logits, batch.tgt_y, batch.ntokens)
        # loss_node = loss_node / accum_iter
        if mode == "train" or mode == "train+log":
            loss_node.backward()
            train_state.step += 1
            train_state.samples += batch.src.shape[0]
            train_state.tokens += batch.ntokens
            if i % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                n_accum += 1
                train_state.accum_step += 1
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 40 == 1 and (mode == "train" or mode == "train+log"):
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
        del loss
        del loss_node
    return total_loss / total_tokens, train_state


def create_dataloaders(train_ds: SentenceDataset, val_ds: SentenceDataset,
                       device: Union[torch.device, int],
                       batch_size: int = 12000,
                       is_distributed: bool = True,
                       ) -> (DataLoader, DataLoader):
    train_sampler = (
        DistributedSampler(train_ds) if is_distributed else None
    )
    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=train_ds.get_collate_fn(device)
    )
    valid_sampler = (
        DistributedSampler(val_ds) if is_distributed else None
    )
    valid_dataloader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=val_ds.get_collate_fn(device)
    )
    return train_dataloader, valid_dataloader


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
    train_dataloader, valid_dataloader = create_dataloaders(
        train_ds, valid_ds, gpu,
        batch_size=config["batch_size"] // ngpus_per_node,
        is_distributed=is_distributed,
    )

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
            (SentenceBatch(b[0], b[1], pad_idx) for b in train_dataloader),
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
            (SentenceBatch(b[0], b[1], pad_idx) for b in valid_dataloader),
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
    exit(4)
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
