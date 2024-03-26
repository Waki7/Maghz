from __future__ import annotations


class SentenceBatch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def collate_batch(*args, **kwargs) -> SentenceBatch:
        raise NotImplementedError

    @staticmethod
    def default_collate_fn(*args, **kwargs):
        raise NotImplementedError
