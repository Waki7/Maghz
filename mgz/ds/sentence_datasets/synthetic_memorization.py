# from functools import partial
#
# import mgz.settings as settings
# import spaces as sp
# from mgz.ds.sentence_datasets.sentence_datasets import SentenceDataset
# from mgz.typing import *
#
#
# class SyntheticMemorization(SentenceDataset):
#     def __init__(self, vocab_size: int, out_vocab_size: int, max_length: int,
#                  n_samples: int, batch_size: int):
#         super(SyntheticMemorization, self).__init__()
#         self.input_space = sp.Sentence(vocab_size, sequence_len=(max_length,))
#         self.target_space = sp.Sentence(out_vocab_size, sequence_len=(max_length,))
#
#         self.max_length = max_length
#         self.n_samples = n_samples
#         self.n_batches = n_samples // batch_size
#         self.batch_size = batch_size
#
#     def __len__(self):
#         'Denotes the total number of samples'
#         return self.n_samples
#
#     def gen(self) -> Generator[Sent2SentBatch, None, None]:
#         for i in range(self.batch_size):
#             data = torch.randint(1, self.input_space.vocab_size,
#                                  size=(self.batch_size, self.max_length))
#             data[:, 0] = 1  # starting token
#             src = data.requires_grad_(False).clone().detach()
#             tgt = data[:, :-1].requires_grad_(False).clone().detach()
#             if self.use_cuda:
#                 src = src.to(settings.DEVICE)
#                 tgt = tgt.to(settings.DEVICE)
#             batch = Sent2SentBatch(src, tgt, 0)
#             yield batch
#
#     def __getitem__(self, index) -> Sent2SentBatch:
#         '''
#         Generates one sample of data
#
#         :param index:
#         :return: an image and a regression target, the regression target is 5 long, as per the __init__ def of the space
#         '''
#         batch: Sent2SentBatch = next(self.gen())
#         return batch
#
#     def _collate_fn(self, device: Union[int, torch.device],
#                     batch: List[Sent2SentBatch]) -> Tuple[
#         LongTensorT['B,SrcSeqLen'], LongTensorT['B,TgtSeqLen']]:
#         assert self.loaded, "Dataset not loaded"
#         func = torch.cat if len(batch[0].src.shape) == 2 else torch.stack
#         src = func([b.src for b in batch])
#         tgt = func([b.tgt for b in batch])
#         print('src', src.shape)
#         return src, tgt
#
#     def get_collate_fn(self, device: Union[int, torch.device]):
#         assert self.loaded, "Dataset not loaded"
#         return partial(self._collate_fn, device)
#
#     def pad_idx(self) -> int:
#         return -1
#
#     def src_vocab_len(self) -> int:
#         return self.input_space.vocab_size
#
#     def tgt_vocab_len(self) -> int:
#         return self.target_space.vocab_size
