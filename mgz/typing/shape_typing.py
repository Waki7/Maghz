from __future__ import annotations

from typing import *

import numpy as np
import torch

#######################
# Named Shape Types
#######################
B = TypeVar("Batch")
N = TypeVar("N") # arbitrary count
NHeads = NewType("NHeads", int)  # something like n attention heads
SrcSeqLen = NewType("SeqLen", int)  # or input sequence length
OutSeqLen = NewType("OutSeqLen", int)  # or output sequence length
OutSeqStep = NewType("OutSeqStep", int)  # or output sequence length
NClasses = NewType("NClasses", int)  # or output sequence length
NBeams = NewType("NBeams", int)  # or output sequence length
NDim = NewType("NDim", int)  # or output sequence length
VocabSize = NewType("NClasses", int)  # or output sequence length
OutNClasses = NewType("OutNClasses", int)  # or output sequence length

C = NewType("Channel", int)
H = NewType("Height", int)
W = NewType("Width", int)
EmbedLen = NewType("EmbeddingSize", int)
Shape = TypeVar("Shape")
DType = TypeVar("DType")

#######################
# Named Primitive Types
#######################
ProbT = NewType("DType", float)
EnglishT = NewType("EnglishSentence", str)
GermanT = NewType("GermanSentence", str)

CaseSourceT = str  # NewType("LegalDocumentCaseSource", str)
SourceListT = List[CaseSourceT]
SummaryT = str  # NewType("SummaryOfCase", str)

SrcStringT = Union[SourceListT, GermanT, EnglishT]
TgtStringT = Union[SourceListT, GermanT, EnglishT]

Opt = Optional


StateDictT = Dict[str, torch.Tensor]

class NDArray(np.ndarray, Generic[Shape, DType]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """
    pass


class ShapedTensorT(torch.Tensor, Generic[Shape, DType]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """

    def __new__(cls, data, stats, requires_grad=False):
        data = torch.as_tensor(data, dtype=torch.float)
        tensor = torch.Tensor._make_subclass(cls, data, requires_grad)
        tensor.stats = stats
        return tensor


class TensorT(torch.Tensor, Generic[Shape]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """

    def __new__(cls, data, stats, requires_grad=False):
        data = torch.as_tensor(data, dtype=torch.float)
        tensor = torch.Tensor._make_subclass(cls, data, requires_grad)
        tensor.stats = stats
        return tensor


class FloatTensorT(torch.Tensor, Generic[Shape]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """
    def __new__(cls, data, stats, requires_grad=False):
        data = torch.as_tensor(data, dtype=torch.float)
        tensor = torch.Tensor._make_subclass(cls, data, requires_grad)
        tensor.stats = stats
        return tensor

class LongTensorT(torch.Tensor, Generic[Shape]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """

    def __new__(cls, data, stats, requires_grad=False):
        data = torch.as_tensor(data, dtype=torch.long)
        tensor = torch.Tensor._make_subclass(cls, data, requires_grad)
        tensor.stats = stats
        return tensor


class IntTensorT(torch.Tensor, Generic[Shape]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """

    def __new__(cls, data, stats, requires_grad=False):
        data = torch.as_tensor(data, dtype=torch.int)
        tensor = torch.Tensor._make_subclass(cls, data, requires_grad)
        tensor.stats = stats
        return tensor


GenericTensor = ShapedTensorT

# TensorT is meant for typing,
# torch.Tensor guides the actual passing and lint checking
#     """
#     Use this to type-annotate tensors, e.g.
#         image: SentenceT['H,W,3', np.uint8]
#         nd_mask: SentenceT['...', bool]
#     """
SentenceT = TensorT
