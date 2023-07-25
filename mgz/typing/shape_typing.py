from __future__ import annotations

from typing import *

import numpy as np
import torch

#######################
# Named Shape Types
#######################
B = TypeVar("Batch")
N = TypeVar("N")  # arbitrary count
NHeads = int  # NewType("NHeads", int)  # something like n attention heads
SrcSeqLen = int  # NewType("SeqLen", int)  # or input sequence length
TgtSeqLen = int  # NewType("OutSeqLen", int)  # or output sequence length
TgtSeqStep = int  # NewType("OutSeqStep", int)  # or output sequence length
NClasses = int  # NewType("NClasses", int)  # or output sequence length
NBeams = int  # NewType("NBeams", int)  # or output sequence length
NDim = int  # NewType("NDim", int)  # or output sequence length
VocabSize = int  # NewType("NClasses", int)  # or output sequence length
OutNClasses = int  # NewType("OutNClasses", int)  # or output sequence length

C = int  # NewType("Channel", int)
H = int  # NewType("Height", int)
W = int  # NewType("Width", int)
EmbedLen = int  # NewType("EmbeddingSize", int)
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
SrcTextT = str  # NewType("SourceText", str)
TgtTextT = str  # NewType("SourceText", str)
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
