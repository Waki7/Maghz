from __future__ import annotations

from typing import *

import numpy as np
import torch

##############################################
#######################
# Named Shape Types, Defines Dimensions
#######################
##############################################
B = TypeVar("Batch")
N = TypeVar("N")  # arbitrary count
NHeads = TypeVar(
    "NHeads")  # NewType("NHeads", int)  # something like n attention heads
SrcSeqLen = TypeVar(
    "SrcSeqLen")  # NewType("SeqLen", int)
TgtSeqLen = TypeVar("TgtSeqLen")  # NewType("TgtSeqLen", int)
TgtSeqStep = TypeVar("TgtSeqStep")  # NewType("TgtSeqStep", int)
NClasses = TypeVar("NClasses")  # NewType("NClasses", int)
NBeams = TypeVar("NBeams")  # NewType("NBeams", int)
NDim = TypeVar("NDim")  # NewType("NDim", int)
VocabSize = TypeVar("VocabSize")  # NewType("VocabSize", int)
OutNClasses = TypeVar("OutNClasses")  # NewType("OutNClasses", int)

# META LEARNING
TaskSize = TypeVar("NShot")
NQuery = TypeVar("NQuery")
NSupport = TypeVar("NSupport")

C = TypeVar("C")  # NewType("Channel", int)
H = TypeVar("H")  # NewType("Height", int)
W = TypeVar("W")  # NewType("Width", int)
Stride = TypeVar("Stride")  # NewType("Stride", int)
EmbedLen = TypeVar("EmbedLen")  # NewType("EmbeddingSize", int)
Shape = TypeVar("Shape")
DType = TypeVar("DType")

##############################################
#######################
# Named Primitive Types, Defines Data Type
#######################
##############################################
ProbT = NewType("DType", float)
EnglishT = NewType("EnglishSentence", str)
GermanT = NewType("GermanSentence", str)

CaseSourceT = str  # NewType("LegalDocumentCaseSource", str)
SourceListT = List[CaseSourceT]
SummaryT = str  # NewType("SummaryOfCase", str)

SentenceT = str
TokenT = str
SrcTokenT = TokenT
TgtTokenT = TokenT

SrcStringT = str  # Union[SourceListT, GermanT, EnglishT]  # untokenized
TgtStringT = str  # Union[SourceListT, GermanT, EnglishT]  # untokenized
LabelT = int  # NewType("Label", int)

Opt = Optional

StateDictT = Dict[str, torch.Tensor]


class IntListT(List, Generic[Shape, DType]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """

    def __new__(cls, tensor, shape: Generic[Shape] = None):
        # todo assert shape
        return tensor


class NDArrayT(np.ndarray, Generic[Shape, DType]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """

    def __new__(cls, tensor, shape: Generic[Shape] = None):
        # todo assert shape
        return tensor


class NDFloatArrayT(np.ndarray, Generic[Shape, DType]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """

    def __new__(cls, tensor, shape: Generic[Shape] = None):
        # todo assert shape
        return tensor


class NDLogitsArrayT(np.ndarray, Generic[Shape, DType]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """

    def __new__(cls, tensor, shape: Generic[Shape] = None):
        # todo assert shape
        return tensor


class NDProbArrayT(np.ndarray, Generic[Shape, DType]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """

    def __new__(cls, tensor, shape: Generic[Shape] = None):
        # todo assert shape
        return tensor


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

    def __new__(cls, tensor, shape: Generic[Shape] = None):
        # todo assert shape
        return torch.as_tensor(tensor).float()


class ProbTensorT(FloatTensorT, Generic[Shape]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """

    def __new__(cls, tensor, shape: Generic[Shape] = None):
        # todo assert shape
        return torch.as_tensor(tensor)


class LogitsTensorT(FloatTensorT, Generic[Shape]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """

    def __new__(cls, tensor, shape: Generic[Shape] = None):
        # todo assert shape
        return torch.as_tensor(tensor)


class LongTensorT(torch.Tensor, Generic[Shape]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """

    def __new__(cls, tensor, shape: Generic[Shape] = None):
        # todo assert shape
        return torch.as_tensor(tensor)


class IntTensorT(torch.Tensor, Generic[Shape]):
    """
    Use this to type-annotate numpy arrays, e.g.
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """

    def __new__(cls, tensor, shape: Generic[Shape] = None):
        # todo assert shape
        return torch.as_tensor(tensor)


GenericTensor = ShapedTensorT

# TensorT is meant for typing,
# torch.Tensor guides the actual passing and lint checking
#     """
#     Use this to type-annotate tensors, e.g.
#         image: SentenceT['H,W,3', np.uint8]
#         nd_mask: SentenceT['...', bool]
#     """
