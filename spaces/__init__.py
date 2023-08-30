from spaces.hierarchical import Hierarchical
from spaces.action_node import DiscreteNode
from spaces.image import Image
from spaces.sentence import SentenceT
from spaces.tagging import Tagging, BinaryTagging
from spaces.generic_box import GenericBox
from spaces.regression_target import *
from gym.spaces.space import Space
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.multi_binary import MultiBinary
from gym.spaces.tuple import Tuple
from gym.spaces.dict import Dict

from spaces.utils import flatdim
from spaces.utils import space_to_out_shapes
from spaces.utils import space_to_network_shapes
from spaces.utils import space_to_out_space
from gym.spaces.utils import flatten
from gym.spaces.utils import unflatten
