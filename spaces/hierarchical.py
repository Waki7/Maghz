from collections import OrderedDict
from typing import *

import numpy as np
import torch
from gym.spaces import Space, MultiDiscrete, Discrete


class Hierarchical(Space):
    """
    A dictionary of simpler spaces.

    Example usage:
    self.observation_space = spaces.Dict({"position": spaces.Discrete(2), "velocity": spaces.Discrete(3)})

    Example usage [nested]:
    self.nested_observation_space = spaces.Dict({
        'sensors':  spaces.Dict({
            'position': spaces.Box(low=-100, high=100, shape=(3,)),
            'velocity': spaces.Box(low=-1, high=1, shape=(3,)),
            'front_cam': spaces.Tuple((
                spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                spaces.Box(low=0, high=1, shape=(10, 10, 3))
            )),
            'rear_cam': spaces.Box(low=0, high=1, shape=(10, 10, 3)),
        }),
        'ext_controller': spaces.MultiDiscrete((5, 2, 2)),
        'inner_state':spaces.Dict({
            'charge': spaces.Discrete(100),
            'system_checks': spaces.MultiBinary(10),
            'job_status': spaces.Dict({
                'task': spaces.Discrete(5),
                'progress': spaces.Box(low=0, high=100, shape=()),
            })
        })
    })
    """

    def __init__(self, spaces: Dict[Any, Union[Space, Dict]] = None,
                 **spaces_kwargs):
        assert isinstance(spaces, OrderedDict)
        assert (spaces is None) or (
                not spaces_kwargs), \
            'Use either Dict(spaces=dict(...)) or Dict(foo=x, bar=z)'
        if spaces is None:
            spaces = spaces_kwargs
        if isinstance(spaces, dict) and not isinstance(spaces, OrderedDict):
            spaces = OrderedDict(sorted(list(spaces.items())))
        if isinstance(spaces, list):
            spaces = OrderedDict(spaces)
        self.spaces: Dict[Any, Union[Space, Dict]] = spaces
        self.function_to_param_spaces: Dict[Any, List[Space]] = {}

        function_n = len(spaces)
        parameter_nvec = []

        # for key, space_dict in spaces.items():
        #     param_spaces = list(space_dict.values())
        #     self.function_to_param_spaces[key] = param_spaces
        #     for space in param_spaces:
        #         parameter_nvec.append(space.n)
        #         assert isinstance(space, Space), \
        #             'Values of the dict should be instances of gym.Space'

        self.function_space: Discrete = Discrete(function_n)
        self.parameter_space: MultiDiscrete = MultiDiscrete(parameter_nvec)
        # tree traversal breadth first
        shape_dict: Dict[Any, int] = {}
        parent_spaces: List[Union[Space, Dict]] = [self.spaces]
        lengths = []
        while len(parent_spaces) > 0:
            new_lengths = []
            children_spaces = []
            for val in parent_spaces:
                if isinstance(val, Dict):
                    # for key, val in val.items():
                    #     if isinstance(val, Dict):
                    #         children_spaces.append(val)
                    if len(val) > 1:
                        print(val)
                        new_lengths.append(len(val))
                        children_spaces.extend(list(val.values()))
                    else:
                        key, val = list(val.items())[0]
                        assert isinstance(val, Discrete)
                        shape_dict[key] = val.n
                        new_lengths.append(val.n)
                else:
                    raise NotImplementedError
            if len(new_lengths) > 0:
                lengths.append(new_lengths)
            parent_spaces = children_spaces
        print(lengths)
        print(spaces)
        print(shape_dict)
        print(exit(7))

        self.nvec = None
        # todo, make this a tuple of spaces not multidiscrete
        self.flat_nvec: np.ndarray = np.array([function_n] + parameter_nvec)
        super(Hierarchical, self).__init__(None,
                                           None)  # None for shape and dtype, since it'll require special handling

    def seed(self, seed=None):
        for param_spaces in self.function_to_param_spaces.values():
            [space.seed(seed) for space in param_spaces]

    def sample(self):
        return OrderedDict(
                [(k, space.sample()) for k, space in self.spaces.items()])

    def contains(self, x):
        if not isinstance(x, dict) or len(x) != len(self.spaces):
            return False
        for k, space in self.spaces.items():
            if k not in x:
                return False
            if not space.contains(x[k]):
                return False
        return True

    def __getitem__(self, key):
        return self.spaces[key]

    def __repr__(self):
        return "Dict(" + ", ".join(
                [str(k) + ":" + str(s) for k, s in self.spaces.items()]) + ")"

    def to_jsonable(self, sample_n):
        # serialize as dict-repr of vectors
        return {key: space.to_jsonable([sample[key] for sample in sample_n]) \
                for key, space in self.spaces.items()}

    def from_jsonable(self, sample_n):
        dict_of_list = {}
        for key, space in self.spaces.items():
            dict_of_list[key] = space.from_jsonable(sample_n[key])
        ret = []
        for i, _ in enumerate(dict_of_list[key]):
            entry = {}
            for key, value in dict_of_list.items():
                entry[key] = value[i]
            ret.append(entry)
        return ret

    def __eq__(self, other):
        return isinstance(other, Hierarchical) and self.spaces == other.spaces

    def empty_param_mask(self, sampled_action_functions,
                         per_probability: bool,
                         device=torch.device('cpu')) -> \
            List[List[torch.Tensor]]:
        masks: List[List[torch.Tensor]] = []
        shape = ()
        if len(sampled_action_functions.shape) > 1:
            shape = sampled_action_functions.shape[:-1]
        for param_set in self.spaces.values():
            masks.append([
                    torch.ones((*shape, space.n if per_probability else
                    1)).to(device) for space in param_set.values()])
        return masks

    def action_function_masking(self,
                                sampled_action_functions: torch.Tensor,
                                per_probability: bool = True) -> \
            List[torch.Tensor]:
        '''
        :param sampled_action_functions: ASSUMES TO ONLY GET ONE ACTION TYPE
        :return:
        '''
        if len(sampled_action_functions.shape) > 2:
            raise NotImplementedError
        batch_size = sampled_action_functions.shape[0]
        # mask.scatter_(dim=-1, index=sampled_action_functions, value=1)
        masks: List[List[torch.Tensor]] = \
            self.empty_param_mask(sampled_action_functions,
                                  per_probability=per_probability,
                                  device=sampled_action_functions.device)
        flattened_masks: List[torch.Tensor] = []
        for function_idx in range(len(masks)):
            valid = sampled_action_functions == function_idx
            flattened_masks.extend([valid * param_mask for
                                    param_mask in masks[function_idx]])
        return flattened_masks

    def tensor_split(self, inputs: torch.Tensor):
        return inputs.split(tuple(self.flat_nvec), dim=-1)
