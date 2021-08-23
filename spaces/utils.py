from typing import *
import gym.spaces

from spaces.hierarchical import Hierarchical
import spaces as rs


def flatdim(space):
    if isinstance(space, Hierarchical):
        parameters_actions = []
        for sub_actions in space.function_to_param_spaces.values():
            parameters_actions.extend(sub_actions)
        return int(sum([flatdim(s) for s in parameters_actions])) + \
               len(space.spaces)
    else:
        return gym.spaces.flatdim(space)


def flatten(space, x):
    if isinstance(space, Hierarchical):
        pass
    else:
        return gym.spaces.flatten(space, x)


def unflatten(space, x):
    if isinstance(space, Hierarchical):
        pass
    else:
        return gym.spaces.unflatten(space, x)


def space_to_network_shapes(space: rs.Space) -> Iterable[int]:
    out_shapes = None
    if isinstance(space, rs.MultiDiscrete):
        out_shapes = space.nvec
    elif isinstance(space, rs.Discrete):
        out_shapes = [space.n]
    elif isinstance(space, rs.Hierarchical):
        out_shapes = space.flat_nvec
    elif isinstance(space, rs.Box):
        out_shapes = space.shape
    elif isinstance(space, rs.ActionGraph):
        out_shapes = space.nvec
    elif isinstance(space, rs.Dict):
        out_shapes = []
        for spc in space.spaces.values():
            out_shapes.extend(space_to_network_shapes(spc))
    else:
        raise NotImplementedError
    return out_shapes


def space_to_out_shapes(space: rs.Space) -> Iterable[int]:
    out_shapes = None
    if isinstance(space, rs.MultiDiscrete):
        out_shapes = [1] * len(space.nvec)
    elif isinstance(space, rs.Discrete):
        out_shapes = [1]
    elif isinstance(space, rs.Hierarchical):
        out_shapes = [len(space.flat_nvec)]
    elif isinstance(space, rs.Box):
        out_shapes = space.shape
    elif isinstance(space, rs.ActionGraph):
        out_shapes = [len(space.nvec)]
    elif isinstance(space, rs.Dict):
        out_shapes = []
        for spc in space.spaces.values():
            out_shapes.extend(space_to_out_shapes(spc))
    else:
        raise NotImplementedError
    return out_shapes


def space_to_out_space(space: rs.Space) -> Iterable[rs.Space]:
    out_shapes = None
    if isinstance(space, rs.MultiDiscrete):
        out_shapes = [space] * len(space.nvec)
    elif isinstance(space, rs.Discrete):
        out_shapes = [space]
    elif isinstance(space, rs.Hierarchical):
        out_shapes = [space] * len(space.flat_nvec)
    elif isinstance(space, rs.Box):
        out_shapes = [space]
    elif isinstance(space, rs.ActionGraph):
        out_shapes = [space] * len(space.nvec)
    elif isinstance(space, rs.Dict):
        out_shapes = []
        for spc in space.spaces.values():
            out_shapes.extend(space_to_out_space(spc))
    else:
        raise NotImplementedError
    return out_shapes
