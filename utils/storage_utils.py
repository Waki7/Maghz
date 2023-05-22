import os
from typing import *

import yaml


def merge_configs(original: Dict, new: Dict):
    new_config = {}
    for key in original.keys():
        new_config[key] = original[key]
    for key in new.keys():
        new_config[key] = new[key]
    return new_config


def save_config(cfg: Dict, dir: str, filename: str):
    if not filename.endswith('.yml'):
        filename = '{}.yml'.format(filename)
    with open(os.path.join(dir, filename), 'w') as file:
        yaml.dump(cfg, file)


def load_config(path):
    with open(path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg
