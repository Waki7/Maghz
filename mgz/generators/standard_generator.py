from mgz.typing import *
from mgz.generators.base_generator import BaseGenerator

class StandardGenerator(BaseGenerator):
    def __init__(self, dir_path: DirPath):
        super(StandardGenerator, self).__init__()