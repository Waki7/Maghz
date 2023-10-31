from __future__ import annotations

import unittest

from mgz.typing import *

#
# MAKE THIS IN ORDER OF THE ACTUAL FLOW OF BERT

# import altair as alt
# import GPUtil

class TestBert(unittest.TestCase):
    def setUp(self):
        pass

    def test_loss(self):
        prediction = torch.ones([2, 3, 4])
        prediction = torch.ones([2, 3, 4])
