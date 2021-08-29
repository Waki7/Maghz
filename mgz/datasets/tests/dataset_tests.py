from typing import Type

from mgz.typing import *
import unittest
from mgz.datasets.spaceship_dataset import Spaceship
from mgz.datasets.img_dataset import ImageDataset


class TestStringMethods(unittest.TestCase):
    image_datasets: List[Type[ImageDataset]] = [Spaceship]

    def shape(self):
        for data_cls in TestStringMethods.image_datasets:
            dataset: ImageDataset = data_cls()

            datum = dataset[0]
            assert len(datum.features) == 1 == len(
                datum.labels), '{} currently is just one feature and ' \
                               'one label'.format(ImageDataset)
            assert len(datum.features.shape) == 3, 'expecting c x w x h'

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


if __name__ == '__main__':
    unittest.main()
