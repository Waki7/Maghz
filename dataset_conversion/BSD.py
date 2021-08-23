from typing import *
import scipy.io
import os
from os.path import join, isfile, isdir
import os.path
import numpy as np
import cv2
import pickle
from PIL import Image
import torch
from mgz.utils.paths import DataPaths
from enum import Enum

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab


class DataSplits(Enum):
    Train = 'train'
    Val = 'val'
    Test = 'test'
    Debug = 'debug'


class DataFileLoader():
    def __init__(self, downsample_ratio=2, pad_value=0):
        self.pad_value = pad_value
        self.stored_data_path = None
        self.stored_file_name = None
        self.sampled_file_name = None
        self.downsample_ratio = downsample_ratio
        self.train_range = None
        self.val_range = None
        self.test_range = None

    def read_file(self, file):
        raise NotImplementedError

    def get_file_sets(self):
        '''
        :return: list of list of files, if applicable the first dimension should be split by train, val, test sets
        '''
        raise NotImplementedError

    def pad(self, image: np.ndarray, max_shape: list):
        pad_width = [(0, max_shape[i] - image.shape[i]) for i in
                     range(len(max_shape))]
        return np.pad(image,
                      pad_width=pad_width, mode='constant',
                      constant_values=self.pad_value)

    def shape_data_uniform(self,
                           data: list):  # todo bucketize if different sizes
        max_dims = [0] * len(data[0].shape)
        for i in range(0, len(data)):
            image = data[i]
            if image.shape[0] > image.shape[
                1]:  # want all images to be in portrait
                image = np.swapaxes(image, 0, 1)
            max_dims = [max(image.shape[dim_idx], max_dims[dim_idx]) for dim_idx
                        in range(0, len(max_dims))]
        return [self.pad(image=image, max_shape=max_dims) for image in data]

    def read_files(self, files):
        data = []
        data_downsampled = []
        for sample_path in files:
            image, image_downsampled = self.read_file(sample_path)
            data.append(image)
            data_downsampled.append(image_downsampled)
        data = self.shape_data_uniform(data)
        data_downsampled = self.shape_data_uniform(data_downsampled)
        return np.stack(data), np.stack(data_downsampled)

    def load_data_from_files(self):
        file_sets = self.get_file_sets()
        return file_sets


class COCO(DataFileLoader):
    def __init__(self, split=DataSplits.Train):
        super(COCO, self).__init__()
        self.split_mapping: Dict[DataSplits, str] = {
            DataSplits.Train: 'train',
            DataSplits.Debug: 'train',
            DataSplits.Val: 'val',
            DataSplits.Test: 'test',
        }
        pylab.rcParams['figure.figsize'] = (8.0, 10.0)

        dataDir = '..'
        dataType = 'val2017'
        annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

        self.folder = self.split_mapping[split]

        self.img_path = os.path.join(DataPaths.BSR_image_folder, self.folder)
        self.img_ext = '.jpg'

        self.label_path = os.path.join(DataPaths.BSR_label_folder, self.folder)
        self.label_ext = '.mat'

        self.img_path_sets = self.get_file_sets(self.img_path, self.img_ext)
        self.label_path_sets = self.get_file_sets(self.label_path,
                                                  self.label_ext)
        print(self.label_path_sets)
        img = self.img_path_sets[0]
        import torchvision.io as torch_io

        img = torch_io.read_image(img)
        print(img.shape)
        print(self.read_file(self.label_path_sets[0])[0].shape)
        print(self.read_file(self.label_path_sets[0])[1].shape)
        print(exit(9))

    def get_file_sets(self, path: str, file_ext: str) -> List[str]:
        file_path_list = []
        for filename in os.listdir(path):
            if filename.endswith(file_ext):
                sample_path = join(path, filename)
                file_path_list.append(sample_path)
        return file_path_list

    def read_file(self, file):
        self.mat_key = 'groundTruth'
        self.segmentation_index = 0
        self.boundary_index = 1

        mat = scipy.io.loadmat(file)
        mat_data = np.squeeze(mat[self.mat_key][0, 0]).item(0)
        datum = mat_data[
            self.segmentation_index]  # segementation ground truth, mat_data[1] is the boundary boxes
        datum = datum - 1
        datum_downsampled = downsample(datum, ratio=self.downsample_ratio,
                                       interpolation=cv2.INTER_NEAREST)
        return datum, datum_downsampled


class BSRImages(DataFileLoader):
    def __init__(self, split=DataSplits.Train):
        super(BSRImages, self).__init__()
        self.split_mapping: Dict[DataSplits, str] = {
            DataSplits.Train: 'train',
            DataSplits.Debug: 'train',
            DataSplits.Val: 'val',
            DataSplits.Test: 'test',
        }
        self.folder = self.split_mapping[split]

        self.img_path = os.path.join(DataPaths.BSR_image_folder, self.folder)
        self.img_ext = '.jpg'

        self.label_path = os.path.join(DataPaths.BSR_label_folder, self.folder)
        self.label_ext = '.mat'

        self.img_path_sets = self.get_file_sets(self.img_path, self.img_ext)
        self.label_path_sets = self.get_file_sets(self.label_path,
                                                  self.label_ext)
        print(self.label_path_sets)
        img = self.img_path_sets[0]
        import torchvision.io as torch_io

        img = torch_io.read_image(img)
        print(img.shape)
        print(self.read_file(self.label_path_sets[0])[0].shape)
        print(self.read_file(self.label_path_sets[0])[1].shape)
        print(exit(9))

    def get_file_sets(self, path: str, file_ext: str) -> List[str]:
        file_path_list = []
        for filename in os.listdir(path):
            if filename.endswith(file_ext):
                sample_path = join(path, filename)
                file_path_list.append(sample_path)
        return file_path_list

    def read_file(self, file):
        self.mat_key = 'groundTruth'
        self.segmentation_index = 0
        self.boundary_index = 1

        mat = scipy.io.loadmat(file)
        mat_data = np.squeeze(mat[self.mat_key][0, 0]).item(0)
        datum = mat_data[
            self.segmentation_index]  # segementation ground truth, mat_data[1] is the boundary boxes
        datum = datum - 1
        datum_downsampled = downsample(datum, ratio=self.downsample_ratio,
                                       interpolation=cv2.INTER_NEAREST)
        return datum, datum_downsampled


class VOCImages(DataFileLoader):
    def __init__(self, downsample_ratio):
        super(VOCImages, self).__init__(downsample_ratio, pad_value=0)
        self.file_ext = '.jpg'
        self.year_paths = ['..\\Data\\VOC\\VOCdevkit\\VOC20{}\\'.format(i) for i
                           in ['08', '09', '10', '11', '12']]
        self.image_paths = [root_path + 'JPEGImages\\' for root_path in
                            self.year_paths]
        self.image_set_paths = ['ImageSets\\Segmentation\\' + i + '.txt' for i
                                in ['train', 'val']]

    def read_file(self, file):
        datum = scipy.misc.imread(file)
        datum_downsampled = downsample(datum, ratio=self.downsample_ratio,
                                       interpolation=cv2.INTER_LINEAR)
        return datum, datum_downsampled

    def get_file_sets(self):
        file_sets = []
        for image_lists_path in self.image_set_paths:
            file_names = []
            for year_path, image_path in zip(self.year_paths, self.image_paths):
                set_list_path = year_path + image_lists_path
                with open(set_list_path, 'r') as f:
                    for line in f:
                        image_name_line = line.strip() + self.file_ext
                        sample_path = join(image_path, image_name_line)
                        file_names.append(sample_path)
            file_sets.append(file_names)
        return file_sets


class VOCLabels(DataFileLoader):
    def __init__(self, downsample_ratio):
        super(VOCLabels, self).__init__(downsample_ratio, pad_value=255)
        self.file_ext = '.png'
        self.year_paths = ['..\\Data\\VOC\\VOCdevkit\\VOC20{}\\'.format(i) for i
                           in ['08', '09', '10', '11', '12']]
        self.image_paths = [root_path + 'SegmentationClass\\' for root_path in
                            self.year_paths]
        self.image_set_paths = ['ImageSets\\Segmentation\\' + i + '.txt' for i
                                in ['train', 'val']]

    def read_file(self, file):
        img = Image.open(file)
        datum = np.array(img)
        datum_downsampled = downsample(datum, ratio=self.downsample_ratio,
                                       interpolation=cv2.INTER_NEAREST)
        return datum, datum_downsampled

    def get_file_sets(self):
        file_sets = []
        for image_lists_path in self.image_set_paths:
            file_names = []
            for year_path, image_path in zip(self.year_paths, self.image_paths):
                set_list_path = year_path + image_lists_path
                with open(set_list_path, 'r') as f:
                    for line in f:
                        image_name_line = line.strip() + self.file_ext
                        sample_path = join(image_path, image_name_line)
                        file_names.append(sample_path)
            file_sets.append(file_names)
        return file_sets


def downsample(img, ratio, interpolation=cv2.INTER_NEAREST):
    new_h = img.shape[0] // ratio
    new_w = img.shape[1] // ratio
    img = cv2.resize(img, (new_w, new_h),
                     interpolation=interpolation)  # opencv takes w x h instead of h x w in numpy
    return img


if __name__ == '__main__':
    a = COCO()
