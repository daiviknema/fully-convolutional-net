import tensorflow as tf
import numpy as np
import os
import random
import logging
import cv2
import sys

class VOCDataReader(object):
  def __init__(self, train_root_dir, test_root_dir, log_level = logging.DEBUG):
    '''Constructs the Dataset Reader object

    This function must be supplied with the root directory of the dataset (test
    or train). This will probably be something like: path/to/Test or
    path/to/TrainVal.

    Example: ../PASCAL-VOC-Dataset/download/Test

    Arguments:
    train_root_dir: Root directory of the TrainVal data.
    test_root_dir: Root directory of the Test data.
    '''
    # Init logger
    self.logger = logging.getLogger('VOCDataReader')
    self.logger.setLevel(log_level)

    # Init instance vars
    self.train_root_dir = train_root_dir
    self.test_root_dir = test_root_dir
    self.channel_means = np.array([103.939, 116.779, 123.68])
    self.segmentation_imageset_train_files = []
    with open(os.path.join(self.train_root_dir,\
        'VOCdevkit/VOC2011/ImageSets/Segmentation/train.txt')) as fp:
      self.segmentation_imageset_train_files = fp.read().split()
    self.voc_color_map = {
      0 : (0, 0, 0),
      1 : (128, 0, 0),
      2 : (0, 128, 0),
      3 : (128, 128, 0),
      4 : (0, 0, 128),
      5 : (128, 0, 128),
      6 : (0, 128, 128),
      7 : (128, 128, 128),
      8 : (64, 0, 0),
      9 : (192, 0, 0),
      10: (64, 128, 0),
      11: (192, 128, 0),
      12: (64, 0, 128),
      13: (192, 0, 128),
      14: (64, 128, 128),
      15: (192, 128, 128),
      16: (0, 64, 0),
      17: (128, 64, 0),
      18: (0, 192, 0),
      19: (128, 192, 0),
      20: (0, 64, 128),
      255: (224, 224, 192),
    }
    self.inv_voc_color_map = {
      (0, 0, 0)       : 0,
      (128, 0, 0)     : 1,
      (0, 128, 0)     : 2,
      (128, 128, 0)   : 3,
      (0, 0, 128)     : 4,
      (128, 0, 128)   : 5,
      (0, 128, 128)   : 6,
      (128, 128, 128) : 7,
      (64, 0, 0)      : 8,
      (192, 0, 0)     : 9,
      (64, 128, 0)    : 10,
      (192, 128, 0)   : 11,
      (64, 0, 128)    : 12,
      (192, 0, 128)   : 13,
      (64, 128, 128)  : 14,
      (192, 128, 128) : 15,
      (0, 64, 0)      : 16,
      (128, 64, 0)    : 17,
      (0, 192, 0)     : 18,
      (128, 192, 0)   : 19,
      (0, 64, 128)    : 20,
      (224, 224, 192) : 21,
    }

  def next_train_batch(self, batch_sz = 10, normalize = True):
    '''Returns a batch sampled from the training data.

    Performs random uniform sampling from the data. Returns a dict of the form:
    {
      'names': list of images in batch
      'batch': iterator object
      'annotations': iterator object
    }

    Arguments:
    batch_sz: Size of the batch to be sampled. Default value is 10.
    '''
    files_in_batch = random.sample(self.segmentation_imageset_train_files,\
                                   batch_sz)
    self.logger.debug('Batch selected: {}'.format(files_in_batch))

    data_files_in_batch = [os.path.join(self.train_root_dir,\
        'VOCdevkit/VOC2011/JPEGImages',\
        '{}.jpg'.format(x))\
        for x in files_in_batch]

    if normalize:
      batch = map(self._read_transform_image, data_files_in_batch)
    else:
      batch = map(self._read_image, data_files_in_batch)

    annotation_files_in_batch = [os.path.join(self.train_root_dir,\
        'VOCdevkit/VOC2011/SegmentationClass',\
        '{}.png'.format(x))\
        for x in files_in_batch]

    annotations = map(self._read_annotation_file, annotation_files_in_batch)

    return {
      'names': files_in_batch,
      'batch': batch,
      'annotations': annotations
    }

  def _read_transform_image(self, img_path):
    '''Translates and scales the input image.

    For each channel, subtracts the dataset mean for that channel and divides
    by 255.0

    Arguments:
    img_path: path to image file.
    '''
    img = cv2.imread(img_path)[:,:,::-1]
    r, g, b = np.split(img, 3, axis=2)
    r = (r - self.channel_means[0])/255.0
    g = (g - self.channel_means[1])/255.0
    b = (b - self.channel_means[2])/255.0
    img = np.concatenate((r, g, b), axis=2)
    return img

  def _read_image(self, img_path):
    '''Translates and scales the input image.

    For each channel, subtracts the dataset mean for that channel and divides
    by 255.0

    Arguments:
    img_path: path to image file.
    '''
    img = cv2.imread(img_path)[:,:,::-1]
    r, g, b = np.split(img, 3, axis=2)
    img = np.concatenate((r, g, b), axis=2)
    return img

  def _read_annotation_file(self, annotation_path):
    '''Returns pixelwise class labels in form of numpy matrix

    Reads an RGB annotation image and maps each pixel to class label.

    Arguments:
    annotation_path -- path to annotation file.
    '''
    annotation = cv2.imread(annotation_path)[:,:,::-1]
    def convert(rgb):
      return self.inv_voc_color_map[tuple(rgb)]
    annotation = np.apply_along_axis(convert, 2, annotation)
    self.logger.debug(annotation.shape)
    return annotation
