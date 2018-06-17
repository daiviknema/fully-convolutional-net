import tensorflow as tf
import numpy as np
from fcn import FCN
from voc_data_reader import VOCDataReader
import cv2
import sys, os, logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

def print_usage():
  print('Usage:')
  print('python train.py MAX_ITERATIONS_COARSE MAX_ITERATIONS_FINE SAVE_PARAMS_AFTER RESTORE_COARSE_PARAMS_PATH')

if len(sys.argv) not in [4, 5]:
  print_usage()
  exit(1)

# TRAINVAL_ROOT_DIR = '/root/PASCAL-VOC-Dataset/TrainVal'
# TEST_ROOT_DIR = '/root/PASCAL-VOC-Dataset/Test'
# VGG_PARAMS_ROOT_DIR = '/root/tf-fcn/vgg-weights'

TRAINVAL_ROOT_DIR = '/home/paperspace/PASCAL-VOC-Dataset/TrainVal'
TEST_ROOT_DIR = '/home/paperpsace/PASCAL-VOC-Dataset/Test'
VGG_PARAMS_ROOT_DIR = '/home/paperspace/FCN/vgg-weights'

MAX_ITERATIONS_COARSE = int(sys.argv[1])
MAX_ITERATIONS_FINE = int(sys.argv[2])
SAVE_PARAMS_AFTER = int(sys.argv[3])
if len(sys.argv) == 5:
  RESTORE_CKPT = sys.argv[4]
else:
  RESTORE_CKPT = None

fcn = FCN(TRAINVAL_ROOT_DIR, TEST_ROOT_DIR, VGG_PARAMS_ROOT_DIR)
fcn.train(MAX_ITERATIONS_COARSE, MAX_ITERATIONS_FINE, SAVE_PARAMS_AFTER, RESTORE_CKPT)
