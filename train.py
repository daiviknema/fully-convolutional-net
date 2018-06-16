import tensorflow as tf
import numpy as np
from fcn import FCN
from voc_data_reader import VOCDataReader
import cv2
import sys, os, logging
import matplotlib.pyplot as plt

def print_usage():
  print('python train.py [--coarse | --fine]')

if len(sys.argv) != 2:
  print_usage()
  exit(1)

assert(sys.argv[-1] in ['--coarse', '--fine'])
if sys.argv[-1] == '--coarse':
  coarse = True
else:
  coarse = False

TRAINVAL_ROOT_DIR = '/root/PASCAL-VOC-Dataset/TrainVal'
TEST_ROOT_DIR = '/root/PASCAL-VOC-Dataset/Test'
VGG_PARAMS_ROOT_DIR = '/root/tf-fcn/vgg-weights'

MAX_ITERATIONS = 1
SAVE_PARAMS_AFTER = 1

fcn = FCN(TRAINVAL_ROOT_DIR, TEST_ROOT_DIR, VGG_PARAMS_ROOT_DIR)

if os.path.exists('best_params') and len(os.listdir('best_params')) > 0:
  best_params_ckpt = os.path.join('best_params', '.'.join(os.listdir('best_params')[0].split('.')[:-1]))
else:
  best_params_ckpt = None
fcn.train(MAX_ITERATIONS, SAVE_PARAMS_AFTER, best_params_ckpt, coarse)
