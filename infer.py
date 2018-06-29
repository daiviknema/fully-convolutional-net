import tensorflow as tf
import numpy as np
from fcn import FCN
from voc_data_reader import VOCDataReader
import cv2
import sys
import os
import logging
import datetime
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)

def print_usage():
  print('Usage:')
  print('python infer.py <tf-checkpoint> <num_examples>')

def label_matrix_to_image(label_mat, image_file = None):
  def convert(label_mod):
    # logging.debug('trying to hash {}'.format(label))
    label = label_mod[0]
    return np.array(dsr.voc_color_map[label], dtype=np.float32)
  logging.debug('Label mat shape: {}'.format(label_mat.shape))
  label_mat[label_mat == 21] = 255
  np.set_printoptions(threshold=np.nan)
  # logging.debug('{}'.format(label_mat))
  label_mat = np.expand_dims(label_mat, axis=2)
  label_mat = np.concatenate(
      [label_mat, np.zeros(label_mat.shape), np.zeros(label_mat.shape)],
      axis=2
    )
  img_mat = np.apply_along_axis(convert, 2, label_mat)
  logging.debug('Converted shape: {}'.format(img_mat.shape))
  if image_file is not None:
    plt.imsave(fname=image_file, arr=img_mat, format='png')
  return img_mat

if len(sys.argv) != 3:
  print_usage()
  exit(1)

num_examples = int(sys.argv[-1])
checkpoint = sys.argv[1]

TRAINVAL_ROOT_DIR = '/home/paperspace/PASCAL-VOC-Dataset/TrainVal'
TEST_ROOT_DIR = '/home/paperspace/PASCAL-VOC-Dataset/Test'
VGG_PARAMS_ROOT_DIR = '/home/paperspace/FCN/vgg-weights'

fcn = FCN(TRAINVAL_ROOT_DIR, TEST_ROOT_DIR, VGG_PARAMS_ROOT_DIR)
logging.debug('Created the fcn object')

dsr = VOCDataReader(TRAINVAL_ROOT_DIR, TEST_ROOT_DIR)
logging.debug('Created the dsr object')

if not os.path.exists('inference_results'):
  os.mkdir('inference_results')
newdir_path = 'inference_results/results-{}'
  .format(datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S'))
os.mkdir(newdir_path)
for i in range(num_examples):
  tb = dsr.next_train_batch(1)
  images = tb['batch']
  annotations = tb['annotations']
  names = tb['names']
  img_path = os.path.join(
      TRAINVAL_ROOT_DIR,
      'VOCdevkit/VOC2011/JPEGImages/{}.jpg'.format(names[0])
    )
  img_unnormalized = cv2.imread(img_path)[:,:,::-1]
  for i in images: img = i
  for a in annotations: ann = a
  segmentation = fcn.infer(img, checkpoint)
  label_matrix_to_image(
      segmentation,
      os.path.join(newdir_path, '{}_predicted_segmentation.png'
        .format(names[0]))
    )
  label_matrix_to_image(
      ann,
      os.path.join(newdir_path, '{}_ground_truth_segmentation.png'
        .format(names[0]))
    )
  plt.imsave(
      fname=os.path.join(newdir_path, '{}_actual_image.png'.format(names[0])),
      arr=img_unnormalized,
      format='png'
    )
