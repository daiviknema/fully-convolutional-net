from voc_data_reader import VOCDataReader
from fcn import FCN
import tensorflow as tf
import numpy as np

#TRAINVAL_ROOT_DIR = '/root/PASCAL-VOC-Dataset/TrainVal'
#TEST_ROOT_DIR = '/root/PASCAL-VOC-Dataset/Test'
#VGG_PARAMS_ROOT_DIR = '/root/tf-fcn/vgg-weights'

TRAINVAL_ROOT_DIR = '/home/paperspace/PASCAL-VOC-Dataset/TrainVal'
TEST_ROOT_DIR = '/home/paperspace/PASCAL-VOC-Dataset/Test'
VGG_PARAMS_ROOT_DIR = '/home/paperspace/FCN/vgg-weights'
dsr = VOCDataReader(TRAINVAL_ROOT_DIR, TEST_ROOT_DIR)
fcn = FCN(TRAINVAL_ROOT_DIR, TEST_ROOT_DIR, VGG_PARAMS_ROOT_DIR)

for i in dsr.next_train_batch(1)['batch']:
    img = i
img = np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]])
fcn.sess.run(tf.global_variables_initializer())
output = fcn.sess.run(
            fcn.net['fc8'],
            feed_dict = {
                fcn.net['input']: img
             }
         )

print(output.shape)
