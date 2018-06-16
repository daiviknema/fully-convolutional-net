import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import logging
import os

from voc_data_reader import VOCDataReader

def get_vgg_params(path_to_vgg_weights, log_lvl = logging.DEBUG):
  logger = logging.getLogger('get_vgg_params')
  logger.setLevel(log_lvl)
  params = np.load('vgg-weights/weights.dat', encoding='bytes')
  params = np.reshape(params, (1,))[0]
  for key in params:
    params[key] = { k.decode('utf-8') : v for k, v in params[key].items() }
  logger.debug('VGG Keys: {}'.format(params.keys()))
  return params

class FCN(object):
  def __init__(self, trainval_root_dir, test_root_dir, vgg_params_path, log_lvl=logging.DEBUG):
    # Disable TF logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Init logger
    self.logger = logging.getLogger('FCN')
    self.logger.setLevel(log_lvl)

    # Init dataset reader
    self.dataset_reader = VOCDataReader(trainval_root_dir, test_root_dir, logging.INFO)

    # Init instance vars
    self.net = None
    self.params = None
    self.optimizer = None
    self.sess = tf.Session()

    # Get VGG params
    self.vgg_params = get_vgg_params(vgg_params_path)

    # Class weights
    self.class_weights_ = np.array([0.01287317, 1.3872424, 2.887279, 0.9050577, 1.5103302, 1.5737615, 0.5479136, 0.6307377, 0.37878844, 0.94728917, 1.4240671, 0.7215936, 0.63659954, 1.0965613, 0.7973126, 0.20773387, 1.733643, 1.1090201, 0.7762861, 0.7039499, 1.0119597], dtype=np.float32)/21.0

    # Build Model
    self.build_model()


  def _get_conv_params(self, ker_sz, in_channels, out_channels, name):
    dic = {
      'weights': tf.Variable(
          tf.truncated_normal([ker_sz, ker_sz, in_channels, out_channels]),
          '{}_weights'.format(name)
          ),
      'biases': tf.Variable(
          tf.zeros([out_channels,]),
          '{}_biases'.format(name)
          ),
    }
    return dic

  def _get_vgg_conv_params(self, name):
    if name in ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']:
      dic = {
        'weights': tf.Variable(self.vgg_params[name]['weights'], '{}_weights'.format(name)),
        'biases': tf.Variable(self.vgg_params[name]['biases'], '{}_biases'.format(name))
      }
      return dic
    elif name == 'fc6':
      converted_wt = np.reshape(self.vgg_params[name]['weights'], [7, 7, 512, 4096])
      dic = {
        'weights': tf.Variable(converted_wt, '{}_weights'.format(name)),
        'biases': tf.Variable(self.vgg_params[name]['biases'], '{}_biases'.format(name))
      }
      return dic
    elif name == 'fc7':
      converted_wt = np.reshape(self.vgg_params[name]['weights'], [1, 1, 4096, 4096])
      dic = {
        'weights': tf.Variable(converted_wt, '{}_weights'.format(name)),
        'biases': tf.Variable(self.vgg_params[name]['biases'], '{}_biases'.format(name))
      }
      return dic
    elif name == 'fc8':
      converted_wt = np.reshape(self.vgg_params[name]['weights'], [1, 1, 4096, 1000])
      dic = {
        'weights': tf.Variable(converted_wt, '{}_weights'.format(name)),
        'biases': tf.Variable(self.vgg_params[name]['biases'], '{}_biases'.format(name))
      }
      return dic
    elif name == 'score_fc':
      dic = {
        'weights': tf.Variable(tf.contrib.layers.xavier_initializer()([1, 1, 1000, 21]), '{}_weights'.format(name)),
        'biases': tf.Variable(tf.zeros([21,]), '{}_biases'.format(name))
      }
      return dic
    elif name == 'score_pool4':
      dic = {
        'weights': tf.Variable(tf.contrib.layers.xavier_initializer()([1, 1, 512, 21]), '{}_weights'.format(name)),
        'biases': tf.Variable(tf.zeros([21,]), '{}_biases'.format(name))
      }
      return dic
    elif name == 'score_pool3':
      dic = {
        'weights': tf.Variable(tf.contrib.layers.xavier_initializer()([1, 1, 256, 21]), '{}_weights'.format(name)),
        'biases': tf.Variable(tf.zeros([21,]), '{}_biases'.format(name))
      }
      return dic

  def _get_upconv_params(self, factor, out_channels, name):
    kernel_sz = 2*factor - factor%2
    weights = np.zeros([kernel_sz, kernel_sz, out_channels, out_channels], dtype=np.float32)

    # Populate weights
    if kernel_sz % 2 == 1:
      center = factor - 1
    else:
      center = factor - 0.5
    tmp = np.ogrid[:kernel_sz, :kernel_sz]
    kernel = (1 - abs(tmp[0] - center)/factor) * (1 - abs(tmp[1] - center)/factor)
    for i in range(out_channels):
      weights[:, :, i, i] = kernel

    # Populate biases
    biases = np.zeros([out_channels,], dtype=np.float32)

    dic = {
      'weights': tf.Variable(weights, '{}_weights'.format(name)),
      'biases': tf.Variable(biases, '{}_biases'.format(name))
    }
    return dic

  def _get_upconv_layer(self, bottom, params):
    pass

  def _get_conv_layer(self, bottom, params):
    layer = tf.nn.bias_add(
        tf.nn.conv2d(
            bottom,
            params['weights'],
            [1, 1, 1, 1],
            'SAME'
          ),
        params['biases']
        )
    return layer

  def _get_relu_layer(self, bottom):
    layer = tf.nn.relu(bottom)
    return layer

  def _get_max_pool_layer(self, bottom):
    layer = tf.nn.max_pool(bottom,
                           [1, 2, 2, 1],
                           [1, 2, 2, 1],
                           'SAME')
    return layer

  def _get_crop_layer(self, big_batch, small_batch):
    h_s = tf.shape(small_batch)[1]
    w_s = tf.shape(small_batch)[2]

    h_b = tf.shape(big_batch)[1]
    w_b = tf.shape(big_batch)[2]

    return big_batch[:,
                   (h_b - h_s)//2 : (h_b - h_s)//2 + h_s,
                   (w_b - w_s)//2 : (w_b - w_s)//2 + w_s,
                   :]

  def _get_loss_layer(self, prediction, annotation, class_weights = None):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = annotation,
                                                          logits = prediction)
    loss_no_nans = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)
    if class_weights is not None:
      for i in range(21):
        weighted_loss = tf.where(tf.equal(annotation, i), class_weights[i]*loss_no_nans, loss_no_nans)
      return tf.reduce_mean(weighted_loss)
    return tf.reduce_mean(loss_no_nans)

  def _get_loss_layer_v2(self, prediction, annotation, class_weights=None):
    one_hot_annotation = tf.one_hot(annotation, 22, on_value=1.0, off_value=0.0, axis=-1) # (1, H, W, 21)
    one_hot_annotation = tf.reshape(tf.slice(one_hot_annotation, [0, 0, 0, 0],
        [tf.shape(one_hot_annotation)[0],
         tf.shape(one_hot_annotation)[1],
         tf.shape(one_hot_annotation)[2],
         tf.shape(one_hot_annotation)[3]-1]), [-1, 21]) # (H*W, 21)
    if class_weights is not None:
      class_weights = tf.expand_dims(class_weights, 0) # (1, 21)
    else:
      class_weights = tf.expand_dims(tf.ones([21,], dtype=tf.float32), 0)
    weighted_annotation = tf.transpose(tf.matmul(one_hot_annotation, tf.transpose(class_weights))) # (1, H*W)
    #probs = tf.nn.softmax(prediction)
    #loss = - tf.reduce_mean(tf.log(probs) * weighted_annotation)
    loss = tf.multiply(weighted_annotation, tf.reshape(tf.nn.softmax_cross_entropy_with_logits(labels = one_hot_annotation, logits = prediction), [1, -1]))
    return tf.reduce_mean(loss)


  def build_model(self):
    self.net = {}
    self.params = {}
    self.net['input'] = tf.placeholder(dtype=tf.float32)
    self.net['annotation'] = tf.placeholder(dtype=tf.int64)
    if self.net['input'] is None:
      self.logger.debug('net[\'input\'] is None')

    # Conv 1
    self.params['conv1_1'] = self._get_vgg_conv_params('conv1_1')
    self.net['conv1_1'] = self._get_conv_layer(self.net['input'],\
                                               self.params['conv1_1'])
    self.net['relu1_1'] = self._get_relu_layer(self.net['conv1_1'])

    self.params['conv1_2'] = self._get_vgg_conv_params('conv1_2')
    self.net['conv1_2'] = self._get_conv_layer(self.net['relu1_1'],
                                               self.params['conv1_2'])
    self.net['relu1_2'] = self._get_relu_layer(self.net['conv1_2'])

    # Pool 1
    self.net['pool1'] = self._get_max_pool_layer(self.net['relu1_2'])

    # Conv 2
    self.params['conv2_1'] = self._get_vgg_conv_params('conv2_1')
    self.net['conv2_1'] = self._get_conv_layer(self.net['pool1'],
                                               self.params['conv2_1'])
    self.net['relu2_1'] = self._get_relu_layer(self.net['conv2_1'])

    self.params['conv2_2'] = self._get_vgg_conv_params('conv2_2')
    self.net['conv2_2'] = self._get_conv_layer(self.net['relu2_1'],
                                               self.params['conv2_2'])
    self.net['relu2_2'] = self._get_relu_layer(self.net['conv2_2'])

    # Pool 2
    self.net['pool2'] = self._get_max_pool_layer(self.net['relu2_2'])

    # Conv 3
    self.params['conv3_1'] = self._get_vgg_conv_params('conv3_1')
    self.net['conv3_1'] = self._get_conv_layer(self.net['pool2'],
                                               self.params['conv3_1'])
    self.net['relu3_1'] = self._get_relu_layer(self.net['conv3_1'])

    self.params['conv3_2'] = self._get_vgg_conv_params('conv3_2')
    self.net['conv3_2'] = self._get_conv_layer(self.net['relu3_1'],
                                               self.params['conv3_2'])
    self.net['relu3_2'] = self._get_relu_layer(self.net['conv3_2'])

    self.params['conv3_3'] = self._get_vgg_conv_params('conv3_3')
    self.net['conv3_3'] = self._get_conv_layer(self.net['relu3_2'],
                                               self.params['conv3_3'])
    self.net['relu3_3'] = self._get_relu_layer(self.net['conv3_3'])

    # Pool 3
    self.net['pool3'] = self._get_max_pool_layer(self.net['relu3_3'])

    # Conv 4
    self.params['conv4_1'] = self._get_vgg_conv_params('conv4_1')
    self.net['conv4_1'] = self._get_conv_layer(self.net['pool3'],
                                               self.params['conv4_1'])
    self.net['relu4_1'] = self._get_relu_layer(self.net['conv4_1'])

    self.params['conv4_2'] = self._get_vgg_conv_params('conv4_2')
    self.net['conv4_2'] = self._get_conv_layer(self.net['relu4_1'],
                                               self.params['conv4_2'])
    self.net['relu4_2'] = self._get_relu_layer(self.net['conv4_2'])

    self.params['conv4_3'] = self._get_vgg_conv_params('conv4_3')
    self.net['conv4_3'] = self._get_conv_layer(self.net['relu4_2'],
                                               self.params['conv4_3'])
    self.net['relu4_3'] = self._get_relu_layer(self.net['conv4_3'])

    # Pool 4
    self.net['pool4'] = self._get_max_pool_layer(self.net['relu4_3'])

    # Conv 5
    self.params['conv5_1'] = self._get_vgg_conv_params('conv5_1')
    self.net['conv5_1'] = self._get_conv_layer(self.net['pool4'],
                                               self.params['conv5_1'])
    self.net['relu5_1'] = self._get_relu_layer(self.net['conv5_1'])

    self.params['conv5_2'] = self._get_vgg_conv_params('conv5_2')
    self.net['conv5_2'] = self._get_conv_layer(self.net['relu5_1'],
                                               self.params['conv5_2'])
    self.net['relu5_2'] = self._get_relu_layer(self.net['conv5_2'])

    self.params['conv5_3'] = self._get_vgg_conv_params('conv5_3')
    self.net['conv5_3'] = self._get_conv_layer(self.net['relu5_2'],
                                               self.params['conv5_3'])
    self.net['relu5_3'] = self._get_relu_layer(self.net['conv5_3'])

    # Pool 5
    self.net['pool5'] = self._get_max_pool_layer(self.net['relu5_3'])

    # FC 6
    self.params['fc6'] = self._get_vgg_conv_params('fc6')
    fc6_pre_relu = self._get_conv_layer(self.net['pool5'], self.params['fc6'])
    self.net['fc6'] = self._get_relu_layer(fc6_pre_relu)

    # FC 7
    self.params['fc7'] = self._get_vgg_conv_params('fc7')
    fc7_pre_relu = self._get_conv_layer(self.net['fc6'], self.params['fc7'])
    self.net['fc7'] = self._get_relu_layer(fc7_pre_relu)

    # FC 8
    self.params['fc8'] = self._get_vgg_conv_params('fc8')
    fc8_pre_relu = self._get_conv_layer(self.net['fc7'], self.params['fc8'])
    self.net['fc8'] = self._get_relu_layer(fc8_pre_relu)

    # Score FC
    self.params['score_fc'] = self._get_vgg_conv_params('score_fc')
    self.net['score_fc'] = self._get_conv_layer(self.net['fc8'], self.params['score_fc'])

    # Upconv 2x
    self.params['upconv2'] = self._get_upconv_params(2, 21, 'upconv2')
    self.net['upconv2'] = tf.nn.conv2d_transpose(self.net['score_fc'], self.params['upconv2']['weights'], output_shape=[1, tf.shape(self.net['score_fc'])[1] * 2, tf.shape(self.net['score_fc'])[2] * 2, 21], strides = [1,2,2,1])
    self.net['cropped_upconv2'] = self._get_crop_layer(self.net['upconv2'], self.net['pool4'])

    # Score pool4
    self.params['score_pool4'] = self._get_vgg_conv_params('score_pool4')
    self.net['score_pool4'] = self._get_conv_layer(self.net['pool4'], self.params['score_pool4'])

    # Fuse pool4
    self.net['fuse_pool4'] = tf.add(self.net['score_pool4'], self.net['cropped_upconv2'])

    # Upconv 4x
    self.params['upconv4'] = self._get_upconv_params(2, 21, 'upconv4')
    self.net['upconv4'] = tf.nn.conv2d_transpose(self.net['fuse_pool4'], self.params['upconv4']['weights'], output_shape=[1, tf.shape(self.net['fuse_pool4'])[1] * 2, tf.shape(self.net['fuse_pool4'])[2] * 2, 21], strides = [1,2,2,1])
    self.net['cropped_upconv4'] = self._get_crop_layer(self.net['upconv4'], self.net['pool3'])

    # Score pool3
    self.params['score_pool3'] = self._get_vgg_conv_params('score_pool3')
    self.net['score_pool3'] = self._get_conv_layer(self.net['pool3'], self.params['score_pool3'])

    # Fuse pool3
    self.net['fuse_pool3'] = tf.add(self.net['score_pool3'], self.net['cropped_upconv4'])

    # Final score layer
    self.params['scores_final'] = self._get_upconv_params(8, 21, 'scores_final')
    self.net['scores_final'] = tf.nn.conv2d_transpose(self.net['fuse_pool3'], self.params['scores_final']['weights'], output_shape=[1,tf.shape(self.net['fuse_pool3'])[1] * 8, tf.shape(self.net['fuse_pool3'])[2] * 8, 21], strides=[1,8,8,1])
    self.net['cropped_scores_final'] = self._get_crop_layer(self.net['scores_final'], self.net['input'])

    class_weights = tf.constant(self.class_weights_)

    self.net['loss'] = self._get_loss_layer_v2(self.net['cropped_scores_final'],
                                               self.net['annotation'])

  def train(self, max_iterations, save_params_after=None, restore_params=None, is_coarse = False):
    self.optimizer = tf.train.AdamOptimizer()

    if is_coarse:
      self.logger.debug('Minimizing only score filters')
      train_step = self.optimizer.minimize(self.net['loss'], var_list=[self.params['score_pool3']['weights'], self.params['score_pool4']['weights'], self.params['score_fc']['weights']])
    else:
      train_step = self.optimizer.minimize(self.net['loss'])

    self.sess.run(tf.global_variables_initializer())
    if is_coarse:
      saver = tf.train.Saver({
        'score_pool3_weights': self.params['score_pool3']['weights'],
        'score_pool4_weights': self.params['score_pool4']['weights'],
        'score_fc_weights': self.params['score_fc']['weights'],
        }, max_to_keep = 1)
    else:
      saver = tf.train.Saver(max_to_keep = 1)
    if restore_params is not None:
      saver.restore(self.sess, restore_params)
    cumul_loss = 0.0
    best_loss = None
    for iteration in range(max_iterations):
      batch = self.dataset_reader.next_train_batch(1)
      for x in batch['batch']: img = x
      for x in batch['annotations']: ann = x
      img = np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]])
      ann = np.reshape(ann, [1, ann.shape[0], ann.shape[1]])
      _, loss = self.sess.run([train_step, self.net['loss']],
                              feed_dict = {
                                self.net['input']: img,
                                self.net['annotation']: ann,
                              })
      cumul_loss += loss
      if save_params_after is not None and (iteration+1)%save_params_after == 0:
        cumul_loss /= save_params_after
        self.logger.debug('Iteration #{} train loss: {}'.format(iteration, cumul_loss))
        # Check if the cumulative loss is better than the best loss
        if best_loss is None:
          best_loss = cumul_loss
        if is_coarse:
          save_path = saver.save(self.sess,
                                 'trained_score_params/fcn_{}.ckpt'.format(iteration+1))
        else:
          save_path = saver.save(self.sess,
                                 'trained_params/fcn_{}.ckpt'.format(iteration+1))
        self.logger.debug('Model params after {} iterations saved to {}'
                         .format(iteration+1, save_path))
        if cumul_loss <= best_loss:
          self.logger.debug('Found best params! Saving to best_params/')
          os.system('rm -rf best_params')
          os.system('mkdir best_params')
          if is_coarse:
            os.system('cp trained_score_params/fcn_{}* best_params/'.format(iteration+1))
          else:
            os.system('cp trained_params/fcn_{}* best_params/'.format(iteration+1))
          best_loss = cumul_loss
        cumul_loss = 0.0

  def infer(self, img, restore_params=None):
    self.sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if restore_params is not None:
      saver.restore(self.sess, restore_params)
    img = np.reshape(img, [1, img.shape[0], img.shape[1], img.shape[2]])
    # ann = np.reshape(ann, [1, ann.shape[0], ann.shape[1]])
    output = self.sess.run(self.net['cropped_scores_final'],
                         feed_dict = {
                           self.net['input']: img,
                           #self.net['annotation']: ann
                         })
    output = np.squeeze(np.argmax(output, axis=3))
    self.logger.debug('Inference output shape: {}'.format(output.shape))
    return output
