import tensorflow as tf

from src.config import config

__all__ = ['resnet', 'resnet20', 'resnet56', 'resnet110', 'resnet1202']

def batch_norm(inputs, training, data_format):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  assert config['bn-gamma-init'] == 'Ones'
  assert config['bn-beta-init'] == 'Zeros'
  return tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=config['bn-momentum'], epsilon=config['bn-epsilon'],
      beta_initializer=tf.zeros_initializer(),
      gamma_initializer=tf.ones_initializer(),
      training=training, fused=True)


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).
  assert config['conv2d-weight-init'] == 'XavierUniform'

  if strides > 1 and config['conv2d-pytorch-style']:
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    if data_format == 'channels_first':
      inputs = tf.pad(inputs, [[0, 0], [0, 0],
                      [pad_beg, pad_end], [pad_beg, pad_end]])
    else:
      inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                      [pad_beg, pad_end], [0, 0]])
  
  return tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('VALID' if strides > 1 and config['conv2d-pytorch-style'] else 'SAME'),
      kernel_initializer=tf.glorot_uniform_initializer(),
      data_format=data_format)


def basic_block(inputs, filters, training, projection_shortcut, strides,
                data_format):
  """A single block for ResNet v1, without a bottleneck."""

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
  else:
    shortcut = inputs

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)

  inputs += shortcut
  inputs = tf.nn.relu(inputs)

  return inputs


class resnet(object):
  """Base class for building the Resnet Model."""

  def __init__(self, resnet_size, block=basic_block, num_classes=10, data_format=None):
    """Creates a model for classifying an image.

    Args:
      resnet_size: A single integer for the size of the ResNet model.
      block: Residual block used in the network.
      num_classes: The number of classes used as labels.
      data_format: Input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
    """
    self.resnet_size = resnet_size
    self.block_fn = block
    self.num_classes = num_classes

    if not data_format:
      data_format = (
          'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')
    self.data_format = data_format

  def __call__(self, inputs, training, reuse=None):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, self.num_classes].
    """
    if reuse == None:
      reuse = not training
    with tf.variable_scope('resnet_model', reuse=reuse):
      # Initial conv2d
      inputs = conv2d_fixed_padding(
          inputs=inputs, filters=16, kernel_size=3,
          strides=1, data_format=self.data_format)
      inputs = batch_norm(inputs, training, self.data_format)
      inputs = tf.nn.relu(inputs)

      num_filters = [16, 32, 64]
      num_strides = [1, 2, 2]
      # Residual blocks 1-3
      for i, (filters, strides) in enumerate(zip(num_filters, num_strides)):
        inputs = self._block_layer(
            inputs=inputs, block_fn=self.block_fn, filters=filters,
            num_blocks=self.resnet_size, strides=strides, training=training,
            data_format=self.data_format)

      # The current top layer has shape
      # `batch_size x pool_size x pool_size x final_size`.
      axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
      inputs = tf.reduce_mean(inputs, axes)

      assert config['dense-weight-init'] == 'XavierUniform'
      assert config['dense-bias-init'] == 'Zeros'
      inputs = tf.layers.dense(
          inputs=inputs, units=self.num_classes, use_bias=config['dense-use-bias'],
          kernel_initializer=tf.glorot_uniform_initializer(),
          bias_initializer=tf.zeros_initializer())
      return inputs

  def _block_layer(self, inputs, block_fn, filters, num_blocks, strides,
                  training, data_format):
    """Creates one layer of blocks for the ResNet model."""
    
    def projection_shortcut(inputs):
      if config['downsample-shortcut'] == 'maxpool+padding':
        if strides != 1:
          if data_format=="channels_first":
            input_channel = inputs.get_shape().as_list()[1]
            pooled_input = tf.nn.max_pool2d(inputs, ksize=[1, 1, 1, 1],
                                            strides=[1, 1, 2, 2], padding='VALID',data_format="NCHW")
            padded_input = tf.pad(pooled_input, [[0, 0],[input_channel // 2,input_channel // 2],[0, 0], [0, 0]])
          else:
            input_channel = inputs.get_shape().as_list()[-1]
            pooled_input = tf.nn.max_pool2d(inputs, ksize=[1, 1, 1, 1],
                                            strides=[1, 2, 2, 1], padding='VALID',data_format="NHWC")
            padded_input = tf.pad(pooled_input, [[0, 0],[0, 0], [0, 0],[input_channel // 2,input_channel // 2]])        
        else:
          padded_input = inputs
        return padded_input
      elif config['downsample-shortcut'] == 'convolution':
        if strides != 1:
          conved_input = conv2d_fixed_padding(
              inputs=inputs, filters=filters, kernel_size=1, strides=2, data_format=data_format)
          bned_input = batch_norm(conved_input, training, data_format)
        else:
          bned_input = inputs
        return bned_input
      assert False
      
    # Only the first block per block_layer uses projection_shortcut and strides
    inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                      data_format)

    for _ in range(1, num_blocks):
      inputs = block_fn(inputs, filters, training, None, 1, data_format)

    return inputs


def resnet20(data_format=None):
    return resnet(3, data_format=data_format)

def resnet56(data_format=None):
    return resnet(9, data_format=data_format)

def resnet110(data_format=None):
    return resnet(18, data_format=data_format)

def resnet1202(data_format=None):
    return resnet(200, data_format=data_format)
