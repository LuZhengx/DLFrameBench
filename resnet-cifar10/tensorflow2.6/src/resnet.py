import tensorflow as tf
from src.config import config

__all__ = ['ResNet', 'resnet20', 'resnet56', 'resnet110', 'resnet1202']


def BatchNormalization(data_format):
  assert config['bn-gamma-init'] == 'Ones'
  assert config['bn-beta-init'] == 'Zeros'
  return tf.keras.layers.BatchNormalization(
      axis=1 if data_format == 'channels_first' else 3,
      momentum=config['bn-momentum'], epsilon=config['bn-epsilon'],
      beta_initializer='zeros', gamma_initializer='ones')


def Conv2D(filters, kernel_size, data_format, strides=1, padding='same'):
  assert config['conv2d-weight-init'] == 'XavierUniform'
  if strides > 1 and config['conv2d-pytorch-style']:
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    return tf.keras.Sequential([
      tf.keras.layers.ZeroPadding2D(
          padding=((pad_beg, pad_end), (pad_beg, pad_end)), data_format=data_format),
      tf.keras.layers.Conv2D(
          filters, kernel_size, strides=strides, padding="valid",
          data_format=data_format, use_bias=config['conv2d-use-bias'],
          kernel_initializer='glorot_uniform')
    ])
  else:
    return tf.keras.layers.Conv2D(
        filters, kernel_size, strides=strides, padding=padding,
        data_format=data_format, use_bias=config['conv2d-use-bias'],
        kernel_initializer='glorot_uniform')


class BasicBlock(tf.keras.layers.Layer):
	expansion = 1

	def __init__(self, filter_num, stride, data_format, shortcut=None):
		super(BasicBlock, self).__init__()

		self.conv1 = Conv2D(filter_num, 3, strides=stride, data_format=data_format)
		self.bn1 = BatchNormalization(data_format)
		self.conv2 = Conv2D(filter_num, 3, strides=1, data_format=data_format)
		self.bn2 = BatchNormalization(data_format)
		self.shortcut = shortcut

	def call(self, inputs, training=False):
		if self.shortcut is not None:
			residual = self.shortcut(inputs, training=training)
		else:
			residual = inputs

		x = self.conv1(inputs)
		x = self.bn1(x, training=training)
		x = tf.nn.relu(x)
		x = self.conv2(x)
		x = self.bn2(x, training=training)
		x += residual
		output = tf.nn.relu(x)
		return output


class ResNet(tf.keras.Model):

	def __init__(self, num_blocks, block=BasicBlock, num_classes=10, data_format='channels_last'):
		super(ResNet, self).__init__()

		self.data_format = data_format

		self.conv1 = Conv2D(16, 3, strides=1, data_format=data_format)
		self.bn1 = BatchNormalization(data_format)

		# Residual blocks 1-3
		self.layer1 = self._make_layer(block, 16, num_blocks, 1, data_format)
		self.layer2 = self._make_layer(block, 32, num_blocks, 2, data_format)
		self.layer3 = self._make_layer(block, 64, num_blocks, 2, data_format)

		# Dense layer
		assert config['dense-weight-init'] == 'XavierUniform'
		assert config['dense-bias-init'] == 'Zeros'
		self.fc = tf.keras.layers.Dense(
				units=num_classes, use_bias=config['dense-use-bias'],
				kernel_initializer='glorot_uniform', bias_initializer='zeros')

	def _make_layer(self, block, planes, num_blocks, stride, data_format):
    # Shortcut
		if config['downsample-shortcut'] == 'maxpool+padding':
			def shortcut(inputs, training=False):
				if stride != 1:
					if data_format=="channels_first":
						pooled_input = tf.nn.max_pool2d(inputs, ksize=[1, 1, 1, 1],
																						strides=[1, 1, 2, 2], padding='VALID',data_format="NCHW")
						padded_input = tf.pad(pooled_input, [[0, 0],[planes // 4,planes // 4],[0, 0], [0, 0]])
					else:
						pooled_input = tf.nn.max_pool2d(inputs, ksize=[1, 1, 1, 1],
																						strides=[1, 2, 2, 1], padding='VALID',data_format="NHWC")
						padded_input = tf.pad(pooled_input, [[0, 0],[0, 0], [0, 0],[planes // 4,planes // 4]])
				else:
					padded_input = inputs
				return padded_input
		elif config['downsample-shortcut'] == 'convolution':
			if stride != 1:
				shortcut = tf.keras.Sequential([
            Conv2D(planes, kernel_size=1, strides=stride, data_format=data_format),
            BatchNormalization(data_format)
				])
			else:
				shortcut = tf.keras.Sequential()
		else:
			assert False

		layers = []
		layers.append(block(planes, stride, data_format, shortcut=shortcut))

		for _ in range(1, num_blocks):
			layers.append(block(planes, 1, data_format))
			
		return tf.keras.Sequential(layers)
		
	def call(self, inputs, training=False):
		x = self.conv1(inputs)
		x = self.bn1(x, training=training)
		x = tf.nn.relu(x)
		x = self.layer1(x, training=training)
		x = self.layer2(x, training=training)
		x = self.layer3(x, training=training)
		axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
		x = tf.math.reduce_mean(x, axes)
		output = self.fc(x)

		return output

def resnet20(data_format='channels_last'):
    return ResNet(3, data_format=data_format)

def resnet56(data_format='channels_last'):
    return ResNet(9, data_format=data_format)

def resnet110(data_format='channels_last'):
    return ResNet(18, data_format=data_format)

def resnet1202(data_format='channels_last'):
    return ResNet(200, data_format=data_format)