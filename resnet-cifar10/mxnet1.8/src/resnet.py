from mxnet.gluon import nn
from mxnet import initializer

from src.config import config

__all__ = ['ResNet', 'resnet20', 'resnet56', 'resnet110', 'resnet1202']

def BatchNorm():
  assert config['bn-gamma-init'] == 'Ones'
  assert config['bn-beta-init'] == 'Zeros'
  return nn.BatchNorm(
      axis=1, momentum=config['bn-momentum'], epsilon=config['bn-epsilon'],
      beta_initializer='zeros', gamma_initializer='ones',
      running_mean_initializer='zeros', running_variance_initializer='ones')

def Conv2D(channels, kernel_size, strides=1, padding=0):
  assert config['conv2d-weight-init'] == 'XavierUniform'
  return nn.Conv2D(
      channels, kernel_size, strides=strides, padding=padding, layout='NCHW',
      use_bias=config['conv2d-use-bias'], 
      weight_initializer=initializer.Xavier(rnd_type='uniform'))

class BasicBlock(nn.HybridBlock):
  expansion = 1

  def __init__(self, planes, stride=1, shortcut=None):
    super(BasicBlock, self).__init__()
    self.conv1 = Conv2D(planes, 3, strides=stride, padding=1)
    self.bn1 = BatchNorm()
    self.conv2 = Conv2D(planes, 3, strides=1, padding=1)
    self.bn2 = BatchNorm()
    self.shortcut = shortcut
    
  def hybrid_forward(self, F, x):
    if self.shortcut is not None:
      if config['downsample-shortcut'] == 'convolution':
        shortcut = self.shortcut(x)
      else:
        shortcut = self.shortcut(x, F)
    else:
      shortcut = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out = out + shortcut
    out = F.relu(out)
    return out

class ResNet(nn.HybridBlock):

  def __init__(self, num_blocks, block=BasicBlock, num_classes=10):
    super(ResNet, self).__init__()
    
    # Initial conv2d
    self.in_planes = 16
    self.conv1 = Conv2D(self.in_planes, 3, strides=1, padding=1)
    self.bn1 = BatchNorm()
    
    # Residual blocks 1-3
    self.layer1 = self._make_layer(block, 16, num_blocks, stride=1)
    self.layer2 = self._make_layer(block, 32, num_blocks, stride=2)
    self.layer3 = self._make_layer(block, 64, num_blocks, stride=2)

    # Dense layer
    assert config['dense-weight-init'] == 'XavierUniform'
    assert config['dense-bias-init'] == 'Zeros'
    self.linear = nn.Dense(
        num_classes, use_bias=config['dense-use-bias'], flatten=False,
        weight_initializer=initializer.Xavier(rnd_type='uniform'),
        bias_initializer='zeros')

  def _make_layer(self, block, planes, num_blocks, stride):
    # Shortcut
    if config['downsample-shortcut'] == 'maxpool+padding':
      def shortcut(x, F):
        if stride != 1:
          x = F.Pooling(x, kernel=(1, 1), pool_type='max', stride=(2, 2))
          x = F.transpose(x, axes=(0, 2, 3, 1))
          x = F.pad(x, mode='constant', pad_width=(0,0,0,0,0,0,planes//4,planes//4))
          x = F.transpose(x, axes=(0, 3, 1, 2))
        return x
    elif config['downsample-shortcut'] == 'convolution':
      shortcut = nn.HybridSequential()
      if stride != 1:
        shortcut.add(
          Conv2D(planes, kernel_size=1, strides=stride),
          BatchNorm()
        )
    else:
      assert False
      
    layers = nn.HybridSequential()
    # Only the first block per block_layer uses projection_shortcut and strides
    layers.add(block(planes, stride=stride, shortcut=shortcut))
    self.in_planes = planes * block.expansion

    for _ in range(1, num_blocks):
      layers.add(block(planes))

    return layers

  def hybrid_forward(self, F, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = F.mean(out, axis=(2, 3), keepdims=False)
    out = self.linear(out)
    return out


def resnet20():
  return ResNet(3, BasicBlock)

def resnet56():
  return ResNet(9, BasicBlock)

def resnet110():
  return ResNet(18, BasicBlock)

def resnet1202():
  return ResNet(200, BasicBlock)
