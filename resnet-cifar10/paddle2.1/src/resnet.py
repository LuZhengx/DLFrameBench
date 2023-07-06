import paddle
import paddle.nn.functional as F
import paddle.nn as nn

from src.config import config

__all__ = ['Resnet', 'resnet20', 'resnet56', 'resnet110', 'resnet1202']


def BatchNorm2D(channels):
  assert config['bn-gamma-init'] == 'Ones'
  weight_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(value=1.0))
  assert config['bn-beta-init'] == 'Zeros'
  bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(value=0.0))
  return nn.BatchNorm2D(
    channels, epsilon=config['bn-epsilon'], momentum=config['bn-momentum'],
    weight_attr=weight_attr, bias_attr=bias_attr)


def Conv2D(in_chs, out_chs, kernel_size, stride=1, padding=0):
  assert config['conv2d-weight-init'] == 'XavierUniform'
  weight_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
  return nn.Conv2D(
    in_chs, out_chs, kernel_size, stride, padding,
    weight_attr=weight_attr, bias_attr=config['conv2d-use-bias'])


class BasicBlock(paddle.nn.Layer):
  expansion = 1

  def __init__(self, in_chs, out_chs, stride=1, shortcut=None):
    super(BasicBlock, self).__init__()
    self.conv1 = Conv2D(in_chs, out_chs, kernel_size=3, stride=stride, padding=1)     
    self.bn1 = BatchNorm2D(out_chs)
    self.conv2 =  Conv2D(out_chs, out_chs, kernel_size=3, stride=1, padding=1)
    self.bn2 = BatchNorm2D(out_chs)
    self.shortcut = shortcut

  def forward(self, inputs):
    if self.shortcut is not None:
      shortcut = self.shortcut(inputs)
    else:
      shortcut = inputs

    x = self.conv1(inputs)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    output = F.relu(paddle.add(x, shortcut))
    return output


class ResNet(paddle.nn.Layer):

  def __init__(self, num_blocks, block=BasicBlock, num_classes=10):
    super(ResNet, self).__init__()
    
    # Initial conv2d
    self.in_chs = 16
    self.conv1 = Conv2D(3, self.in_chs, kernel_size=3, stride=1, padding=1)
    self.bn1 = BatchNorm2D(self.in_chs)

    # Residual blocks 1-3
    self.layer1 = self._make_layer(block, 16, num_blocks, 1)
    self.layer2 = self._make_layer(block, 32, num_blocks, 2)
    self.layer3 = self._make_layer(block, 64, num_blocks, 2)

    # Dense layer
    assert config['dense-weight-init'] == 'XavierUniform'
    weight_attr = paddle.ParamAttr(initializer=nn.initializer.XavierUniform())
    assert config['dense-bias-init'] == 'Zeros'
    bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(value=0.0))
    self.fc = paddle.nn.Linear(
        in_features=64, out_features=num_classes,
        weight_attr=weight_attr, bias_attr=bias_attr if config['dense-use-bias'] else False)
  
  def _make_layer(self, block, chs, num_blocks, stride):
    # Shortcut
    if config['downsample-shortcut'] == 'maxpool+padding':
      def shortcut(inputs):
        if stride != 1:
          inputs = F.max_pool2d(inputs, kernel_size=1, stride=2)
          zeroTensor_b = paddle.zeros([
              inputs.shape[0], chs // 4, inputs.shape[2], inputs.shape[3]])
          zeroTensor_a = paddle.zeros([
              inputs.shape[0], chs // 4, inputs.shape[2], inputs.shape[3]])
          inputs = paddle.concat([zeroTensor_b, inputs, zeroTensor_a], axis=1)
        return inputs
    elif config['downsample-shortcut'] == 'convolution':
      if stride != 1:
        shortcut = nn.Sequential(
            Conv2D(self.in_chs, chs, kernel_size=1, stride=stride),
            BatchNorm2D(chs)
        )
      else:
        shortcut = nn.Sequential()
    else:
      assert False

    layers = []
    # Only the first block per block_layer uses projection_shortcut and strides
    layers.append(block(self.in_chs, chs, stride=stride, shortcut=shortcut))
    self.in_chs = chs * block.expansion
    
    for _ in range(1, num_blocks):
      layers.append(block(self.in_chs, chs))

    return paddle.nn.Sequential(*tuple(layers))
  
  #@paddle.jit.to_static
  def forward(self, inputs):
    x = self.conv1(inputs)
    x = self.bn1(x)
    x = F.relu(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = paddle.mean(x, (2, 3), keepdim=False)
    output = self.fc(x)
    return output


def resnet20():
  return ResNet(3)

def resnet56():
  return ResNet(9)

def resnet110():
  return ResNet(18)

def resnet1202():
  return ResNet(200)
