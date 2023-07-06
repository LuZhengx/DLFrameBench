import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from src.config import config

__all__ = ['ResNet', 'resnet20', 'resnet56', 'resnet110', 'resnet1202']

def BatchNorm2d(channels):
  return nn.BatchNorm2d(
      channels, eps=config['bn-epsilon'], momentum=1-config['bn-momentum'])

def Conv2d(in_chs, out_chs, kernel_size, stride=1, padding=0):
  return nn.Conv2d(
      in_chs, out_chs, kernel_size, stride, padding, bias=config['conv2d-use-bias'])

class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_chs, out_chs, stride=1, shortcut=None):
    super(BasicBlock, self).__init__()
    self.conv1 = Conv2d(in_chs, out_chs, kernel_size=3, stride=stride, padding=1)
    self.bn1 = BatchNorm2d(out_chs)
    self.conv2 = Conv2d(out_chs, out_chs, kernel_size=3, stride=1, padding=1)
    self.bn2 = BatchNorm2d(out_chs)
    self.shortcut = shortcut

  def forward(self, x):
    if self.shortcut is not None:
      shortcut = self.shortcut(x)
    else:
      shortcut = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out)
    out = self.conv2(out)
    out = self.bn2(out)
    out += shortcut
    out = F.relu(out)
    return out

class ResNet(nn.Module):
  
  def __init__(self, num_blocks, block=BasicBlock, num_classes=10):
    super(ResNet, self).__init__()

    # Initial conv2d
    self.in_chs = 16
    self.conv1 = Conv2d(3, self.in_chs, kernel_size=3,stride=1, padding=1)
    self.bn1 = BatchNorm2d(self.in_chs)

    # Residual blocks 1-3
    self.layer1 = self._make_layer(block, 16, num_blocks, stride=1)
    self.layer2 = self._make_layer(block, 32, num_blocks, stride=2)
    self.layer3 = self._make_layer(block, 64, num_blocks, stride=2)
    
    # Dense layer
    self.linear = nn.Linear(64, num_classes, bias=config['dense-use-bias'])

    # Params init
    self.apply(self._weights_init)

  def _make_layer(self, block, chs, num_blocks, stride):
    # Shortcut
    if config['downsample-shortcut'] == 'maxpool+padding':
      def shortcut(x):
        if stride != 1:
          x = F.max_pool2d(x, 1, 2)
          x = F.pad(x, (0, 0, 0, 0, chs//4, chs//4))
        return x
    elif config['downsample-shortcut'] == 'convolution':
      if stride != 1:
        shortcut = nn.Sequential(
          Conv2d(self.in_chs, chs, kernel_size=1, stride=stride),
          BatchNorm2d(chs)
        )
      else:
        shortcut = nn.Identity()
    else:
      assert False
      
    layers = []
    # Only the first block per block_layer uses projection_shortcut and strides
    layers.append(block(self.in_chs, chs, stride=stride, shortcut=shortcut))
    self.in_chs = chs * block.expansion

    for _ in range(1, num_blocks):
      layers.append(block(self.in_chs, chs))

    return nn.Sequential(*layers)

  def _weights_init(self, m):
    if isinstance(m, nn.Conv2d):
      assert config['conv2d-weight-init'] == 'XavierUniform'
      init.xavier_uniform_(m.weight)

    if isinstance(m, nn.BatchNorm2d):
      assert config['bn-gamma-init'] == 'Ones'
      assert config['bn-beta-init'] == 'Zeros'
      init.ones_(m.weight)
      init.zeros_(m.bias)

    if isinstance(m, nn.Linear):
      assert config['dense-weight-init'] == 'XavierUniform'
      assert config['dense-bias-init'] == 'Zeros'
      init.xavier_uniform_(m.weight)
      init.zeros_(m.bias)
      
  def forward(self, x):
    out = self.conv1(x)
    out = self.bn1(out)
    out = F.relu(out)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = torch.mean(out, (2, 3), keepdim=False)
    out = self.linear(out)
    return out


def resnet20():
  return ResNet(3, BasicBlock)
  
def resnet38():
  return ResNet(6, BasicBlock)

def resnet56():
  return ResNet(9, BasicBlock)

def resnet110():
  return ResNet(18, BasicBlock)

def resnet218():
  return ResNet(36, BasicBlock)

def resnet434():
  return ResNet(72, BasicBlock)

def resnet866():
  return ResNet(144, BasicBlock)

def resnet1202():
  return ResNet(200, BasicBlock)
