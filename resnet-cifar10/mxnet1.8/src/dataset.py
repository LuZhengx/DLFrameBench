import mxnet as mx
from mxnet.gluon.nn import Block
from mxnet.gluon.data import vision, DataLoader
from mxnet.gluon.data.vision import transforms

from src.config import config

# Self implement for RandomCrop
# Random crop the images after padding
class RandomCrop(Block):
  def __init__(self, size, padding=0):
    super(RandomCrop, self).__init__()
    self.padding = padding
    self.size = size
  def forward(self, x):
    out = mx.image.copyMakeBorder(
        x, top=self.padding, bot=self.padding, left=self.padding, right=self.padding)
    return mx.image.random_crop(out, self.size)[0]

def Cifar10DataLoader(is_training, data_dir, batch_size,
            parallel_workers=config['dataset-num-workers']):
  if is_training:
    transform = transforms.Compose([
      RandomCrop(config['dataset-img-size'][1:], padding=config['dataset-img-pad']),
      transforms.RandomFlipLeftRight(),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=config['dataset-img-means'], std=config['dataset-img-stds'])
    ])
  else:
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(
          mean=config['dataset-img-means'], std=config['dataset-img-stds'])
    ])

  dataset = vision.datasets.CIFAR10(root=data_dir, train=is_training)
  dataset = dataset.transform_first(transform)

  dataloader = DataLoader(
      dataset, batch_size=batch_size, shuffle=is_training,
      last_batch='keep', num_workers=parallel_workers)
  
  return dataloader
