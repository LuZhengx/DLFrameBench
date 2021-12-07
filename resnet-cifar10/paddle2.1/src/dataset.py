from paddle.vision import transforms, Cifar10
from paddle.io import DataLoader

from src.config import config

def Cifar10DataLoader(is_training, data_file, batch_size,
                      parallel_workers=config['dataset-num-workers']):
  if is_training:
    transform = transforms.Compose([
      transforms.Pad(padding=config['dataset-img-pad']),
      transforms.RandomCrop(size=config['dataset-img-size'][1:]),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(
          mean=config['dataset-img-means'],std=config['dataset-img-stds'])
    ])
  else:
    transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(
          mean=config['dataset-img-means'], std=config['dataset-img-stds'])
    ])
  
  cifar10_dataset = Cifar10(data_file=data_file,
                            mode='train' if is_training else 'test',
                            download=False,
                            transform=transform)
  
  return DataLoader(cifar10_dataset, shuffle=is_training,
                    batch_size=batch_size, num_workers=parallel_workers)
