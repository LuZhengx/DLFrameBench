import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from src.config import config

def Cifar10DataLoader(is_training, data_dir, batch_size,
                      parallel_workers=config['dataset-num-workers']):
  if is_training:
    transform = transforms.Compose([
      transforms.RandomCrop(
          config['dataset-img-size'][1:], padding=config['dataset-img-pad']),
      transforms.RandomHorizontalFlip(),
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

  dataset = torchvision.datasets.CIFAR10(
    root=data_dir, train=is_training, download=False, transform=transform)

  dataloader = DataLoader(
      dataset, batch_size=batch_size, shuffle=is_training, pin_memory=True,
      num_workers=parallel_workers, persistent_workers=(parallel_workers > 0))
  
  return dataloader