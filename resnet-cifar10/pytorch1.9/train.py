import time
import argparse
import ctypes

import torch

from src import resnet
from src.lr_scheduler import WarmUpAndMultiStepLR
from src.dataset import Cifar10DataLoader
from src.config import config

def parse_arg():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', nargs='?',
                      default='/data', type=str)
  parser.add_argument('--arch', nargs='?', default="resnet20", type=str)
  parser.add_argument('--epochs', nargs='?', default=160, type=int)
  parser.add_argument('--batch-size', nargs='?', default=128, type=int)
  parser.add_argument('--is-prof', action='store_true')
  args = parser.parse_args()
  return args

def main():
  args = parse_arg()
  print("---------configurations--------------")
  for k, v in vars(args).items():
    print(k,':',v)
  print("-------------------------------------")

  _cuda_tools_ext = ctypes.CDLL("libnvToolsExt.so")
  torch.backends.cudnn.benchmark = True

  # ResNet model
  model = resnet.__dict__[args.arch]()
  model = model.cuda()

  # DataLoader
  train_dataloader = Cifar10DataLoader(
      is_training=True, data_dir=args.data_dir, batch_size=args.batch_size)
  val_dataloader = Cifar10DataLoader(
      is_training=False, data_dir=args.data_dir, batch_size=args.batch_size)

  # Loss function
  loss_fn = torch.nn.CrossEntropyLoss().cuda()
  # Optimizer
  assert config['optimizer-type'] == 'SGD'
  lr = config['lr-base'] * args.batch_size / config['lr-batch-denom']
  optimizer = torch.optim.SGD(
      model.parameters(), lr=lr, weight_decay=config['sgd-weight-decay'],
      momentum=config['sgd-momentum'], nesterov=config['sgd-nesterov'])

  # LR scheduler, using warmup
  lr_scheduler = WarmUpAndMultiStepLR(optimizer)

  total_train_time = 0
  step_size = (config['dataset-train-size'] + args.batch_size - 1) // args.batch_size
  # For momentum adjustment
  last_lrs = lr_scheduler.get_last_lr()
  last_mmt = config['sgd-momentum']
  # Main loop
  for epoch in range(args.epochs):
    # Log some infomations
    print(f"--------Epoch: {epoch:03}, " +
          f"lr: {optimizer.param_groups[0]['lr']:.4f}--------")
    
    # Training process
    model.train()
    start_time = time.time()
    _cuda_tools_ext.nvtxRangePushA(ctypes.c_char_p(f"epoch:{epoch}".encode('utf-8')))
    _cuda_tools_ext.nvtxRangePushA(ctypes.c_char_p(f"prepare data".encode('utf-8')))
    for X, y in train_dataloader:
      # Adjust the momentum in optimizer
      # For the sake of fairness, don't do the adjustment in profiling
      if not args.is_prof:
        new_mmt = [config['sgd-momentum'] * llr / nlr
                   for (llr, nlr) in zip(last_lrs, lr_scheduler.get_last_lr())]
        if last_mmt != new_mmt:
          opt_state_dict = optimizer.state_dict()
          for i, mmt in enumerate(new_mmt):
            opt_state_dict['param_groups'][i]['momentum'] = mmt
          optimizer.load_state_dict(opt_state_dict)
          # print(opt_state_dict['param_groups'][0])
          last_mmt = new_mmt
        last_lrs = lr_scheduler.get_last_lr()

      # Move to gpu
      X = X.cuda()
      y = y.cuda()
      # Compute prediction and loss
      _cuda_tools_ext.nvtxRangePop()
      _cuda_tools_ext.nvtxRangePushA(ctypes.c_char_p(f"forward".encode('utf-8')))
      pred = model(X)
      loss = loss_fn(pred, y)
      # Backpropagation
      _cuda_tools_ext.nvtxRangePop()
      _cuda_tools_ext.nvtxRangePushA(ctypes.c_char_p(f"gradient clean".encode('utf-8')))
      optimizer.zero_grad()
      _cuda_tools_ext.nvtxRangePop()
      _cuda_tools_ext.nvtxRangePushA(ctypes.c_char_p(f"backpropagation".encode('utf-8')))
      loss.backward()
      # Update
      _cuda_tools_ext.nvtxRangePop()
      _cuda_tools_ext.nvtxRangePushA(ctypes.c_char_p(f"gradient update".encode('utf-8')))
      optimizer.step()
      _cuda_tools_ext.nvtxRangePop()
      _cuda_tools_ext.nvtxRangePushA(ctypes.c_char_p(f"prepare data".encode('utf-8')))

    _cuda_tools_ext.nvtxRangePop()
    # One step for lr scheduler
    lr_scheduler.step()
    # Training end, wait for all compute on gpu complete
    loss = loss.cpu()
    _cuda_tools_ext.nvtxRangePop()
    total_train_time += time.time() - start_time
    print(f"Train: Loss(last step): {loss:>.4e}, " + 
          f"Batch Time: {(time.time() - start_time) * 1e3 /step_size:>.2f}ms")

    if not args.is_prof:
      # Validation process
      model.eval()
      loss, correct = 0, 0
      with torch.no_grad():
        for X, y in val_dataloader:
          # Move to gpu
          X = X.cuda()
          y = y.cuda()
          # Forward compute
          pred = model(X)
          loss += loss_fn(pred, y).item() * y.shape[0]
          correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        loss /= config['dataset-val-size']
        correct /= config['dataset-val-size']
      print(f"Test: Accuracy: {(correct):>0.2%}, Avg loss: {loss:>.4e} ")
    
    # Print the training time
    print("Time used: %.2fs" % (total_train_time))


if __name__ == '__main__':
  main()
