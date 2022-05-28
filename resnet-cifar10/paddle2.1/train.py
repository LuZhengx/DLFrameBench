import os
import argparse
import ctypes
import time
import warnings

os.environ['FLAGS_cudnn_exhaustive_search']='True'

import paddle

from src import resnet
from src.lr_scheduler import WarmUpAndMultiStep
from src.dataset import Cifar10DataLoader
from src.config import config

warnings.filterwarnings("ignore", category=DeprecationWarning)

def parse_arg():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-file', nargs='?',
                      default='/data/cifar-10-python.tar.gz', type=str)
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
  
  # ResNet model
  model = resnet.__dict__[args.arch]()
  # spec = [paddle.static.InputSpec([None] + config['dataset-img-size'])]
  # model = paddle.jit.to_static(model, input_spec=spec)
  # DataLoader
  train_dataloader = Cifar10DataLoader(
      is_training=True, data_file=args.data_file, batch_size=args.batch_size)
  val_dataloader = Cifar10DataLoader(
      is_training=False, data_file=args.data_file, batch_size=args.batch_size)

  # Loss function
  loss_fn = paddle.nn.CrossEntropyLoss()
  # Optimizer
  assert config['optimizer-type'] == 'SGD'
  lr_scheduler = WarmUpAndMultiStep(args.batch_size)
  optimizer = paddle.optimizer.Momentum(
      learning_rate=lr_scheduler, parameters=model.parameters(),
      weight_decay=config['sgd-weight-decay'], momentum=config['sgd-momentum'],
      use_nesterov=config['sgd-nesterov'])

  total_train_time = 0
  step_size = (config['dataset-train-size'] + args.batch_size - 1) // args.batch_size
  # For momentum adjustment
  last_lr = lr_scheduler.last_lr
  last_mmt = config['sgd-momentum']
  # Main loop
  for epoch in range(args.epochs):
    # Log some infomations
    print(f"--------Epoch: {epoch:03}, " +
          f"lr: {optimizer._learning_rate():.4f}--------")

    # Training process
    model.train()
    start_time = time.time()
    _cuda_tools_ext.nvtxRangePushA(ctypes.c_char_p(f"epoch:{epoch}".encode('utf-8')))
    for images, labels in train_dataloader():
      # Adjust the momentum in optimizer
      # For the sake of fairness, don't do the adjustment in profiling
      if not args.is_prof:
        new_mmt = config['sgd-momentum'] * last_lr / lr_scheduler.last_lr
        if last_mmt != new_mmt:
          optimizer._momentum = new_mmt
          last_mmt = new_mmt
          # print(last_lr, last_mmt, lr_scheduler.last_lr)
        last_lr = lr_scheduler.last_lr
      # Compute prediction and loss
      logits = model(images)
      loss = loss_fn(logits, labels)
      # Backpropagation
      optimizer.clear_grad()
      loss.backward()
      # Update
      optimizer.step()

    # One step for lr scheduler
    lr_scheduler.step()
    # Training end, wait for all compute on gpu complete
    loss = loss.numpy()
    _cuda_tools_ext.nvtxRangePop()
    total_train_time += time.time() - start_time
    print(f"Train: Loss(last step): {loss[0]:>.4e}, " + 
          f"Batch Time: {(time.time() - start_time) * 1e3 /step_size:>.2f}ms")

    if not args.is_prof:
      # Validation process
      model.eval()
      loss, correct = 0, 0
      with paddle.no_grad():
        for images, labels in val_dataloader():
          # Forward compute
          logits = model(images)
          loss += loss_fn(logits, labels).numpy()[0] * labels.shape[0]
          correct += (logits.argmax(1) == labels).numpy().sum()
        avg_acc = correct/config['dataset-val-size']
        avg_loss = loss/config['dataset-val-size']
      print(f"Test: Accuracy: {(avg_acc):>0.2%}, Avg loss: {avg_loss:>.4e} ")
    
    # Print the training time
    print("Time used: %.2fs" % (total_train_time))

if __name__ == '__main__':
  main()