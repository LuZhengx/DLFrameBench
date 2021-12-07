import os
import time
import argparse
import ctypes

os.environ["MXNET_USE_FUSION"] = "0"

import mxnet

from src import resnet
from src.lr_scheduler import WarmUpAndMultiStepLR
from src.dataset import Cifar10DataLoader
from src.config import config

def parse_arg():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', nargs='?',
                      default='/data/cifar-10-batches-bin', type=str)
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
  model.hybridize()
  model.initialize(ctx=mxnet.gpu())

  # DataLoader
  train_dataloader = Cifar10DataLoader(
      is_training=True, data_dir=args.data_dir, batch_size=args.batch_size)
  val_dataloader = Cifar10DataLoader(
      is_training=False, data_dir=args.data_dir, batch_size=args.batch_size)

  # Loss function
  loss_fn = mxnet.gluon.loss.SoftmaxCrossEntropyLoss()
  # Optimizer
  assert config['optimizer-type'] == 'SGD'
  assert not config['sgd-nesterov']
  lr_scheduler = WarmUpAndMultiStepLR(args.batch_size)
  optimizer = mxnet.optimizer.SGD(
      momentum=config['sgd-momentum'], lr_scheduler=lr_scheduler,
      wd=config['sgd-weight-decay'])
  trainer = mxnet.gluon.Trainer(model.collect_params(), optimizer=optimizer)

  total_train_time = 0
  step_size = (config['dataset-train-size'] + args.batch_size - 1) // args.batch_size
  # Main loop
  for epoch in range(args.epochs):
    # Log some infomations
    print(f"--------Epoch: {epoch:03}, " +
          f"lr: {trainer.learning_rate:.4f}--------")

    # Training process
    start_time = time.time()
    _cuda_tools_ext.nvtxRangePushA(ctypes.c_char_p(f"epoch:{epoch}".encode('utf-8')))
    for images, label in train_dataloader:
      # Move to gpu
      images = images.copyto(mxnet.gpu())
      label = label.copyto(mxnet.gpu())
      # Compute prediction and loss
      with mxnet.autograd.record():
        logit = model(images)
        loss = loss_fn(logit, label)
      # Backpropagation
      loss.backward()
      # Update
      trainer.step(args.batch_size)
    # Training end, wait for all compute on gpu complete
    loss = loss.mean().asscalar()
    _cuda_tools_ext.nvtxRangePop()
    total_train_time += time.time() - start_time
    print(f"Train: Loss(last step): {loss:>.4e}, " + 
          f"Batch Time: {(time.time() - start_time) * 1e3 /step_size:>.2f}ms")
    
    if not args.is_prof:
      # Validation process
      loss, correct = 0, 0
      with mxnet.autograd.pause():
        for data, label in val_dataloader:
          # Move to gpu
          data = data.copyto(mxnet.gpu())
          label = label.copyto(mxnet.gpu())
          # Forward compute
          logit = model(data)
          loss += loss_fn(logit, label).sum().asscalar()
          correct += (logit.argmax(axis=1).astype('int32') == label).astype('int32').sum().asscalar()
        loss /= config['dataset-val-size']
        correct /= config['dataset-val-size']
      print(f"Test: Accuracy: {(correct):>0.2%}, Avg loss: {loss:>.4e} ")
    
    # Print the training time
    print("Time used: %.2fs" % (total_train_time))
    
if __name__ == '__main__':
  main()