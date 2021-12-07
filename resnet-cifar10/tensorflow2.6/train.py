# encoding: utf-8
import os
import argparse
import time
import ctypes

import tensorflow as tf
from tensorflow.keras import losses, optimizers

from src import resnet
from src.lr_scheduler import LRScheduler
from src.config import config
from src.dataset import Cifar10Dataset


def parse_arg():
  parser = argparse.ArgumentParser()
  parser.add_argument('--data-dir', nargs='?',
                      default='/data/cifar-10-batches-bin/', type=str)
  parser.add_argument('--arch', nargs='?', default="resnet20", type=str)
  parser.add_argument('--epochs', nargs='?', default=160, type=int)
  parser.add_argument('--batch-size', nargs='?', default=128, type=int)
  parser.add_argument('--data-format', nargs='?', default="channels_first", type=str)
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

  # Model
  model = resnet.__dict__[args.arch](args.data_format)

  # Dataset
  train_dataset = Cifar10Dataset(True, args.data_dir, args.batch_size, args.data_format)
  val_dataset = Cifar10Dataset(False, args.data_dir, args.batch_size, args.data_format)

  # LR & Optimizer
  piece_wise_constant_decay = LRScheduler(args.batch_size)
  assert config['optimizer-type'] == 'SGD'
  optimizer = optimizers.SGD(
      learning_rate=piece_wise_constant_decay,
      momentum=config['sgd-momentum'], nesterov=config['sgd-nesterov'])

  # Loss function
  criterion = losses.SparseCategoricalCrossentropy(from_logits=True)
  
  @tf.function
  def train_step(images, labels):
    with tf.GradientTape() as tape:
      preds = model(images, training=True)
      loss = criterion(labels, preds)
    
    # Backward propagation
    grads_and_vars = optimizer._compute_gradients(
        loss, var_list=model.trainable_variables, tape=tape)
    # Add weight decay
    decayed_grads_and_vars = [
      (g + v * config['sgd-weight-decay'], v) for g, v in grads_and_vars
    ]
    # Update weight
    optimizer.apply_gradients(decayed_grads_and_vars)
    return loss
    
  # Validation loss and accuracy
  test_loss = tf.keras.metrics.Mean(name='test_loss')
  test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

  @tf.function
  def val_step(images, labels):
    preds = model(images, training=False)
    loss = criterion(labels, preds)
    test_loss(loss)
    test_accuracy(labels, preds)

  total_train_time = 0
  step_size = (config['dataset-train-size'] + args.batch_size - 1) // args.batch_size
  # Main loop
  for epoch in range(args.epochs):
    # adjust the learning rate
    print(f"--------Epoch: {epoch:03}, lr: {optimizer._decayed_lr(tf.float32).numpy():.4f}--------")

    # Training process
    start_time = time.time()
    _cuda_tools_ext.nvtxRangePushA(ctypes.c_char_p(f"epoch:{epoch}".encode('utf-8')))
    for images, labels in train_dataset:
      loss = train_step(images, labels)
    
    # Training process end
    _cuda_tools_ext.nvtxRangePop()
    total_train_time += time.time() - start_time
    print(f"Train: Loss(last step): {loss:>.4e}, " +
          f"Batch Time: {(time.time() - start_time) * 1e3 /step_size:>.2f}ms")

    if not args.is_prof:
      # Reset metrics
      test_loss.reset_states()
      test_accuracy.reset_states()
      # Validation process
      for images, labels in val_dataset:
        val_step(images, labels)
      print(f"Test: Accuracy: {test_accuracy.result().numpy():>0.2%}, Avg loss: {test_loss.result().numpy():>.4e} ")
    
    # Print the training time
    print("Time used: %.2fs" % (total_train_time))

if __name__ == '__main__':
  main()