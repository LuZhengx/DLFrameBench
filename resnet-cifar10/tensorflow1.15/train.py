
import time
import argparse
import ctypes
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

from src import resnet
from src.dataset import cifar10_dataset
from src.lr_scheduler import learning_rate_with_decay
from src.config import config

import tensorflow as tf

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

def build_train_validation_graph(input_data, input_label, args,
                                 training, lr_scheduler_fn=None):
  '''
  This function builds the train graph or validation graphs
  
  Returns:
    loss, train_op or valid_op
  '''
  # Calc the logit
  model = resnet.__dict__[args.arch](data_format=args.data_format)
  logits = model(input_data, training=training)

  # Calc the cross entropy loss
  cross_entropy = tf.losses.sparse_softmax_cross_entropy(
      logits=logits, labels=input_label)

  if training:
    # Get learning rate
    global_step = tf.train.get_or_create_global_step()
    learning_rate = lr_scheduler_fn(global_step)

    # Calc gradient
    assert config['optimizer-type'] == 'SGD'
    optimizer = tf.train.MomentumOptimizer(
      learning_rate=learning_rate,
      momentum=config['sgd-momentum'],
      use_nesterov=config['sgd-nesterov']
    )
    grads_vars = optimizer.compute_gradients(cross_entropy)
    # Add weight decay
    decayed_grads_vars = [(g + v * config['sgd-weight-decay'], v)
                          for g, v in grads_vars]

    # Apply gradient and update params
    minimize_op = optimizer.apply_gradients(decayed_grads_vars, global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group([minimize_op, update_ops])

    return cross_entropy, train_op
  else:
    # Calc top-1 accuracy
    pred = tf.argmax(logits, axis=1)
    _, accuracy = tf.metrics.accuracy(input_label, pred, name="acc_metric")
    tf.summary.scalar('train_accuracy', accuracy)
    
    return cross_entropy, accuracy


def main():
  # Parse args
  args = parse_arg()
  print("---------configurations--------------")
  for k, v in vars(args).items():
      print(k,':',v)
  print("-------------------------------------")

  _cuda_tools_ext = ctypes.CDLL("libnvToolsExt.so")
  
  # Training set
  train_dataset = cifar10_dataset(
      True, args.data_dir, args.batch_size, args.data_format)
  train_iterator = train_dataset.make_initializable_iterator()
  train_data, train_label = train_iterator.get_next()

  # Build training graph
  lr_scheduler = learning_rate_with_decay(args.batch_size)
  train_loss, train_op = build_train_validation_graph(
      train_data, train_label, args, True, lr_scheduler)

  # Validation set
  val_dataset = cifar10_dataset(
      False, args.data_dir, args.batch_size, args.data_format)
  val_iterator = val_dataset.make_initializable_iterator()
  val_data, val_label = val_iterator.get_next()

  # Build validation graph
  val_loss, val_acc = build_train_validation_graph(
      val_data, val_label, args, False)

  # metric initializer
  metric_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="acc_metric")
  metric_vars_initializer = tf.variables_initializer(var_list=metric_vars)

  # Start training
  total_train_time, step = 0, 0
  step_size = (config['dataset-train-size'] + args.batch_size - 1) // args.batch_size
  
  with tf.Session() as sess:
    # Init variable
    sess.run(tf.global_variables_initializer())

    # Main loop
    for epoch in range(args.epochs):
      # Log some infomations
      print(f"--------Epoch: {epoch:03}, " +
            f"lr: {sess.run(lr_scheduler(tf.constant(step))):.4f}--------")
    
      # Training process
      start_time = time.time()
      _cuda_tools_ext.nvtxRangePushA(ctypes.c_char_p(f"epoch:{epoch}".encode('utf-8')))
      # Prepare training dataset
      sess.run(train_iterator.initializer)
      while True:
        try:
          loss, _ = sess.run([train_loss, train_op])
        except tf.errors.OutOfRangeError:
          break
      # Training end, wait for all compute on gpu complete
      _cuda_tools_ext.nvtxRangePop()
      total_train_time += time.time() - start_time
      print(f"Train: Loss(last step): {loss:>.4e}, " + 
            f"Batch Time: {(time.time() - start_time) * 1e3 /step_size:>.2f}ms")
      step += step_size

      if not args.is_prof:
        # Reset accuracy metric
        sess.run(metric_vars_initializer)
        # Prepare validation dataset
        sess.run(val_iterator.initializer)
        # Validation process
        while True:
          try:
            loss, acc = sess.run([val_loss, val_acc])
          except tf.errors.OutOfRangeError:
            break
        # Validation process end
        print(f"Test: Accuracy: {(acc):>0.2%}, Loss(last step): {loss:>.4e} ")

      # Print the training time
      print("Time used: %.2fs" % (total_train_time))

if __name__ == '__main__':
  main()