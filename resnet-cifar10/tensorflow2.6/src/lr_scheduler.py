from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay

from src.config import config

def LRScheduler(batch_size):
  # Base LR
  lr = config['lr-base'] * batch_size / config['lr-batch-denom']
  # Step size
  step_size = (config['dataset-train-size'] + batch_size - 1) // batch_size
  # PiecewiseConstantDecay
  boundaries = [x * step_size - 1 for x in config['lr-decay-boundaries']]
  values=[lr]
  for _ in boundaries:
    values.append(values[-1] * config['lr-decay-rate'])
  # Warm up
  boundaries = [config['lr-warmup-epochs'] * step_size - 1] + boundaries
  values = [lr * config['lr-warmup-rate']] + values
  piece_wise_constant_decay = \
      PiecewiseConstantDecay(boundaries=boundaries, values=values)
  return piece_wise_constant_decay