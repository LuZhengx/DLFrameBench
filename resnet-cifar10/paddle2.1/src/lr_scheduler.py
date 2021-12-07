from paddle.optimizer.lr import PiecewiseDecay

from src.config import config

def WarmUpAndMultiStep(batch_size, boundary_epochs=config['lr-decay-boundaries'],
    decay_rates=config['lr-decay-rate'], base_lr=config['lr-base'], last_epoch=-1,
    warmup_rate=config['lr-warmup-rate'], warmup_epochs=config['lr-warmup-epochs']):

  base_lr = config['lr-base'] * batch_size / config['lr-batch-denom']

  # Reduce the learning rate at certain epochs.
  boundaries = [epoch for epoch in boundary_epochs]
  vals = [base_lr]
  for i in range(len(boundaries)):
    vals.append(vals[i] * decay_rates)

  # Warm up
  boundaries = [warmup_epochs] + boundaries
  vals = [base_lr * warmup_rate] + vals

  return PiecewiseDecay(boundaries=boundaries, values=vals, last_epoch=last_epoch)
