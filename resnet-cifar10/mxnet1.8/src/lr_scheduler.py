from src.config import config

class WarmUpAndMultiStepLR():
  """Decays the learning rate of each parameter group by gamma once the
  number of epoch reaches one of the milestones. Notice that such decay can
  happen simultaneously with other changes to the learning rate from outside
  this scheduler. When last_epoch=-1, sets initial lr as lr.

  Args:
    optimizer (Optimizer): Wrapped optimizer.
    milestones (list): List of epoch indices. Must be increasing.
    gamma (float): Multiplicative factor of learning rate decay.
      Default: 0.1.
    warmup_gamma: lr rate in warmup phase.
    warmup_epochs: the number of epochs to warmup.
  """

  def __init__(self, batch_size, milestones=config['lr-decay-boundaries'],
               gamma=config['lr-decay-rate'], warmup_gamma=config['lr-warmup-rate'],
               warmup_epochs=config['lr-warmup-epochs']):
    # Base learning rate
    self.base_lr = config['lr-base'] * batch_size / config['lr-batch-denom']
    # Number of step per epoch
    step_size = (config['dataset-train-size'] + batch_size - 1) // batch_size
    
    # Warm up
    self.warmup_gamma = warmup_gamma
    self.warmup_epochs = warmup_epochs * step_size

    # Reduce the learning rate at certain epochs.
    self.milestones = [ms * step_size for ms in milestones]
    self.vals = [self.base_lr]
    for i in range(len(self.milestones)):
      self.vals.append(self.vals[i] * gamma)

  def __call__(self, iteration):
    if iteration < self.warmup_epochs:
      return self.base_lr * self.warmup_gamma

    for ms, lr in zip(self.milestones, self.vals):
      if iteration < ms:
        return lr
    return self.vals[-1]
