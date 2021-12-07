import warnings

from torch.optim.lr_scheduler import _LRScheduler

from src.config import config

class WarmUpAndMultiStepLR(_LRScheduler):
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
    last_epoch (int): The index of last epoch. Default: -1.
    verbose (bool): If ``True``, prints a message to stdout for
      each update. Default: ``False``.
  """

  def __init__(self, optimizer, milestones=config['lr-decay-boundaries'],
               gamma=config['lr-decay-rate'],
               warmup_gamma=config['lr-warmup-rate'],
               warmup_epochs=config['lr-warmup-epochs'],
               last_epoch=-1, verbose=False):
    self.milestones = milestones
    self.gamma = gamma
    self.warmup_gamma = warmup_gamma
    self.warmup_epochs = warmup_epochs
    super(WarmUpAndMultiStepLR, self).__init__(optimizer, last_epoch, verbose)

  def get_lr(self):
    if not self._get_lr_called_within_step:
      warnings.warn("To get the last learning rate computed by the scheduler, "
              "please use `get_last_lr()`.", UserWarning)
    
    if self.last_epoch < self.warmup_epochs:
      return [lr * self.warmup_gamma for lr in self.base_lrs]
    else:
      lr_scale = 1.0
      for ms in self.milestones:
        if self.last_epoch < ms:
          break
        lr_scale = lr_scale * self.gamma
    return [lr_scale * lr for lr in self.base_lrs]
