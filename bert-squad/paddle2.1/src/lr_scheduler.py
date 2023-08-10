from paddle.optimizer.lr import LRScheduler

class LinearWarmUpScheduler(LRScheduler):
    """
    Applies a warm up period to the learning rate.
    """

    def __init__(self, lr, warmup, total_steps, last_epoch=-1):
        self.warmup = warmup
        self.total_steps = total_steps
        super(LinearWarmUpScheduler, self).__init__(lr, last_epoch)

    def get_lr(self):
        progress = self.last_epoch / self.total_steps
        if progress < self.warmup:
            return self.base_lr * progress / self.warmup
        else:
            return self.base_lr * max(( progress - 1.0)/(self.warmup - 1.0), 0.)
