from mxnet.lr_scheduler import LRScheduler

class LinearWarmUpScheduler(LRScheduler):
    """
    Applies a warm up period to the learning rate.
    """

    def __init__(self, lr, warmup, total_steps):
        self.warmup = warmup
        self.total_steps = total_steps
        super(LinearWarmUpScheduler, self).__init__(lr)

    def __call__(self, num_update):
        progress = num_update / self.total_steps
        if progress < self.warmup:
            return self.base_lr * progress / self.warmup
        else:
            return self.base_lr * max(( progress - 1.0)/(self.warmup - 1.0), 0.)
