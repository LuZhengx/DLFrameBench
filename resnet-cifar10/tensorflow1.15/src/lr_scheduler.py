import tensorflow as tf

from src.config import config

def learning_rate_with_decay(
    batch_size, boundary_epochs=config['lr-decay-boundaries'],
    decay_rates=config['lr-decay-rate'], base_lr=config['lr-base'],
    warmup_rate=config['lr-warmup-rate'], warmup_epochs=config['lr-warmup-epochs']):
  """Get a learning rate that decays step-wise as training progresses.

  Args:
    batch_size: the number of examples processed in each training batch.
    boundary_epochs: list of ints representing the epochs at which we
      decay the learning rate.
    decay_rates: list of floats representing the decay rates to be used
      for scaling the learning rate. It should have one more element
      than `boundary_epochs`, and all elements should have the same type.
    base_lr: Initial learning rate scaled based on batch_denom.
    warmup_rate: lr rate in warmup phase.
    warmup_epochs: the number of epochs to warmup.
  Returns:
    Returns a function that takes a single argument - the number of batches
    trained so far (global_step)- and returns the learning rate to be used
    for training the next batch.
  """
  initial_learning_rate = base_lr * batch_size / config['lr-batch-denom']
  step_size = (config['dataset-train-size'] + batch_size - 1) // batch_size

  # Reduce the learning rate at certain epochs.
  boundaries = [step_size * epoch - 1 for epoch in boundary_epochs]
  vals = [initial_learning_rate]
  for i in range(len(boundaries)):
    vals.append(vals[i] * decay_rates)
  
  # Warm up
  boundaries = [step_size * warmup_epochs -1] + boundaries
  vals = [initial_learning_rate * warmup_rate] + vals

  def learning_rate_fn(global_step):
    """Builds scaled learning rate function with 1 epoch warm up."""
    return tf.train.piecewise_constant(global_step, boundaries, vals)

  return learning_rate_fn