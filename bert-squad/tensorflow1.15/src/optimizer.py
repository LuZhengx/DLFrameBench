import re
import tensorflow as tf

def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps,
                     optimizer_type="adam", **kwargs):
    """Creates an optimizer training op."""
    global_step = tf.compat.v1.train.get_or_create_global_step()
    
    # avoid step change in learning rate at end of warmup phase
    if optimizer_type == "adam":
        power = 1.0
        decayed_learning_rate_at_crossover_point = init_lr * (
                                (1.0 - float(num_warmup_steps) / float(num_train_steps)) ** power)
    else:
        raise NotImplementedError()

    adjusted_init_lr = init_lr * (init_lr / decayed_learning_rate_at_crossover_point)
    print('decayed_learning_rate_at_crossover_point = %e, adjusted_init_lr = %e' % (decayed_learning_rate_at_crossover_point, adjusted_init_lr))

    learning_rate = tf.constant(value=adjusted_init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.compat.v1.train.polynomial_decay(
            learning_rate,
            global_step,
            num_train_steps,
            end_learning_rate=0.0,
            power=power,
            cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    if optimizer_type == "lamb":
        raise NotImplementedError()
    else:
        print("Initializing ADAM Weight Decay Optimizer")
        # It is recommended that you use this optimizer for fine tuning, since this
        # is how the model was trained (note that the Adam m/v variables are NOT
        # loaded from init_checkpoint.)
        optimizer = AdamWeightDecayOptimizer(
                learning_rate=learning_rate,
                **kwargs)

    tvars = tf.compat.v1.trainable_variables()
    # Update (TODO: gradient clipping max_grad_norm=1.0)
    train_op = optimizer.minimize(loss, tvars)
    new_global_step = tf.identity(global_step + 1, name='step_update')
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])
    return train_op


class AdamWeightDecayOptimizer(tf.compat.v1.train.Optimizer):
    """A basic Adam optimizer that includes "correct" L2 weight decay."""

    def __init__(self,
                learning_rate,
                weight_decay_rate=0.0,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-6,
                exclude_from_weight_decay=None,
                name="AdamWeightDecayOptimizer"):
        """Constructs a AdamWeightDecayOptimizer."""
        super(AdamWeightDecayOptimizer, self).__init__(False, name)

        self.learning_rate = tf.identity(learning_rate, name='learning_rate')
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.exclude_from_weight_decay = exclude_from_weight_decay

    def apply_gradients(self, grads_and_vars, global_step=None, name=None,
            manual_fp16=False):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)
            has_shadow = manual_fp16 and param.dtype.base_dtype != tf.float32
            if has_shadow:
                # create shadow fp32 weights for fp16 variable
                param_fp32 = tf.compat.v1.get_variable(
                        name=param_name + "/shadow",
                        dtype=tf.float32,
                        trainable=False,
                        initializer=tf.cast(param.initialized_value(),tf.float32))
            else:
                param_fp32 = param

            m = tf.compat.v1.get_variable(
                    name=param_name + "/adam_m",
                    shape=param.shape.as_list(),
                    dtype=tf.float32,
                    trainable=False,
                    initializer=tf.zeros_initializer())
            v = tf.compat.v1.get_variable(
                    name=param_name + "/adam_v",
                    shape=param.shape.as_list(),
                    dtype=tf.float32,
                    trainable=False,
                    initializer=tf.zeros_initializer())

            # Standard Adam update.
            next_m = (
                    tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
            next_v = (
                    tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2, tf.square(grad)))

            update = next_m / (tf.sqrt(next_v) + self.epsilon)

            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want to decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param_fp32

            update_with_lr = self.learning_rate * update

            next_param = param_fp32 - update_with_lr

            if has_shadow:
                # cast shadow fp32 weights to fp16 and assign to trainable variable
                param.assign(tf.cast(next_param, param.dtype.base_dtype))
            assignments.extend(
                    [param_fp32.assign(next_param),
                     m.assign(next_m),
                     v.assign(next_v)])
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name