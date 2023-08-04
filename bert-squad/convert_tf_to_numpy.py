import os
import argparse


try:
    import numpy as np
    import tensorflow as tf
except ImportError:
    print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
        "https://www.tensorflow.org/install/ for installation instructions.")
    raise


def load_tf_weights_in_bert(tf_checkpoint_path):
    """ Load tf checkpoints and save them as numpy files
    """
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    arrays = {}
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        arrays[name] = tf.train.load_variable(tf_path, name)

    # save to a .npz file
    numpy_path = os.path.join(os.path.dirname(tf_path), 'ckpt.npz')
    print("Saving weights to {}".format(numpy_path))
    np.savez(numpy_path, **arrays)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tf_path", default=None, type=str, required=True,
                        help="Path to TensorFlow checkpoint file. For example: ./uncased_L-12_H-768_A-12/bert_model.ckpt")
    args = parser.parse_args()
    load_tf_weights_in_bert(args.tf_path)