import os
import tensorflow as tf
from src.config import config

_NUM_DATA_FILES = 5
_DEFAULT_IMAGE_BYTES = \
    config['dataset-img-size'][0] * \
    config['dataset-img-size'][1] * \
    config['dataset-img-size'][2]
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1


def Cifar10Dataset(is_training, data_dir, batch_size, data_format, repeats=None,
                    num_parallel_batches=config['dataset-num-workers']):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    datasets_num_private_threads: Number of private threads for tf.data.
    num_parallel_batches: Number of parallel batches for tf.data.

  Returns:
    A dataset that can be used for iteration.
  """

  # Constructs dataset from files
  assert tf.io.gfile.exists(data_dir), ('CIFAR-10 dataset not exists.')
  if is_training:
    filenames = [
        os.path.join(data_dir, 'data_batch_%d.bin' % i)
        for i in range(1, _NUM_DATA_FILES + 1)
    ]
  else:
    filenames = [os.path.join(data_dir, 'test_batch.bin')]
  dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)

  # Shuffles records & repeating for training
  if is_training:
    dataset = dataset.shuffle(buffer_size=config['dataset-train-size'])

  # Parses the raw records into images and labels.
  dataset = dataset.map(
      lambda value: parse_and_preprocess(value, is_training, data_format),
      num_parallel_calls=num_parallel_batches)
  dataset = dataset.batch(batch_size, drop_remainder=False)

  # Shuffles records & repeating for training
  if repeats:
    dataset = dataset.repeat(repeats)

  # Prefetch
  dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

  return dataset


def parse_and_preprocess(raw_record, is_training, data_format):
  """Parse CIFAR-10 image and label from a raw record."""
  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.io.decode_raw(raw_record, tf.uint8)

  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  label = tf.cast(record_vector[0], tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],
                           config['dataset-img-size'])

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

  if is_training:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_with_crop_or_pad(
        image, config['dataset-img-size'][1] + config['dataset-img-pad'] * 2,
        config['dataset-img-size'][2] + config['dataset-img-pad'] * 2)

    # Randomly crop a [HEIGHT, WIDTH] section of the image.
    image = tf.image.random_crop(
        image, [config['dataset-img-size'][x] for x in (1, 2, 0)])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  # image = tf.image.per_image_standardization(image)
  image = image/255.0
  image = (image - tf.constant(config['dataset-img-means']))/tf.constant(config['dataset-img-stds'])

  if data_format == 'channels_first':
    image = tf.transpose(image, [2, 0, 1])

  return image, label
