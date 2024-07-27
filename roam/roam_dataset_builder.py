import tensorflow_datasets as tfds
import tensorflow as tf
import os
local_dir = os.path.dirname(os.path.realpath(__file__))


def parse_tfrecord_fn(example):
    feature_description = {
        'image_left': tf.io.FixedLenFeature([], tf.string),
        'action': tf.io.FixedLenFeature([], tf.string),
        'folder_name': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.io.parse_tensor(example['image_left'], out_type=tf.uint8)
    action = tf.io.parse_tensor(example['action'], out_type=tf.float32)
    folder_name = tf.io.parse_tensor(example['folder_name'], out_type=tf.string)
    return image, action, folder_name

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for my_data dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(my_data): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like video, labels ...
            'video': tfds.features.Tensor(shape=(25, 64, 64, 4),dtype = tf.uint8),
            'action': tfds.features.Tensor(shape=(25, 1, 2), dtype = tf.float32),
            'folder_name': tfds.features.Tensor(shape=(), dtype = tf.string)
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('video', 'action','folder_name'),  # Set to `None` to disable
        homepage='https://meenakshisarkar.github.io/Motion-Prediction-and-Planning/dataset/',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(my_data): Downloads the data and defines the splits
    train_path = '../data/tfds_samples/train_tfrecords'
    # test_path = '../data/tfds_samples/test_tfrecords'

    # TODO(my_data): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(train_path),
        # 'test': self._generate_examples(test_path),
    }

  def _generate_examples(self, dir_path):
    """Yields examples."""
    index = 0
    for path in os.listdir(dir_path):
      tfrecord_file = os.path.join(dir_path,path)

      # Create a TFRecordDataset
      dataset = tf.data.TFRecordDataset(tfrecord_file)

      # Map the parsing function over the dataset
      parsed_dataset = dataset.map(parse_tfrecord_fn)
      for video, action, folder_name in parsed_dataset:
        yield index, {
          'video' : video.numpy(),
          'action' : action.numpy(),
          'folder_name': folder_name.numpy()
        }
        index += 1