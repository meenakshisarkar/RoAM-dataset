import tensorflow as tf
import numpy as np
import os
import glob

def parse_example(example_proto):
    feature_description = {
        'image_left': tf.io.FixedLenFeature([], tf.string),
        'action': tf.io.FixedLenFeature([], tf.string),
        'folder_name': tf.io.FixedLenFeature([], tf.string)
    }

    example = tf.io.parse_single_example(example_proto, feature_description)

    image_left = tf.io.parse_tensor(example['image_left'], out_type=tf.uint8)
    # image_left = tf.cast(image_left, tf.float32)  # Cast to float32 if needed
    action = tf.io.parse_tensor(example['action'], out_type=tf.float32)
    folder_name = tf.io.parse_tensor(example['folder_name'], out_type=tf.string)

    return image_left, action, folder_name

def read_and_save_sequences(record_file, output_dir):
    dataset = tf.data.TFRecordDataset(record_file)
    dataset = dataset.map(parse_example)
    count = 0

    for image_left, action, folder_name in dataset:
        count += 1
        folder_name = folder_name.numpy().decode('utf-8')
        sequence_dir = os.path.join(output_dir, str(count))
        print(sequence_dir)
        os.makedirs(sequence_dir, exist_ok=True)

        for i, frame in enumerate(image_left.numpy()):
            frame_path = os.path.join(sequence_dir, f"frame_{i}.png")
            tf.keras.utils.save_img(frame_path, frame)

        action_path = os.path.join(sequence_dir, "action.npy")
        np.save(action_path, action.numpy())

# Example usage
data_folder = "/home/gcdsl_tbl/meenakshi/catkin_ws/RoAM_dataset/processed/tfrecord_tp/"
record_file = glob.glob(os.path.join(data_folder, "*.tfrecord-000"))
output_dir = "/home/gcdsl_tbl/meenakshi/catkin_ws/RoAM_dataset/processed/processed_tf/bilinear"
read_and_save_sequences(record_file, output_dir)