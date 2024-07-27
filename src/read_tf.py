
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
import tensorflow as tf
import glob
from argparse import ArgumentParser
from os import listdir, makedirs, system
from os.path import exists
import imageio
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
V_MAX=0.1
Turn_MAX=1.8
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
  if not exists(output_dir):
      makedirs(output_dir)
  
  partition=['train','test']
  for part in partition:
    tfrecord_path=f'{record_file}/{part}_tfrecords'
    part_dir=f'{output_dir}/{part}'
    os.makedirs(part_dir, exist_ok=True)
    for T1 in os.listdir(tfrecord_path):
     if T1.split('.')[0] != 'json':
      dataset = tf.data.TFRecordDataset(os.path.join(tfrecord_path,T1))
      dataset = dataset.map(parse_example)
      for image_left, action, folder_name in dataset:
        seq_dir = str(folder_name)
        seq_dir=seq_dir.split('/')[7]+'/'+seq_dir.split('/')[8]+'/'+seq_dir.split('/')[9]
        sequence_dir=f'{part_dir}/{seq_dir}'
        # sequence_dir = os.path.join(output_dir, str(count))
        print(f'Saving images and action to directory: {sequence_dir}')
        os.makedirs(sequence_dir, exist_ok=True)
        img_dir=sequence_dir+'/images'
        action_dir=sequence_dir+'/action'
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(action_dir, exist_ok=True)
        for i, frame in enumerate(image_left.numpy()):
            frame_path = os.path.join(img_dir, f"frame_{i:07d}.png")
            imageio.imwrite(frame_path, frame[...,:3])
        ac=action.numpy()
        action_path = os.path.join(action_dir, "action.npz")
        np.savez(action_path, action0=ac[...,0]*V_MAX,action1=ac[...,1]*Turn_MAX)




def main(data_folder, output_dir):
    read_and_save_sequences(data_folder, output_dir)
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_folder", type=str, dest="data_folder",
                        default="../data", help="Path to folder containing tfrecord files")
    parser.add_argument("--output_dir", type=str, dest="output_dir",
                        default="../data/processed", help="Output Dir path for processed image files")
    args = parser.parse_args()
    main(**vars(args))

