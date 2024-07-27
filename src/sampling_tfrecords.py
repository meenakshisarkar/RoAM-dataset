#Code developed by Meenakshi Sarkar
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
import tensorflow as tf
import numpy as np
import sys
import tensorflow_io as tfio
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Feature
from PIL import Image
from os.path import exists
from os import makedirs
from tqdm import trange
from argparse import ArgumentParser

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.getLogger('tensorflow').setLevel(logging.FATAL)
# VIDEO_NO=10
# SEQ_NO=2
# LEN=25 #for training
# LEN=40 #for testing
SEED=77
np.random.seed(SEED)
tf.random.set_seed(SEED)
# def _float_feature(value):
#   """Returns a float_list from a float / double."""
#   return tf.train.Feature(float_list=tf.train.FloatList(value=value))
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

def serialize_features(vid_frames,actions, folder_name):
  image_feature = Feature(
    bytes_list=BytesList(value=[
        tf.io.serialize_tensor(vid_frames).numpy()
    ]))
  action_feature = Feature(
      bytes_list=BytesList(value=[
          tf.io.serialize_tensor(actions).numpy(),
      ]))
  folder_name = Feature(
    bytes_list=BytesList(value=[
        tf.io.serialize_tensor(folder_name).numpy()
    ]))
  feature = {
      'image_left': image_feature,
      'action': action_feature,
      'folder_name': folder_name,
  }
  _proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return _proto.SerializeToString()

def tf_serialize(f0,f1):
  tf_string = tf.py_function(
    serialize_features,
    (f0, f1),  # Pass these args to the above function.
    tf.string)      # The return type is `tf.string`.
  return tf.reshape(tf_string, ())

def read_and_save_sequences(record_file, output_dir,part,LEN,SEQ_NO,VIDEO_NO):
  if not exists(output_dir):
      makedirs(output_dir)
  
  tfrecord_path=f'{record_file}/{part}_tfrecords'
  tf_dir=f'{output_dir}/{part}_tfrecords'
  os.makedirs(tf_dir, exist_ok=True)
  tfrecord_list=os.listdir(tfrecord_path)
  totla_files=len(tfrecord_list)
  for counter in range(SEQ_NO):
    TFfile_path=tf_dir+'/roam_train_sample.tfrecord-{:03d}'.format(counter)
    with tf.io.TFRecordWriter(TFfile_path) as writer:
     for i in trange(VIDEO_NO):
      c = np.random.randint(totla_files)
      file=tfrecord_list[c]
      # print(f'file path {file}')
      dataset = tf.data.TFRecordDataset(os.path.join(tfrecord_path,file))
      dataset = dataset.map(parse_example)
      for image_list, action_list, folder_name in dataset:
        t_0 = np.random.randint(len(image_list.numpy()) - LEN + 1)
        # print(f'image_list length: {len(image_list.numpy())}')
        vid_frames=[]
        actions=[]
        for t in range(LEN):
          t=t+t_0
          image=image_list[t,...]
          action_tensor=action_list[t,...]
          crop_img=tf.image.convert_image_dtype(image,dtype=tf.float32)
          crop_img=tf.image.resize(crop_img,[64,64],method='bilinear',preserve_aspect_ratio=True)
          # crop_img=tf.image.resize(image,[64,64],method='lanczos3',preserve_aspect_ratio=True)
          crop_img=tf.image.convert_image_dtype(crop_img,dtype=tf.uint8)
          vid_frames.append(crop_img)
          action_tensor= tf.expand_dims(action_tensor, axis=0)
          actions.append(action_tensor)
        folder_name=str(folder_name)
        vid_frames=tf.stack(vid_frames, axis=0)	
        # print(f'shape of video frame {vid_frames.shape}')
        actions=tf.stack(actions, axis=0)
        data_srl=serialize_features(vid_frames,actions,folder_name)
        writer.write(data_srl)
    writer.close()


def main(data_folder, output_dir,part,LEN,SEQ_NO,VIDEO_NO):
    read_and_save_sequences(data_folder, output_dir,part,LEN,SEQ_NO,VIDEO_NO)
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_folder", type=str, dest="data_folder",
                        default="../data", help="Path to folder containing tfrecord files")
    parser.add_argument("--output_dir", type=str, dest="output_dir",
                        default="../data/tfds_samples", help="Output Dir path for sampled tfrecord files")
    parser.add_argument("--part", type=str, dest="part",
                        default="train", help="Whether we are creating samples from training or testing data")
    parser.add_argument("--len", type=int, dest="LEN",
                        default=25, help="Total lenght of image sequence sample for training and testing. During training we kept it as 25 and training it is 40")
    parser.add_argument("--seq_no", type=int, dest="SEQ_NO",
                        default=2, help="Total no of sampling sequence")
    parser.add_argument("--vid_no", type=int, dest="VIDEO_NO",
                        default=10, help="Total no of video snippets of length LEN in each sequence of tfrecord file")
    args = parser.parse_args()
    main(**vars(args))
'''
# datapath="../../../catkin_ws/RoAM_dataset/processed/train"
datapath="../../../catkin_ws/RoAM_dataset/processed/test"
# file_path= "/date_2023_03_10/corridor_lab2"
# path=datapath+file_path
# print(path)
# image_location_path=os.path.join(path, "Zed_0002/left")
# action_path=os.path.join(path,"Rosbag_0002")
# print(action_path)

tf_dir=datapath+'/../tfrecord_test'
if not exists(tf_dir):
        makedirs(tf_dir)

counter=0
zedfile_list=[]
for d1 in os.listdir(datapath):
  for d2 in os.listdir(os.path.join(datapath, d1)):
    date_path=os.path.join(datapath, d1)
    for d3 in os.listdir(os.path.join(date_path, d2)):
      location_path=os.path.join(date_path, d2)
      if d3.split('_')[0]== "Zed":
        zedfile_path=os.path.join(location_path,d3)
        print(zedfile_path)
        zedfile_list.append(zedfile_path)
# print(f'length of zedfile_list: {len(zedfile_list)}')
# print(zedfile_list[0])

for file in zedfile_list:
   print(file)
   file_path=file+"/../"
   rosbag_path=os.path.join(file_path,"Rosbag_"+file.split('_')[-1]+"/")
  #  print(os.listdir(rosbag_path))
  #  print(rosbag_path)
totla_files=len(zedfile_list)
for counter in range(SEQ_NO):
  TFfile_path=tf_dir+'/roam_train_sample.tfrecord-{:03d}'.format(counter)
  with tf.io.TFRecordWriter(TFfile_path) as writer:
   for i in trange(TOTAL_SEQ):
    c = np.random.randint(totla_files)
    file=zedfile_list[c]
    print(f'file path {file}')
    images_fnames = sorted(os.listdir(file+"/left/"))
    t_0 = np.random.randint(len(images_fnames) - LEN + 1)
    file_path=file+"/../"
    rosbag_path=os.path.join(file_path,"Rosbag_"+file.split('_')[-1]+"/")
    with open(rosbag_path+"control_actions.txt") as f: 
      action_list = f.readlines()
    vid_frames=[]
    actions=[]
    folder_names=[]
    fpath = file+"/left"
    for t in range(LEN):
      t=t+t_0+1
      fname=f'{fpath}/left{t:08d}.png'
      if (t-1)//3 <len(action_list):
        action_str= action_list[(t-1)//3].strip('\n')
        ac=action_str.strip('][').split(', ')
      # image=tf.keras.utils.load_img(fname,color_mode='rgb',target_size=[64,64],interpolation='bilinear')

      #image=tf.keras.utils.load_img(fname,color_mode='rgb',keep_aspect_ratio=True)
      image=tf.image.convert_image_dtype(image,dtype=tf.float32)
      # print(f'shape of image before crop {image.shape}')
      crop_img=image[:,280:1000]  #cropping the 720 x 1280 image to 720 x 720 image
      # print(f'shape of image {image.shape}')
      image = Image.open(fname)
      # image=image[:,280:1000]
      crop_img=image.crop((280,0,1000,720))
      crop_img=tf.image.convert_image_dtype(crop_img,dtype=tf.float32)
      crop_img=tf.image.resize(crop_img,[64,64],method='bilinear',preserve_aspect_ratio=True)
      # crop_img=tf.image.resize(image,[64,64],method='lanczos3',preserve_aspect_ratio=True)
      crop_img=tf.image.convert_image_dtype(crop_img,dtype=tf.uint8)
      # print(f'some values from cropped images {crop_img[20:25,40:45,2]}')
      vid_frames.append(crop_img)
      action_tensor=[float(ac[0])/V_MAX, float(ac[1])/Turn_MAX] ##normalizing control actions.
                # im= tf.expand_dims(image, axis=0)
      action_tensor= tf.expand_dims(action_tensor, axis=0)
      actions.append(action_tensor)
    folder_names=fpath+"/left/left_{t_0:08d}"
    vid_frames=tf.stack(vid_frames, axis=0)	
    print(f'shape of video frame {vid_frames.shape}')
    actions=tf.stack(actions, axis=0)
    data_srl=serialize_features(vid_frames,actions,folder_names)
    writer.write(data_srl)
  writer.close()'''

        
 