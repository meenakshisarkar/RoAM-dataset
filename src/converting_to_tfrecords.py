#Code developed by Meenakshi Sarkar
import os
import tensorflow as tf
import numpy as np
import sys
import tensorflow_io as tfio
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Example, Features, Feature
from PIL import Image
from os.path import exists
from os import makedirs
from tqdm import trange

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
TOTAL_SEQ=1024
SEQ_NO=1
# LEN=25 #for training
LEN=40 #for testing
SEED=77
V_MAX=0.1
Turn_MAX=1.8
np.random.seed(SEED)
tf.random.set_seed(SEED)
# def _float_feature(value):
#   """Returns a float_list from a float / double."""
#   return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def serialize_features(vid_frames,actions, folder_name):
  """
  Creates a tf.train.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.train.Example-compatible
  # data type.
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
  TFfile_path=tf_dir+'/kth-train.tfrecord-{:03d}'.format(counter)
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
      '''image=tf.keras.utils.load_img(fname,color_mode='rgb',keep_aspect_ratio=True)
      image=tf.image.convert_image_dtype(image,dtype=tf.float32)
      # print(f'shape of image before crop {image.shape}')
      crop_img=image[:,280:1000]  #cropping the 720 x 1280 image to 720 x 720 image'''
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
  writer.close()

        
        
          # dir_len=int(len(os.listdir(os.path.join(location_path,d3+"/left/"))))
          # TFfile_path=tf_dir+'/roam-train.tfrecord-{:04d}-of-0018'.format(counter)
          # with tf.io.TFRecordWriter(TFfile_path) as writer:
          #   for l in range(dir_len//40):
          #     low=l*40
          #     high=l*40+40
          #     image_location_path=os.path.join(location_path,d3+"/left/")
          #     vid_frames=[]
          #     actions=[]
          #     for t in range(low+1, high+1):
          #       fname =  "{}/left{:08d}.png".format(image_location_path, t)
          #       # print((stidx+t-1)//3)
                
          #       image=tf.keras.utils.load_img(fname,color_mode='rgba',target_size=[64,64],interpolation='bilinear')
          #       image=tfio.experimental.color.rgba_to_rgb(image)
                
          #       # im=im[...,:]
          #       # im = im.reshape(1, resize_shape[0], resize_shape[1], 3)
          #       vid_frames.append(tf.image.convert_image_dtype(image,dtype=tf.float32))
          #     vid_frames=tf.stack(vid_frames, axis=0)	
          #     actions=tf.stack(actions, axis=0)
              
          #   writer.close()
          # counter=counter+1



# counter=0
# for d1 in os.listdir(datapath):
#   for d2 in os.listdir(os.path.join(datapath, d1)):
#     date_path=os.path.join(datapath, d1)
#     for d3 in os.listdir(os.path.join(date_path, d2)):
#       location_path=os.path.join(date_path, d2)
#       if d3.split('_')[0]== "Zed":
#         rosbag_path=os.path.join(location_path,"Rosbag_"+d3.split('_')[1]+"/")
#         with open(rosbag_path+"control_actions.txt") as f: 
#           action_list = f.readlines()
#           dir_len=int(len(os.listdir(os.path.join(location_path,d3+"/left/"))))
#           TFfile_path=tf_dir+'/roam-train.tfrecord-{:04d}-of-0018'.format(counter)
#           with tf.io.TFRecordWriter(TFfile_path) as writer:
#             for l in range(dir_len//40):
#               low=l*40
#               high=l*40+40
#               image_location_path=os.path.join(location_path,d3+"/left/")
#               vid_frames=[]
#               actions=[]
#               for t in range(low+1, high+1):
#                 fname =  "{}/left{:08d}.png".format(image_location_path, t)
#                 # print((stidx+t-1)//3)
#                 if (t-1)//3 <len(action_list):
#                   action_str= action_list[(t-1)//3].strip('\n')
#                   ac=action_str.strip('][').split(', ')
#                 image=tf.keras.utils.load_img(fname,color_mode='rgba',target_size=[64,64],interpolation='bilinear')
#                 image=tfio.experimental.color.rgba_to_rgb(image)
#                 action_tensor=[float(ac[0])/V_MAX, float(ac[1])/Turn_MAX] ##normalizing control actions.
#                 # im= tf.expand_dims(image, axis=0)
#                 action_tensor= tf.expand_dims(action_tensor, axis=0)
#                 actions.append(action_tensor)
#                 # im=im[...,:]
#                 # im = im.reshape(1, resize_shape[0], resize_shape[1], 3)
#                 vid_frames.append(tf.image.convert_image_dtype(image,dtype=tf.float32))
#               vid_frames=tf.stack(vid_frames, axis=0)	
#               actions=tf.stack(actions, axis=0)
#               data_srl=serialize_features(vid_frames,actions)
#               writer.write(data_srl)
#             writer.close()
#           counter=counter+1
      
# vid_frames=tf.expand_dims(vid_frames, axis=0)
            
# vid_data=[]
# action_data=[]
# # vid_frames=tf.expand_dims(vid_frames, axis=0)
# # actions=tf.expand_dims(actions, axis=0)
# vid_data.append(vid_frames)
# # vid_data=tf.stack(vid_data, axis=0)
# action_data.append(actions)
# action_data=tf.stack(action_data, axis=0)
# ds = tf.data.Dataset.from_tensor_slices((vid_frames,actions))
# image_feature = Feature(
#     bytes_list=BytesList(value=[
#         tf.io.serialize_tensor(vid_frames).numpy(),
#     ]))
# features = Features(feature={
#     'image': image_feature,
# })
# example = Example(features=features)
# # print(example)
# example_bytes = example.SerializeToString()
# print(example_bytes)
# action_feature = Feature(
#     bytes_list=BytesList(value=[
#         tf.io.serialize_tensor(actions),
#     ]))
# ds_bytes = ds.map(lambda image, action: tf.py_function(func=serialize_features, inp=[image, action], Tout=tf.string))

# ds_bytes = ds.map(tf.io.serialize_tensor)
# def generator():
#   for features in ds:
#     yield serialize_features(*features)
# # serialized_features_dataset = features_dataset.map(tf_serialize)
# ds_bytes = tf.data.Dataset.from_generator(
#     generator, output_types=tf.string, output_shapes=())
# print(serialized_features_dataset)
# for x in serialized_features_dataset:
#     print(x)
# filename = 'test3.tfrecord'
# writer = tf.data.experimental.TFRecordWriter(filename)
# writer.write(ds_bytes)


# file_path = 'data_00.tfrecords'
# with tf.io.TFRecordWriter(file_path) as writer:
#     for i in [0,1]:
#      ex_srl=serialize_features(vid_frames,actions)
#      writer.write(ex_srl)



'''
def _parse_tfr_element(element):
  parse_dic = {
    'image_left': tf.io.FixedLenFeature([], tf.string), # Note that it is tf.string, not tf.float32
    'action': tf.io.FixedLenFeature([], tf.string),
    }
  example_message = tf.io.parse_single_example(element, parse_dic)

  image= example_message['image_left'] # get byte string
  actions= example_message['action'] # get byte string
  image_fr = tf.io.parse_tensor(image, out_type=tf.float32) # restore 2D array from byte string
  act = tf.io.parse_tensor(actions, out_type=tf.float32) # restore 2D array from byte string
  return image_fr, act

tfr_dataset = tf.data.TFRecordDataset('data_00.tfrecords') 
# for serialized_instance in tfr_dataset:
#   print(serialized_instance)
dataset = tfr_dataset.map(_parse_tfr_element)
# print(dataset[0])
for instance, __ in dataset:
  print()
  print(instance.shape) 
  '''
