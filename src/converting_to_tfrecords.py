#Code developed by Meenakshi Sarkar
import os
import tensorflow as tf
import sys
import tensorflow_io as tfio
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Example, Features, Feature
from os.path import exists
from os import makedirs


# def _float_feature(value):
#   """Returns a float_list from a float / double."""
#   return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def serialize_features(vid_frames,actions):
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
  feature = {
      'image_left': image_feature,
      'action': action_feature,
  }
  _proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return _proto.SerializeToString()

def tf_serialize(f0,f1):
  tf_string = tf.py_function(
    serialize_features,
    (f0, f1),  # Pass these args to the above function.
    tf.string)      # The return type is `tf.string`.
  return tf.reshape(tf_string, ())



datapath="../../../catkin_ws/RoAM_dataset/processed/train"
# file_path= "/date_2023_03_10/corridor_lab2"
# path=datapath+file_path
# print(path)
# image_location_path=os.path.join(path, "Zed_0002/left")
# action_path=os.path.join(path,"Rosbag_0002")
# print(action_path)
low=1
high=41
V_MAX=0.1
Turn_MAX=1.8
tf_dir=datapath+'/tfrecord'
if not exists(tf_dir):
        makedirs(tf_dir)
counter=0
for d1 in os.listdir(datapath):
	for d2 in os.listdir(os.path.join(datapath, d1)):
		date_path=os.path.join(datapath, d1)
		for d3 in os.listdir(os.path.join(date_path, d2)):
			location_path=os.path.join(date_path, d2)
			if d3.split('_')[0]== "Zed":
				rosbag_path=os.path.join(location_path,"Rosbag_"+d3.split('_')[1]+"/")
				with open(rosbag_path+"control_actions.txt") as f: 
					action_list = f.readlines()
					dir_len=int(len(os.listdir(os.path.join(location_path,d3+"/left/"))))
					TFfile_path=tf_dir+'/roam-train.tfrecord-{:04d}-of-0018'.format(counter)
					with tf.io.TFRecordWriter(TFfile_path) as writer:
						for l in range(dir_len//40):
							low=l*40
							high=l*40+40
							image_location_path=os.path.join(location_path,d3+"/left/")
							vid_frames=[]
							actions=[]
							for t in range(low+1, high+1):
								fname =  "{}/left{:08d}.png".format(image_location_path, t)
								# print((stidx+t-1)//3)
								if (t-1)//3 <len(action_list):
									action_str= action_list[(t-1)//3].strip('\n')
									ac=action_str.strip('][').split(', ')
								image=tf.keras.utils.load_img(fname,color_mode='rgba',target_size=[64,64],interpolation='bilinear')
								image=tfio.experimental.color.rgba_to_rgb(image)
								action_tensor=[float(ac[0])/V_MAX, float(ac[1])/Turn_MAX] ##normalizing control actions.
								# im= tf.expand_dims(image, axis=0)
								action_tensor= tf.expand_dims(action_tensor, axis=0)
								actions.append(action_tensor)
								# im=im[...,:]
								# im = im.reshape(1, resize_shape[0], resize_shape[1], 3)
								vid_frames.append(tf.image.convert_image_dtype(image,dtype=tf.float32))
							vid_frames=tf.stack(vid_frames, axis=0)	
							actions=tf.stack(actions, axis=0)
							data_srl=serialize_features(vid_frames,actions)
							writer.write(data_srl)
						writer.close()
					counter=counter+1
			
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
