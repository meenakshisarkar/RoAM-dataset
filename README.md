# About
<p align="center">
  <img src="https://github.com/meenakshisarkar/Motion-Prediction-and-Planning/blob/updated/images/fig_roman_burger_mod.png" width="45%" />
  <img src="https://github.com/meenakshisarkar/Motion-Prediction-and-Planning/blob/updated/images/fig_roman_intro_mod.jpg" width="40%" />
</p>
We introduce **Robot Autonomous Motion (RoAM)**, a unique video-action dataset that includes 50 long video sequences collected over 7 days at 14 different indoor spaces, capturing various indoor human activities from the ego-motion perspective of the Turtlebot3 robot. Along with the stereo image sequences, RoAM also contains time-stamped robot action sequences that are synchronised with the video data. The dataset primarily includes a range of human movements, such as walking, jogging, hand-waving, gestures, and sitting actions, which an indoor robot might encounter while navigating environments populated by people. Each of the 50 recorded video sequences, is started with unique initial conditions such that there is sufficient diversity and variations in the dataset. The dataset pre-dominantly records human walking motion while the robot slowly explores its surroundings. 
<!-- # RoAM Data
The RoAM dataset is collected using a custom-built Turtlebot3 Burger robot. We have used the Tensorflow Dataset API to generate **3,07,200** video-action sequences of length 25 for training our variational and diffusion models. It also contains the corresponding action values from the robot's motion to capture the movement of the camera.  We have used Zed mini stereo vision camera for capturing the left and right timestamped image pairs. Other than that the robot is equipped with an LDS-01 2-dimensional LiDAR, a TP-link WiFi communication module as shown in Figure above. The Turtlebot3 employs two DYNAMIXEL XL430-W250 servo motors for navigation, utilizing current-based torque control. These motors are actuated and controlled by the OpenCR-01 board, which is integrated into the platform. For our specific application, we have selected the Jetson TX2 board as the onboard computer, operating on the ROS Melodic framework and the Ubuntu 18.04 operating system. This setup offers the advantage of leveraging the Jetson TX2's high computational power to support complex robotic tasks, such as perception, navigation, and machine learning. 
<p align="center">
<img src="https://github.com/meenakshisarkar/Motion-Prediction-and-Planning/blob/updated/images/fig_roman_processed_data.png" width="45%" />
</p>
**The processed data file structure of RoAM** -->
# Samples
<p align="center">
  <img src="https://github.com/meenakshisarkar/Motion-Prediction-and-Planning/blob/updated/images/gt_1.gif" width="24%" />
  <img src="https://github.com/meenakshisarkar/Motion-Prediction-and-Planning/blob/updated/images/gt_2.gif" width="24%" />
  <img src="https://github.com/meenakshisarkar/Motion-Prediction-and-Planning/blob/updated/images/gt_3.gif" width="24%" />
  <img src="https://github.com/meenakshisarkar/Motion-Prediction-and-Planning/blob/updated/images/gt_4.gif" width="24%" />
</p>
**Video Samples**

We are hosting the Tfrecord files for training and testing on Huggingface-hub.

```bash
# Installing dependencies from requirements.txt
pip install -r requirements.txt


# To download the train and test tfrecord files
cd data
python download_roam.py

# To create 300K training video samples from the downloaded tfrecord files please run
python sampling_tfrecords.py --data_folder /path/to/tfrecord files --output_dir /path/to/desination/dir --part train --len length_of_vid_seq

# To create the custome TFDS dataset object
cd roam
tfds build roam

```bibtex
@InProceedings{acpnet2023,
  author={Sarkar, Meenakshi and Honkote, Vinayak and Das, Dibyendu and Ghose, Debasish},
  booktitle={2023 32nd IEEE International Conference on Robot and Human Interactive Communication (RO-MAN)}, 
  title={Action-conditioned Deep Visual Prediction with RoAM, a new Indoor Human Motion Dataset for Autonomous Robots}, 
  year={2023},
  volume={},
  number={},
  pages={1115-1120},
  doi={10.1109/RO-MAN57019.2023.10309423}
}

