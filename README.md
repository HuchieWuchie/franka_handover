# Learning to Segment Object Affordances on Synthetic Data for Task-oriented Robotic Handovers

By Albert Daugbjerg Christensen

Aalborg University (2022)

Electronics and IT

## Description

This repository contains a ROS node implementation of the paper "Learning to Segment Object Affordances on Synthetic Data for Task-oriented Robotic Handovers" on a Franka Emika Panda robot. The system is capable of performing task-oriented handovers, where an object is grasped by its functional affordance and handover with an appropriate orientaiton. Object affordances are detected using our deep neural network AffNet-DR, which was trained solely on synthetic data. A synthetic dataset already generated can be downloaded from here https://drive.google.com/file/d/1yRL2JbZhZEsL2O9fKbM_DDJ7U53dogkl/view?usp=sharing .

This readme file refers to 3 different pc's.

Franka pc: The pc for which this code is intended to run. This pc needs to be connected with an ethernet cable to the Franka controller.
Interface pc: A pc of any OS that is connected to the Franka robot ethernet port. This pc can acccess the control panel at robot.franka.de
ROS pc: The pc for where all the other computations takes place such as visual reasoning, handover tasks, etc.

## Requirements:

General system requirements
```
CUDA version 11.6
NVIDIA GPU driver 510.60.02
ROS melodic
ros-melodic-moveit
ros-melodic-panda-moveit-config
ros-melodic-realsense2-description
libfranka
ros-franka
```

C++:
```
realsense2
PCL (point cloud library)
OpenCV
```

Python 3.6.9
```
open3d 0.15.2
cv2 4.2.0
numpy 1.19.5
scipy 1.5.4
scikit_learn 0.24.2
torch (Pytorch) 1.10.2 cuda version
torchvision 0.11.2 cuda
scikit_image 0.17.2
PIL 8.4.0
rospkg 1.4.0
```

The system ran on a Lenovo Thinkpad P53 laptop with a Quadro RTX 4000 GPU with 8 GB VRAM and an Intel Core i9-9880H CPU 2.3 GHZ and 32 GB RAM.


## Installation:


### Setup the Franka pc

Start  by installing libfranka, ros-franka and the aau_franka_moveit package on the Franka pc, as described here: https://github.com/HuchieWuchie/AAU_franka_moveit

### Setup the ROS pc, the one where you will run this repository

Setup the ros workspace

```
mkdir ros_ws
mkdir ros_ws/src
cd ros_ws/src
```

Git clone the required repositories

git clone https://github.com/justagist/franka_panda_description.git
git clone https://github.com/HuchieWuchie/AAU_franka_moveit.git
git clone https://github.com/HuchieWuchie/franka_handover.git
```

Make and build the repository

```
cd ..
catkin_make
source devel/setup.bash
```

Download pretrained weights from: https://drive.google.com/file/d/1psCn_aT5KUyQDJrdxqR7GJgHeCewGokS/view?usp=sharing

Place and rename the weights file to ros_ws/src/affordanceAnalyzer/scripts/affordancenet/weights.pth

## Setup the network:

Follow the network configuration guide from: https://github.com/HuchieWuchie/AAU_franka_moveit

## Usage

For all of the usage case scenarios, remember to activate FCI on the Interface pc, by going to robot.franka.de and unlocking the motors, and then activating FCI.

Connect the Lidar scanner and the intel realsense d435 mounted on the Franka to the ROS pc via USB.

launch roscore and launch file
```
source devel/setup.bash
sudo chmod a+rw /dev/ttyACM0
roslaunch fh_handover fh_bringup.launch
```

Launch whatever experiement you want, chose between the ones listed below:
```
rosrun fh_handover final_test_observation.py
```

In order to command the robot to pick up an object you must send a command to the rostopic /objects_affordances_id. The integer id corresponds to the object classes of AffNet-DR, eg. 1 (knife), 16 (mallet), etc.
