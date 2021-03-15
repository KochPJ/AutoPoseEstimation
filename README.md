## AutoPoseEstimation

# Introduction
Recently developed deep neural networks achieved state-of-the-art results in the subject of 6D object pose estimation for robot manipulation. However, those supervised deep learning methods require expensive annotated training data. Current methods for reducing those costs frequently use synthetic data from simulations, but rely on expert knowledge and suffer from the domain gap when shifting to the real world. Here, we present a proof of concept for a novel approach of autonomously generating annotated training data for 6D object pose estimation. This approach is designed for learning new objects in operational environments while requiring little interaction and no expertise on the part of the user. We evaluate our autonomous data generation approach in two grasping experiments, where we archive a similar grasping success rate as related work on a non autonomously generated data set. 

# Data
All data used in the components of this project can be downloaded. A download link and instructions can be found in the readme of the each component.


# Hardware
In order to conduct your own grasping experiments or aquire new data you need a RGB-D Camera, an industrial robot armm, and a gripper. We use an Realsense-435 depth camera, the UR-5 robot arm, and the Robotiq 2F-85 gripper. 

1. Robot and Gripper: We are not providing any drivers for the robot and gripper. If you want to use your own setup you will need to write your own drivers and comunication. We provide a robot/gripper controller in robotcontroller/TestController.py. It uses a robot/gripper client to interact with the hardware. You can take this as a starting point to connect your hardware. Please make sure that all functions in the RobotController are callable. 
 
2. Camera: If you want to use your own RGB-D Camera you can replace our DepthCam in depth_camera/DepthCam.py . Please make sure that the functions of the DepthCam are working simliar to our implementation. If you have a Realsense-435 you can use our DepthCam implementation. Please make sure you have installed the realsense sdk and pyrealsense2.


# Installation:

1. Download the data for each module as described in their README

2. Implement your own robot/gripper client

3. Install missing dependencies






# Run:

1. set the working direktory to the AutoPoseEstimation dir

2. run the user iterface from pipeline/main.py
