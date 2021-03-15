## AutoPoseEstimation

# Introduction
Recently developed deep neural networks achieved state-of-the-art results in the subject of 6D object pose estimation for robot manipulation. However, those supervised deep learning methods require expensive annotated training data. Current methods for reducing those costs frequently use synthetic data from simulations, but rely on expert knowledge and suffer from the domain gap when shifting to the real world. Here, we present a proof of concept for a novel approach of autonomously generating annotated training data for 6D object pose estimation. This approach is designed for learning new objects in operational environments while requiring little interaction and no expertise on the part of the user. We evaluate our autonomous data generation approach in two grasping experiments, where we archive a similar grasping success rate as related work on a non autonomously generated data set. 

# Hardware
In order to conduct your own grasping experiments or aquire new data you need a RGB-D Camera, an industrial robot armm, and a gripper. We use an "Realsense-435" depth camera, the "UR-5" robot arm, and the "Robotiq 2F-85" gripper. 

1. Robot and Gripper: We are not providing any drivers for the robot and gripper. If you want to use your own setup you will need to write your own drivers and comunication. We provide a "robot and gripper" controller in "robotcontroller/TestController.py". It uses a "robot and gripper" client to interact with the hardware. You can take this as a starting point to connect your hardware. Please make sure that all functions in the "RobotController"are callable. 
 
2. Camera: If you want to use your own RGB-D Camera you can replace our DepthCam in ".depth_camera/DepthCam.py". Please make sure that the functions of the "DepthCam" are working simliar to our implementation. If you have a Realsense-435 you can use our DepthCam implementation. Please make sure you have installed the realsense sdk and pyrealsense2.

3. Hand-Eye-Calibration: We use a aruco board for hand-eye-calibration. You can use our hand-eye-calibration implementations in the folder "hand_eye_calibration" to get the camera poses. However, in order to get the robot poses you need to implement your own robot controller first. We do not provide the implementation our hand-eye-calibration method, since we reused it from an other project and it is written in c++ with furhter requirements. 

4. View-Points: The data aquireing requires a set of view points, which are unique to your setup. So remember to make your own set of viewpoints for your setup.The grasping also requires a set of viewpoints, which need to be updated according to your setup. Please find the viewpoints under ".robot_controller/robot_path". You can use the our path creation under ".robot_controller/createPath.py" or implement your own method. 

# Data
All data used in the components of this project can be downloaded. A download link and instructions can be found in the readme of the each component.


# Installation:

1. Download the data for each module as described in their README

2. Implement your own robot/gripper client

3. Install missing dependencies


# Terminal User Interface
We have created a Terminal User Interface to quickly test our implementations. You can run it with:

$ python main.py

The Terminal User Interface gives you the following options:
1. Acquire New Data from Object: Collects new data samples for a given object.
2. Create Labels: Generates a segmentation label for each data sample of a given object via the background subtraction or an available trained segmentation model.
3. Create Data Set: Creates a segmentation training data set or converts an existing segmentation training data set into a pose estimation training data set (Requires the pose labels).
4. Train Segmentation Model: Trains a given segmentation model with a given segmentation training data set. 
5. Create Pose labels: Generates the point clouds of the objects of a given segmentation data set and the finds the target object pose for each data sample based on the available segmentation label. 
6. Train Pose Estimation Model: Trains a "Dense Fusion" 6D pose estimation model given a pose estimation training data set. 
7. Run Live Prediction: Either predicts validation data samples of a pose estimation training data set or predicts an incoming camera stream of RGB-D images.
8. Visualise: Visualises the created segmentation labels, the pose labels, and the object point clouds. 
9. Teach Grasping: Allows the user to teach the robot to grasp the objects included in a given pose estimation training data set by demonstraction.
10. Grasp objects: Given a trained pose estimation model and the corresponding trained segmentation model, the robot moves into a grasping position. The user can now request the robot to move to a predefined view point in order to perceive the known objects in the scene and their pose. Afterwards, the user can request the robot to grasp the objects one by one in a random order. A corresponding video can be found here: https://youtu.be/qPjmZSX0crU.





# Run:

1. set the working direktory to the AutoPoseEstimation dir


