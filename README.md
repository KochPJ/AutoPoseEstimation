
## Generating Annotated Training Data for 6D Object Pose Estimation in Operational Environments with Minimal User Interaction
This repository is the official implementation of the source code used in the paper "Generating Annotated Training Data for 6D Object Pose Estimation in Operational Environments with Minimal User Interaction" by Koch et al. 2021. 

![alt text](https://github.com/KochPJ/AutoPoseEstimation/blob/main/pipeline/data_gen.png)


# Abstract
Recently developed deep neural networks achieved state-of-the-art results in the subject of 6D object pose estimation for robot manipulation. However, those supervised deep learning methods require expensive annotated training data. Current methods for reducing those costs frequently use synthetic data from simulations, but rely on expert knowledge and suffer from the domain gap when shifting to the real world. Here, we present a proof of concept for a novel approach of autonomously generating annotated training data for 6D object pose estimation. This approach is designed for learning new objects in operational environments while requiring little interaction and no expertise on the part of the user. We evaluate our autonomous data generation approach in two grasping experiments, where we archive a similar grasping success rate as related work on a non autonomously generated data set. 


# Terminal User Interface
We created a Terminal User Interface to quickly test our implementations. After setting up your hardware and installation, you can run it with:

$ python main.py

The Terminal User Interface gives you the following options:
1. Acquire New Data from Object: Collects new data samples for a given object.
2. Create Data Set: Creates a segmentation training data set or converts an existing segmentation training data set into a pose estimation training data set (Requires the pose labels).
3. Create Labels: Generates a segmentation label for each data sample of a given object via the background subtraction or an available trained segmentation model.
4. Create Pose labels: Generates the point clouds of the objects of a given segmentation data set and the finds the target object pose for each data sample based on the available segmentation label. 
5. Grasp objects: Given a trained pose estimation model and the corresponding trained segmentation model, the robot moves into a grasping position. The user can now request the robot to move to a predefined view point in order to perceive the known objects in the scene and their pose. Afterwards, the user can request the robot to grasp the objects one by one in a random order. A corresponding video can be found here: https://youtu.be/qPjmZSX0crU.
6. Run Live Prediction: Either predicts validation data samples of a pose estimation training data set or predicts an incoming camera stream of RGB-D images.
7. Teach Grasping: Allows the user to teach the robot to grasp the objects included in a given pose estimation training data set by demonstraction.
8. Train Segmentation Model: Trains a given segmentation model with a given segmentation training data set. 
9. Train Pose Estimation Model: Trains a "Dense Fusion" 6D pose estimation model given a pose estimation training data set. 
10. Visualise: Visualises the created segmentation labels, the pose labels, and the object point clouds. 

The selections 1, 5, 6, and 7 of the Terminal User Interface are hardware dependent (please see the Hardware section for setup instructions). The rest of the selections can be used with either data aquired by your setup, or with our data (please see the Data section for download instructions). 

# Installation
Dependencies:
1. Linux Distribution (we use [Ubuntu 18.04 LTS](https://releases.ubuntu.com/18.04/))
2. Nvidia GPU with [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html). Find a CUDA version suited for your [Pytorch](https://pytorch.org/get-started/locally/) version (We use CUDA11.0).
3. Install python 3.6, we use for that [Anaconda-Naviagtor](https://www.anaconda.com/products/individual)
4. [Realsense SDK](https://github.com/IntelRealSense/librealsense/releases) and [pyrealsense2](https://pypi.org/project/pyrealsense2/)  (optional for your RealSense DepthCamera) 

Python 3.6 Packages (we put the pgk vision we used in brackets): 
1. torch (torch==1.7.1+cu110)
2. torchvision (0.8.2+cu110)
3. open3d (0.9.0.0)
4. segmentation-models-pytorch (0.1.3) 
5. Pillow (8.1.2)
6. opencv-python (4.5.1.48)
7. matplotlib (3.2.1)
8. transforms3d (0.3.1)
9. [optional] pyrealsense2 (2.33.1.1388)

Run Terminal User Interface:
1. $ python main.py

# Hardware
In order to conduct your own grasping experiments or aquire new data you need a RGB-D Camera, an industrial robot armm, and a gripper. We use an ["Realsense-435"](https://www.intelrealsense.com/depth-camera-d435/) depth camera, the ["UR-5 CB3"](https://www.universal-robots.com/cb3/) robot arm, and the ["Robotiq 2F-85"](https://robotiq.com/products/2f85-140-adaptive-robot-gripper) gripper. Futhermore, you need to find the hand-eye-calibration for your setup, and adapt the robot view points used for data acquisition and grasping.

1. Robot and Gripper: We are not providing any drivers for the robot and gripper. If you want to use your own setup you will need to write your own drivers and comunication. We provide a "robot and gripper" controller in ["robotcontroller/TestController.py"](https://github.com/KochPJ/AutoPoseEstimation/blob/main/robot_controller/TestController.py). It uses a "robot and gripper" client to interact with the hardware. You can take this as a starting point to connect your hardware. Please make sure that all functions in the ["RobotController"](https://github.com/KochPJ/AutoPoseEstimation/blob/b7e27e59aa1e5fd1f337615585ac569d41a74d03/robot_controller/TestController.py#L19) are callable. 
 
2. Camera: If you want to use your own RGB-D Camera you can replace our ["DepthCam"](https://github.com/KochPJ/AutoPoseEstimation/blob/b7e27e59aa1e5fd1f337615585ac569d41a74d03/depth_camera/DepthCam.py#L6) in [".depth_camera/DepthCam.py"](https://github.com/KochPJ/AutoPoseEstimation/blob/main/depth_camera/DepthCam.py). Please make sure that the functions of the ["DepthCam"](https://github.com/KochPJ/AutoPoseEstimation/blob/b7e27e59aa1e5fd1f337615585ac569d41a74d03/depth_camera/DepthCam.py#L6) are working simliar to our implementation. If you have a Realsense-435 you can use our DepthCam implementation. Please make sure you have installed the realsense sdk and pyrealsense2 (see dependencies).

3. Hand-Eye-Calibration: We use a aruco board for hand-eye-calibration. You can use our [hand-eye-calibration implementations](https://github.com/KochPJ/AutoPoseEstimation/blob/b7e27e59aa1e5fd1f337615585ac569d41a74d03/hand_eye_calibration/getPoses.py) to get the camera poses. However, in order to get the robot poses you need to implement your own robot controller first (see the Hardware section). We do not provide the implementation our hand-eye-calibration method, since we reused it from an other project and it is written in c++ with furhter requirements. (our implementation is based on ["CamOdoCal"](https://github.com/hengli/camodocal)) 

4. View-Points: The data aquireing requires a set of view points, which are unique to your setup. So remember to make your own set of viewpoints for your unique setup. The grasping also requires a set of viewpoints, which need to be updated according to your setup. Please find the viewpoints [here](https://github.com/KochPJ/AutoPoseEstimation/tree/b7e27e59aa1e5fd1f337615585ac569d41a74d03/robot_controller/robot_path). You can use our [path creation](https://github.com/KochPJ/AutoPoseEstimation/blob/b7e27e59aa1e5fd1f337615585ac569d41a74d03/robot_controller/createPath.py) or implement your own method. 

5. Reference Point: In our setup we have defined a reference point

# Data
All data used in the components of this project can be downloaded. A download link and instructions can be found in the readme of the each component. The comonents with data are ["background_subtraction"](https://github.com/KochPJ/AutoPoseEstimation/tree/b7e27e59aa1e5fd1f337615585ac569d41a74d03/background_subtraction), ["data_generation"](https://github.com/KochPJ/AutoPoseEstimation/tree/b7e27e59aa1e5fd1f337615585ac569d41a74d03/data_generation), ["DenseFusion"](https://github.com/KochPJ/AutoPoseEstimation/tree/b7e27e59aa1e5fd1f337615585ac569d41a74d03/DenseFusion), ["label_generator"](https://github.com/KochPJ/AutoPoseEstimation/tree/b7e27e59aa1e5fd1f337615585ac569d41a74d03/label_generator), ["pc_reconstruction"](https://github.com/KochPJ/AutoPoseEstimation/tree/main/pc_reconstruction), ["experiments"](https://github.com/KochPJ/AutoPoseEstimation/blob/main/experiments/README.md), and ["segmentation"](https://github.com/KochPJ/AutoPoseEstimation/tree/main/segmentation). 

1. Aquired Data: If you do not have a hardware setup, you can download the RGB-D data aquired for this project follwing the instructions [here](https://github.com/KochPJ/AutoPoseEstimation/blob/b7e27e59aa1e5fd1f337615585ac569d41a74d03/data_generation/README.md).

2. Generate labels: You can generate your own segmentation label, point cloud, and target pose with selection 3 and 4 of the Terminal User Interface. Otherwise, you can download our generated labels by following the instructions [here](https://github.com/KochPJ/AutoPoseEstimation/blob/main/label_generator/README.md).

3. Extra Data: During data acquisition, we took every 50mm traveled an extra data sample, while the robot was moving. That data has no background image and a segmentation label can not be generated via background subtraction. However, we can use a segmentation model - once trained on the other data - to annotate the extra data. The extra data, in turn, can be used with the other data for the pose estimation training. However, experiments we conducted did not show any improvements when training with extra data. This might be due to poor labels related to motion blur and a time offset between the capured image and the robot pose. More detail can be found in the thesis report connected to this paper(see section Thesis Report below).


# Experiments
You can reproduce our experiments with the Terminal User Interface. 
1. You can use the ["eval.py"](https://github.com/KochPJ/AutoPoseEstimation/blob/main/experiments/eval.py) to evaluate the ADD performance of your pose estimation model. 
2. You can use the ["gt_test.py"](https://github.com/KochPJ/AutoPoseEstimation/blob/main/experiments/gt_test.py) to evaluate the IoU, accurancy, precision, recall, and f1 score of the segmentation model and background subtraction model, with respekt to 20% hand labelled data. 
3. We use the ["train_pose_estimation_exp.py"](https://github.com/KochPJ/AutoPoseEstimation/blob/main/experiments/train_pose_estimation_exp.py) to train multiple pose estimation models with a decreasing numer of view points, and with or without extra data. You can use the ["eval_exp.py"](https://github.com/KochPJ/AutoPoseEstimation/blob/main/experiments/eval_exp.py) to evaluate how the ADD performacne changes when training pose estimation with data from less view points, and what effect the extra data has on the training. You can also quickly look into the thesis report for that (see section Thesis Report). 

# Background Subtraction
If you want to investigate the training of our background subtraction model you can do that [here](https://github.com/KochPJ/AutoPoseEstimation/blob/b7e27e59aa1e5fd1f337615585ac569d41a74d03/background_subtraction/__init__.py). You can also download the data used for the training [here](https://github.com/KochPJ/AutoPoseEstimation/blob/b7e27e59aa1e5fd1f337615585ac569d41a74d03/background_subtraction/README.md)

# Thesis Report
This work is based on the MasterThesis by Paul Koch. If you are interested in more details, you can download the full thesis report [here](https://drive.google.com/file/d/14JZ0-bFKVdxH3xTPsmLR0Vqkz8tzS0uX/view?usp=sharing). 




