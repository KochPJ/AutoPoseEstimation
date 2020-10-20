from robot_controller.TestController import RobotController
import numpy as np
import time
import os
import cv2
from depth_camera.DepthCam import DepthCam
import json
from hand_eye_calibration.calib import get_cam_poses
import transforms3d


def take_images(positions, via_pos, controller, vel, DC, intr):

    images = []
    depths = []
    data = {'joints':[],
            'cart_pose': []}
    for i, joints in enumerate(positions):
        print('currentjoints = {}, joints = {}, diff = {}'.format(controller.get_joints(),joints, joints - controller.get_joints()))

        controller.move_joints(np.deg2rad(joints), moveType='p', vel=vel)
        print('start moving')
        while not controller.at_target(joints):
            print('not at target')
            time.sleep(0.5)

        if int(via_pos[i]) == 0:
            time.sleep(0.5)

            data['joints'].append(list(controller.get_joints()))
            data['cart_pose'].append(controller.get_pose(return_mm=True))
            out = DC.get_frames()
            images.append(out['image'])
            depths.append(out['depth'])
            cv2.imshow('image', out['image'])
            cv2.waitKey(100)

            try:
                _ = get_cam_poses([out['image']], intr)
                print('got cam poses')
            except:
                print('failed to get cam poses')

        else:
            print('via point')


    return images, depths, data

def get_camera_poses(DC, name):
    dir = './data'
    if not os.path.exists(dir):
        os.makedirs(dir)

    path = os.path.join('/home/kochpaul/projects/masterthesis/robot_controller/robot_path', name)

    intr = DC.get_intrinsics()
    meta = {'width': intr.width,
            'height': intr.height,
            'ppx': intr.ppx,
            'ppy': intr.ppy,
            'fx': intr.ppx,
            'fy': intr.ppy,
            'coeffs': intr.coeffs
    }

    with open(path) as f:
        data = json.load(f)


    controller = RobotController()
    vel = 0.30

    input('start obtaining images')
    print('robot home = ', controller.is_home())
    if controller.is_home():

        # get images
        images, depths, meta2 = take_images(data['joints'], data['via_points'], controller, vel, DC, intr)
        meta['joints'] = meta2['joints']
        meta['cart_pose'] = meta2['cart_pose']

        # get robot poses
        robot_poses = []
        for pose in meta['cart_pose']:
            r = [pose['a'], pose['b'], pose['c']]
            anlge = np.linalg.norm(r)
            axis = r/anlge
            trans_mat = np.zeros((4, 4))
            trans_mat[3, 3] = 1
            rot_mat = transforms3d.axangles.axangle2mat(axis, anlge)
            trans_mat[:3, :3] = rot_mat
            trans_mat[:3, 3] = [pose['x'], pose['y'], pose['z']]
            trans_mat = trans_mat.flatten()
            for i in range(len(trans_mat)):
                trans_mat[i] = float(trans_mat[i])
            trans_mat = list(trans_mat)
            robot_poses.append(trans_mat)
        meta['robot_poses'] = robot_poses

        robot_poses = np.array(robot_poses)

        # save robot poses
        save_path = '/home/kochpaul/projects/masterthesis/hand_eye_calibration/data'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        robot_poses_path = os.path.join(save_path, 'robot_poses.yaml')
        fs_write = cv2.FileStorage(robot_poses_path, cv2.FILE_STORAGE_WRITE)
        fs_write.write("poses", robot_poses)
        fs_write.release()

        # get and save cam poses
        cam_poses_np = get_cam_poses(images, intr, save_path)
        cam_poses = []
        for i in range(len(cam_poses_np)):
            pose = cam_poses_np[i]
            for j in range(len(pose)):
                pose[j] = float(pose[j])
            pose = list(pose)
            cam_poses.append(pose)

        meta['cam_poses'] = cam_poses

        with open(dir + '/meta.json', 'w') as f:
            json.dump(meta, f)

    else:
        print('robot not in home position')



if __name__ == '__main__':
    #DC = DepthCam(fps=6, height=480, width=640)
    DC = DepthCam(fps=15, height=720, width=1280)
    #DC.stream()

    print('getting images')
    name = 'handEyeCalibPath4.json'
    get_camera_poses(DC, name)



