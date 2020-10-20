from robot_controller.TestController import RobotController
import numpy as np
import os
from depth_camera.DepthCam import DepthCam
import json
import transforms3d
import imageio
from threading import Thread
from pathlib import Path
import time

def get_extra_samples(stop, controller, DC, extra_save_dir, object_pose, symmetric, hand_eye_calibration, view_point_id, min_dist_travelled):

    pose = controller.get_pose(return_mm=True)
    last_pos = np.array([pose['x'], pose['y'], pose['z']])
    while True:

        time.sleep(0.1)
        if stop():
            break

        pose = controller.get_pose(return_mm=True)
        current_pos = np.array([pose['x'], pose['y'], pose['z']])
        dist = np.linalg.norm((current_pos-last_pos))
        if dist >= min_dist_travelled:
            if get_extra_data_sample(controller, DC, extra_save_dir, object_pose, symmetric, hand_eye_calibration, view_point_id):
                last_pos = current_pos

        if stop():
            break


def get_extra_data_sample(controller, DC, extra_save_dir, object_pose, symmetric, hand_eye_calibration, view_point_id):

    # get meta data
    meta = {}
    meta['pose'] = controller.get_pose(return_mm=True)
    meta['joints'] = list(controller.get_joints())

    # get and write frames
    out, success = DC.get_frames(with_repair=False, return_first_try=True, return_first=True, check_state=True)
    if not success:
        return False

    object_tf = np.identity(4)
    object_tf[:3, :3] = transforms3d.euler.euler2mat(np.deg2rad(object_pose.get('a')),
                                                     np.deg2rad(object_pose.get('b')),
                                                     np.deg2rad(object_pose.get('c')))
    object_tf[:3, 3] = [object_pose.get('z'), object_pose.get('y'), object_pose.get('z')]
    object_tf = list(object_tf.flatten())

    meta['object_pose'] = object_tf
    r = [meta['pose']['a'], meta['pose']['b'], meta['pose']['c']]
    anlge = np.linalg.norm(r)
    axis = r / anlge
    trans_mat = np.zeros((4, 4))
    trans_mat[3, 3] = 1
    trans_mat[:3, :3] = transforms3d.axangles.axangle2mat(axis, anlge)
    trans_mat[:3, 3] = [meta['pose']['x'], meta['pose']['y'], meta['pose']['z']]
    trans_mat = trans_mat.flatten()
    for value in range(len(trans_mat)):
        trans_mat[value] = float(trans_mat[value])
    meta['robot2endEff_tf'] = list(trans_mat)

    intr = DC.get_intrinsics()
    meta['intr'] = {'width': intr.width,
                    'height': intr.height,
                    'ppx': intr.ppx,
                    'ppy': intr.ppy,
                    'fx': intr.fx,
                    'fy': intr.fy,
                    'coeffs': intr.coeffs
                    }
    t = time.time()
    meta['depth_scale'] = DC.get_depth_scale()
    meta['symmetric'] = symmetric
    meta['hand_eye_calibration'] = hand_eye_calibration
    meta['view_point_id'] = view_point_id
    imageio.imwrite(extra_save_dir + '/{}.color.png'.format(t), out['image'])
    imageio.imwrite(extra_save_dir + '/{}.depth.png'.format(t), out['depth'])

    with open(extra_save_dir + '/{}.meta.json'.format(t), 'w') as f:
        json.dump(meta, f)
    return True




def get_data(DC, robot_path, name, run, object_pose, symmetric, hand_eye_calibration):

    if symmetric:
        symmetric = 1
    else:
        symmetric = 0

    global moving_robot
    dir = os.path.join(str(Path(__file__).resolve().parent), 'data')
    if not os.path.exists(dir):
        os.makedirs(dir)

    path = os.path.join(str(Path(__file__).resolve().parent.parent), 'robot_controller/robot_path', robot_path)
    state_path = os.path.join(str(Path(__file__).resolve().parent.parent), 'data_generation/state.json')

    with open(path) as f:
        data = json.load(f)

    save_dir = os.path.join(str(Path(__file__).resolve().parent.parent), 'data_generation/data', name, run)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #controller = None
    controller = RobotController()
    vel = 0.60
    acc = 0.3
    min_dist_travelled = 25 # mm
    extra_save_dir = os.path.join(str(Path(__file__).resolve().parent.parent), 'data_generation/data', name, 'extra')
    if not os.path.exists(extra_save_dir):
        os.makedirs(extra_save_dir)

    #input('start obtaining images')
    thread = None
    stop = None
    print('robot home = ', controller.is_home())
    if controller.is_home():
        # get images
        # move to images
        point = 0
        for i, joints in enumerate(data['joints']):

            while True:
                with open(state_path) as f:
                    state_check = json.load(f)
                if state_check['state'] == 1:
                    break
                else:
                    time.sleep(1)
                    print('paused')

            print('_______________________________________________')
            print('position: {}'.format(i))


            if run != 'background' and int(data['via_points'][i]) == 0:
                # take every second run an extra image
                stop = False
                thread = Thread(target=get_extra_samples, args=(lambda : stop,
                                                                controller,
                                                                DC,
                                                                extra_save_dir,
                                                                object_pose,
                                                                symmetric,
                                                                hand_eye_calibration,
                                                                point,
                                                                min_dist_travelled))
                thread.daemon = False
                thread.start()



            print('start moving')
            controller.move_joints(np.deg2rad(joints), moveType='p', vel=vel)

            while (not controller.at_target(joints)) or controller.is_moving():
                print('not at target')
                time.sleep(0.5)


            if run != 'background' and int(data['via_points'][i]) == 0:
                stop = True
                thread.join()


            print('at target')

            if int(data['via_points'][i]) == 0:
                time.sleep(0.5)
                meta = {}
                # get meta data
                meta['joints'] = list(controller.get_joints())
                meta['pose'] = controller.get_pose(return_mm=True)
                object_tf = np.identity(4)
                object_tf[:3, :3] = transforms3d.euler.euler2mat(np.deg2rad(object_pose.get('a')),
                                                                 np.deg2rad(object_pose.get('b')),
                                                                 np.deg2rad(object_pose.get('c')))
                object_tf[:3, 3] = [object_pose.get('z'), object_pose.get('y'), object_pose.get('z')]
                object_tf = list(object_tf.flatten())

                meta['object_pose'] = object_tf
                r = [meta['pose']['a'], meta['pose']['b'], meta['pose']['c']]
                anlge = np.linalg.norm(r)
                axis = r / anlge
                trans_mat = np.zeros((4, 4))
                trans_mat[3, 3] = 1
                trans_mat[:3, :3] = transforms3d.axangles.axangle2mat(axis, anlge)
                trans_mat[:3, 3] = [meta['pose']['x'], meta['pose']['y'], meta['pose']['z']]
                trans_mat = trans_mat.flatten()
                for value in range(len(trans_mat)):
                    trans_mat[value] = float(trans_mat[value])
                meta['robot2endEff_tf'] = list(trans_mat)


                # get and write frames
                out = DC.get_frames(with_repair=True, secure_image=True)

                intr = DC.get_intrinsics()
                meta['intr']={'width': intr.width,
                             'height': intr.height,
                             'ppx': intr.ppx,
                             'ppy': intr.ppy,
                             'fx': intr.fx,
                             'fy': intr.fy,
                             'coeffs': intr.coeffs
                             }
                meta['depth_scale'] = DC.get_depth_scale()
                meta['symmetric'] = symmetric
                meta['hand_eye_calibration'] = hand_eye_calibration
                meta['view_point_id'] = point
                imageio.imwrite(save_dir + '/{:06d}.color.png'.format(point), out['image'])
                imageio.imwrite(save_dir + '/{:06d}.depth.png'.format(point), out['depth'])
                with open(save_dir + '/{:06d}.meta.json'.format(point), 'w') as f:
                    json.dump(meta, f)
                point += 1
                print('got data sample')
            else:
                print('via point')

        moving_robot = False

    else:
        print('robot not in home position')



if __name__ == '__main__':
    DC = DepthCam(fps=30, height=480, width=640)
    #DC = DepthCam(fps=30, height=720, width=1280)
    print('getting images')
    robot_path = 'viewpointsPath2.json'
    name = 'bluedude3'
    runs = [['background', {'x': 0, 'y':0, 'z': 0, 'a': 0, 'b':0, 'c':0}],
        ['foreground', {'x': 0, 'y': 0, 'z': 0, 'a': 0, 'b': 0, 'c': 0}],
        ['foreground180', {'x': 0, 'y': 0, 'z': 0, 'a': 0, 'b': 0, 'c': 180}]]

    for run in runs:
        current_name = name + '/{}'.format(run[0])
        input('current name: {}'.format(current_name))
        get_data(DC, robot_path, name, run[0], run[1])



