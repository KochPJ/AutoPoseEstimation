from pipeline.utils import *
import time
from robot_controller.TestController import RobotController
import transforms3d


constraints = {
    "home": ['j', [0.0, -90.0, 0.0, -90.0, 0.0, 0.0]],
    "via_point": ['j', [-1.93293161e+01, -8.25593825e+01, -8.47928270e+01, -9.00302434e+01,
                        3.57270253e-02, 1.57928464e-02]],
    "grasp_pos": ['j', [-79.3068464, -125.35420593, -45.72337998, -98.47686513, 88.83903427, 21.43752372]],
    "view_points": [['j', [-56.57611344, -125.54468625,  -60.90790138, -102.53858739, 40.49850361, 27.27815167]],
                    ['j', [-49.58489573, -103.54265252, -105.98638492, -40.72687804, 28.49001676, -22.5935181]],
                    ['j', [ -64.02976228, -113.13764762, -125.48807764, 34.6443109, 52.05968136, -79.16827552]],
                    ['j', [-102.16350072, -112.44105029, -117.86479422, 17.05826768, 132.82784992, -148.84610883]],
                    ['j', [-83.63292429, -96.79734894, -90.29489956, -67.31125837, 92.8942132, -271.21859887]]],

    'max_x': 0.24705265462,
    'min_x': -0.2185443788766861,
    'max_y': -0.6827195882797241,
    'min_y': -0.8518663644790649,
    'max_z': 0.09871791303,
    'min_z': -0.02057011425,
    'approach_dist': 0.1
}


def move_to_grasp_position(controller, vel=0.1):
    print('robot home = ', controller.is_home())
    if not controller.is_home():
        return False

    print('move to via point')
    controller.move_joints(np.deg2rad(constraints['via_point'][1]), moveType='p', vel=vel)
    while (not controller.at_target(constraints['via_point'][1])) or controller.is_moving():
        print('not at target')
        time.sleep(0.5)

    print('move to grasp position')
    controller.move_joints(np.deg2rad(constraints['grasp_pos'][1]), moveType='p', vel=vel)
    while (not controller.at_target(constraints['grasp_pos'][1])) or controller.is_moving():
        print('not at target')
        time.sleep(0.5)
    return True


def move_home(controller, vel=0.1):
    if not controller.at_target(constraints['grasp_pos'][1]):
        return False

    print('move to via point')
    controller.move_joints(np.deg2rad(constraints['via_point'][1]), moveType='p', vel=vel)
    while (not controller.at_target(constraints['via_point'][1])) or controller.is_moving():
        print('not at target')
        time.sleep(0.5)

    print('move to home')
    controller.move_joints(np.deg2rad(constraints['home'][1]), moveType='p', vel=vel)
    while (not controller.at_target(constraints['home'][1])) or controller.is_moving():
        print('not at target')
        time.sleep(0.5)
    return True

def get_predictions(controller, DC, end2cam, prediction_dict, vel=0.1):
    predictions = {}
    if not controller.at_target(constraints['grasp_pos'][1]):
        return False, predictions

    for i, joints in enumerate(constraints['view_points']):

        print('move to view point {}/{}'.format(i+1, len(constraints['view_points'])))
        controller.move_joints(np.deg2rad(joints[1]), moveType='p', vel=vel)
        while (not controller.at_target(joints[1])) or controller.is_moving():
            print('not at target')
            time.sleep(0.5)

        cam_data = DC.get_frames()
        prediction = full_prediction(cam_data['image'], cam_data['depth'], **prediction_dict)
        prediction = get_robot2object(prediction, controller, end2cam)

        for cls in prediction['predictions']:
            if cls not in predictions.keys():
                predictions[cls] = {'position': [],
                                    'rotation': []}
            predictions[cls]['position'].append(prediction['predictions'][cls]['position'])
            predictions[cls]['rotation'].append(prediction['predictions'][cls]['rotation'])

    print('move to grasp position')
    controller.move_joints(np.deg2rad(constraints['grasp_pos'][1]), moveType='p', vel=vel)
    while (not controller.at_target(constraints['grasp_pos'][1])) or controller.is_moving():
        print('not at target')
        time.sleep(0.5)

    del_keys = []
    for cls in predictions.keys():
        if len(predictions[cls]['position']) != len(constraints['view_points']):
            del_keys.append(cls)
            continue

        predictions[cls]['position'] = np.mean(np.array(predictions[cls]['position']), axis=0)
        predictions[cls]['rotation'] = np.mean(np.array(predictions[cls]['rotation']), axis=0)

    for key in del_keys:
        del predictions[key]

    return True, predictions

def check_object_position_constraints(pos):
    in_x = False
    in_y = False
    in_z = False
    if constraints['max_x'] > pos[0] > constraints['min_x']:
        in_x = True
    else:
        print('Not in x, x = {}, max_x = {}, min_x = {}'.format(pos[0], constraints['max_x'], constraints['min_x']))

    if constraints['max_y'] > pos[1] > constraints['min_y']:
        in_y = True
    else:
        print('Not in y, y = {}, max_y = {}, min_y = {}'.format(pos[1], constraints['max_y'], constraints['min_y']))

    if constraints['max_z'] > pos[2] > constraints['min_z']:
        in_z = True
    else:
        print('Not in z, z = {}, max_z = {}, min_z = {}'.format(pos[2], constraints['max_z'], constraints['min_z']))

    if in_x and in_y and in_z:
        check = True
    else:
        check = False

    return check


def approach_object(pos, rotation, controller, moveType='p', vel=0.1, acc=0.1):
    if not check_object_position_constraints(pos):
        print('The object does not fulfill the position constraints')
        return False

    pose = {
        'x': pos[0],
        'y': pos[1],
        'z': pos[2] + constraints['approach_dist'],
        'a': rotation[0],
        'b': rotation[1],
        'c': rotation[2],
    }

    move_to_pose, move_on = get_True_or_False('Move to pose {}'.format(pose), default=True)
    if not move_to_pose or not move_on:
        return False

    controller.move_to_pose(pose, moveType=moveType, vel=vel, acc=acc)
    while controller.is_moving():
        print('not at target')
        time.sleep(0.5)
    return True


def return_2_grasp_position(controller, vel=0.1):
    print('move to grasp position')
    controller.move_joints(np.deg2rad(constraints['grasp_pos'][1]), moveType='p', vel=vel)
    while (not controller.at_target(constraints['grasp_pos'][1])) or controller.is_moving():
        print('not at target')
        time.sleep(0.5)
    return True


def move_down(pos, rotation, controller, moveType='l', vel=0.1, acc=0.1):
    pose = {
        'x': pos[0],
        'y': pos[1],
        'z': pos[2],
        'a': rotation[0],
        'b': rotation[1],
        'c': rotation[2],
    }

    move_to_pose, move_on = get_True_or_False('Move to pose {}'.format(pose), default=True)
    if not move_to_pose or not move_on:
        return False

    controller.move_to_pose(pose, moveType=moveType, vel=vel, acc=acc)
    while controller.is_moving():
        print('not at target')
        time.sleep(0.5)

    return True

def main():
    controller = RobotController()
    while True:
        print('______________________________________________________________________________________________')


        pose = controller.get_pose(return_mm=True)
        r = [pose['a'], pose['b'], pose['c']]
        anlge = np.linalg.norm(r)
        axis = r / anlge
        robot2end = np.zeros((4, 4))
        robot2end[3, 3] = 1
        robot2end[:3, :3] = transforms3d.axangles.axangle2mat(axis, anlge)
        robot2end[:3, 3] = [pose['x'], pose['y'], pose['z']]
        euler = np.rad2deg(transforms3d.euler.mat2euler(robot2end[:3, :3]))
        axis, angle = transforms3d.axangles.mat2axangle(robot2end[:3, :3])
        r1 = axis*angle

        tf = transforms3d.euler.euler2mat(*np.deg2rad([0, 0, 180]))
        euler1 = np.rad2deg(transforms3d.euler.mat2euler(tf))
        tf = np.dot(np.linalg.inv(tf), robot2end[:3, :3])
        euler2 = np.rad2deg(transforms3d.euler.mat2euler(tf))
        axis, angle = transforms3d.axangles.mat2axangle(tf)
        r2 = axis*angle

        #print('joints', controller.get_joints())
        print('r', r)
        print('r1', r1)
        print('r2', r2)
        print('euler', euler)
        print('euler1', euler1)
        print('euler2', euler2)
        #print('rot mat', robot2end[:3, :3])
        time.sleep(0.5)

if __name__ == '__main__':
    main()