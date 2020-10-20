from .TestController import RobotController
#import DepthCam
import json
from threading import Thread
import os


def create_path(name):

    path = './robot_path'
    if not os.path.exists(path):
        os.makedirs(path)
    print('creating Robot Controller')
    controller = RobotController()

    data = {
        'joints': [],
        'cart_pose': [],
        'via_points': []
    }
    via = True
    vias = 0
    points = 0
    print('starting loop')
    while True:
        p = input('enter for next position, v for via point, e for exit')
        if p == 'e':
            break
        elif p == 'v':
            via = True
        else:
            via = False


        currentJoints = list(controller.get_joints())
        currentPose = controller.get_pose()
        if len(data['joints']) > 0:
            same = True
            for i in range(len(currentJoints)):
                if currentJoints[i] != data['joints'][-1][i]:
                    same = False
                    break

            if same:
                print('same point')
            else:
                data['joints'].append(currentJoints)
                data['cart_pose'].append(currentPose)
                print()
                if via:
                    data['via_points'].append('1')
                    vias += 1
                    print('Added via. vias = {}, points = {}, total points = {}, joints = {}'.format(vias, points, vias+points, currentJoints))
                else:
                    points += 1
                    data['via_points'].append('0')
                    print('Added point. vias = {}, points = {}, total points = {}, joints = {}'.format(vias, points, vias + points, currentJoints))
        else:
            data['joints'].append(currentJoints)
            data['cart_pose'].append(currentPose)
            if via:
                data['via_points'].append('1')
                vias += 1
                print('Added via. vias = {}, points = {}, total points = {}, joints = {}'.format(vias, points, vias + points, currentJoints))
            else:
                points += 1
                data['via_points'].append('0')
                print('Added point. vias = {}, points = {}, total points = {}, joints = {}'.format(vias, points, vias + points, currentJoints))

    save_path = os.path.join(path, '{}.json'.format(name))

    #print(save_path)
    with open(save_path, 'w') as f:
        json.dump(data, f)




def main():
    name = 'handEyeCalibPath4'
    create_path(name)


if __name__ == '__main__':
	main()
