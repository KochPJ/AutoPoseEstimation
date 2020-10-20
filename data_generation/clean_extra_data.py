import os
import json
import numpy as np
import transforms3d

'''
use only for data with foreground and foreground 180
'''

path = './data'
classes = list(os.listdir(path))
tag = '.meta.json'
l = len(tag)
for cls in classes:
    print('_________________')
    print('class: ', cls)
    extra_path = os.path.join(path, cls, 'extra')
    extra_dirs = sorted(list(os.listdir(extra_path)))
    extra_dirs = sorted([d for d in extra_dirs if tag in d])
    times = [int(float(d[:-l])) for d in extra_dirs]
    max_dist = 0
    at = 0
    for i, t in enumerate(times[:-1]):
        dist = times[i+1]-t
        if dist > max_dist:
            max_dist = dist
            at = i

    '''
    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    fig.suptitle(cls,fontsize=16)
    plt.subplot(1, 2, 1)
    plt.title('foreground')
    plt.plot(times[:at], list(range(len(times[:at]))))
    plt.subplot(1, 2, 2)
    plt.title('foreground 180')
    plt.plot(times[at+1:], list(range(len(times[at+1:]))))
    plt.show()
    '''

    print('check for foreground')
    with open(os.path.join(extra_path, extra_dirs[0])) as f:
        meta = json.load(f)
        pc_rotation = np.array(meta.get('object_pose')).reshape(4, 4)[:3, :3]
        first_rotation = np.rad2deg(transforms3d.euler.mat2euler(pc_rotation))

    rotations = [{'rot': first_rotation,
                  'indexes': [0]}]

    for i, d in enumerate(extra_dirs[1:at]):
        with open(os.path.join(extra_path, d)) as f:
            meta = json.load(f)
            pc_rotation = np.array(meta.get('object_pose')).reshape(4, 4)[:3, :3]
            this_rotation = np.rad2deg(transforms3d.euler.mat2euler(pc_rotation))

            not_in_rotatins = True
            for rot in rotations:
                if np.array_equal(rot['rot'], this_rotation):
                    not_in_rotatins = False
                    rot['indexes'].append(i+1)
                    break

            if not_in_rotatins:
                rotations.append({'rot': this_rotation,
                                  'indexes': [i+1]})

    for rot in rotations:
        print('rot: {}, n indexes: {}, first and last index: {}, dist: {}'.format(
            rot['rot'],  len(rot['indexes']), [rot['indexes'][0], rot['indexes'][-1]],
            rot['indexes'][-1]-rot['indexes'][0]+1))
        if not np.array_equal(rot['rot'], first_rotation):
            print('delete indexes of rot: {}'.format(rot['rot']))
            print([rot['indexes'][0], rot['indexes'][-1]], rot['indexes'][-1]-rot['indexes'][0]+1)
            for index in rot['indexes']:
                id = extra_dirs[index][:-l]
                curr_path = os.path.join(extra_path, '{}.color.png'.format(id))
                if os.path.exists(curr_path):
                    os.remove(curr_path)
                curr_path = os.path.join(extra_path, '{}.depth.png'.format(id))
                if os.path.exists(curr_path):
                    os.remove(curr_path)
                curr_path = os.path.join(extra_path, '{}.meta.json'.format(id))
                if os.path.exists(curr_path):
                    os.remove(curr_path)


            print('')
    print('check for foreground 180')
    with open(os.path.join(extra_path, extra_dirs[at+1])) as f:
        meta = json.load(f)
        pc_rotation = np.array(meta.get('object_pose')).reshape(4, 4)[:3, :3]
        first_rotation = np.rad2deg(transforms3d.euler.mat2euler(pc_rotation))

    rotations = [{'rot': first_rotation,
                  'indexes': [at+1]}]
    for i, d in enumerate(extra_dirs[at+2:]):
        with open(os.path.join(extra_path, d)) as f:
            meta = json.load(f)
            pc_rotation = np.array(meta.get('object_pose')).reshape(4, 4)[:3, :3]
            this_rotation = np.rad2deg(transforms3d.euler.mat2euler(pc_rotation))

            not_in_rotatins = True
            for rot in rotations:
                if np.array_equal(rot['rot'], this_rotation):
                    not_in_rotatins = False
                    rot['indexes'].append(i+at+2)
                    break

            if not_in_rotatins:
                rotations.append({'rot': this_rotation,
                                  'indexes': [i+at+2]})

    for rot in rotations:
        print('rot: {}, n indexes: {}, first and last index: {}, dist: {}'.format(
            rot['rot'],  len(rot['indexes']), [rot['indexes'][0], rot['indexes'][-1]],
            rot['indexes'][-1]-rot['indexes'][0]+1))

        if not np.array_equal(rot['rot'], first_rotation):
            print('delete indexes of rot: {}'.format(rot['rot']))
            print([rot['indexes'][0], rot['indexes'][-1]], rot['indexes'][-1]-rot['indexes'][0]+1)
            for index in rot['indexes']:
                id = extra_dirs[index][:-l]
                curr_path = os.path.join(extra_path, '{}.color.png'.format(id))
                if os.path.exists(curr_path):
                    os.remove(curr_path)
                curr_path = os.path.join(extra_path, '{}.depth.png'.format(id))
                if os.path.exists(curr_path):
                    os.remove(curr_path)
                curr_path = os.path.join(extra_path, '{}.meta.json'.format(id))
                if os.path.exists(curr_path):
                    os.remove(curr_path)







