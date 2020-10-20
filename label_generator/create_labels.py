from label_generator.utils import createLabel_RGBD
import os
import numpy as np
import matplotlib.pyplot as plt
import transforms3d
from pc_reconstruction.create_pointcloud import load_point_cloud
import open3d as o3d
import pc_reconstruction.open3d_utils as pc_utils
import json
from segmentation.utils import get_model
import torch
from PIL import Image
from torchvision import transforms
import cv2
from torch.nn import functional as F
import copy
import time


def get_default_model(root, ds_name, n_classes):
    segmentation_config = {'encoder_name': 'resnet34',
                           'encoder_weights': None,
                           'activation': 'softmax',
                           'in_channels': 3,
                           'classes': n_classes}
    name = 'Unet'
    model = get_model(name, segmentation_config)

    cp = torch.load(os.path.join(root,
                                 'segmentation',
                                 'trained_models',
                                 ds_name,
                                 '{}_{}.ckpt'.format(name, segmentation_config['encoder_name'])),
                    map_location=torch.device('cpu'))
    model.load_state_dict(cp['state_dict'])

    return model


def create_pose_data(root, classes, ds_name, reference_point=np.array([]), new_pred=True, get_extra_labels=False,
                     plot=False, use_cuda=True):

    if torch.cuda.is_available() and use_cuda:
        device = torch.device('cuda:0')
        cuda = True
    else:
        cuda = False
        device = torch.device('cpu')

    if new_pred:
        mode = 'new_pred'
    else:
        mode = 'pred'

    model = get_default_model(root, ds_name, len(classes) + 1)
    model.to(device)
    model.eval()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean, std)
    print('-----------')
    stats = {'n_samples': 0,
             'n_extra_samples': 0,
             'bs_copied': 0,
             'no_depth_overlap': 0,
             'not_in_center': 0}

    elapsed_times = []
    times_new_pred = []
    times_pc = []
    times_pose = []
    for class_id, cls in enumerate(classes):
        t_start = time.time()
        print('class: {}'.format(cls))
        data_path = os.path.join(root, 'data_generation', 'data', cls)
        dirs = os.listdir(data_path)
        if 'background' in dirs:
            i = dirs.index('background')
            del dirs[i]

        if not get_extra_labels:
            if 'extra' in dirs:
                i = dirs.index('extra')
                del dirs[i]

        t_new_pred = time.time()
        for k, d in enumerate(dirs):
            data_dir = os.path.join(data_path, d)
            samples = list(os.listdir(data_dir))
            label_path = os.path.join(root, 'label_generator/data', cls, d)
            if not os.path.exists(label_path):
                os.makedirs(label_path)

            samples = sorted([d[:-10] for d in samples if '.color.png' in d])
            for i, id in enumerate(samples):
                print('class {}/{}, rotation {}/{}, sample {}/{}'.format(class_id + 1, len(classes),
                                                                         k+1, len(dirs),
                                                                         i + 1, len(samples)))
                if d == 'extra' or new_pred:
                    with open(os.path.join(data_dir, '{}.meta.json'.format(id))) as f:
                        meta = json.load(f)
                        robot2endEff_tf = np.array(meta.get('robot2endEff_tf')).reshape(4, 4)
                        hand_eye_calibration = np.array(meta.get('hand_eye_calibration')).reshape(4, 4)
                        robot2cam = np.dot(robot2endEff_tf, hand_eye_calibration)
                        pos = robot2cam[:3, 3]
                        measure_dist = np.linalg.norm(reference_point - pos)
                        max_measure_dist = measure_dist + 150
                        min_measure_dist = measure_dist - 150

                    with open(os.path.join(data_dir, '{}.depth.png'.format(id)), 'rb') as f:
                        depth = np.array(Image.open(f), dtype=np.float)
                        depth[depth > max_measure_dist] = 0
                        depth[depth < min_measure_dist] = 0

                    with open(os.path.join(data_dir, '{}.color.png'.format(id)), 'rb') as f:
                        x = Image.open(f).convert('RGB')
                        if plot:
                            x_copy = np.array(copy.deepcopy(x), dtype=np.uint8)

                    x = to_tensor(x)
                    x = normalize(x)
                    x = x.to(device)
                    x = x.unsqueeze(0)

                    pred = model.predict(x)
                    pred = F.softmax(pred, dim=1)
                    if cuda:
                        pred = pred.cpu()[0]
                    else:
                        pred = pred[0]
                    pred_arg = torch.argmax(pred, dim=0).numpy()
                    pred_arg[pred_arg != class_id+1] = 0
                    pred = pred_arg*pred[class_id+1].numpy()

                    ret, labels = cv2.connectedComponents(np.array(pred_arg, dtype=np.uint8), connectivity=8)
                    biggest = 1
                    biggest_score = 0
                    for u in np.unique(labels)[1:]:
                        score = np.mean(pred[labels == u])
                        if score > biggest_score:
                            biggest_score = score
                            biggest = u

                    pred[labels != biggest] = 0
                    pred[pred != 0] = 255
                    pred = np.array(pred, dtype=np.uint8)

                    if plot:
                        plt.subplot(1, 3, 1)
                        plt.axis('off')
                        plt.title('image')
                        plt.imshow(x_copy)
                        plt.subplot(1, 3, 2)
                        plt.axis('off')
                        plt.title('pred target = {}'.format(class_id+1))
                        plt.imshow(np.array(pred_arg, dtype=np.uint8))
                        plt.subplot(1, 3, 3)
                        plt.axis('off')
                        plt.title('pred cca')
                        plt.imshow(pred)
                        plt.show()

                    # check upon we can trust the new prediction
                    save = False
                    if d != 'extra':
                        with open(os.path.join(label_path, '{}.pred.label.png'.format(id, mode)), 'rb') as f:
                            bs_label = np.array(Image.open(f), dtype=np.uint8)
                        if len(np.unique(pred[bs_label != 0])) <= 1:
                            area = np.sum([pred != 0])
                            before_area = np.sum([pred_arg != 0])
                            area_diff = np.round((1-(area/before_area))*100, 2)
                            print('no pred, copy background subtraction pred.')
                            print('area = {}, area before cca: {}, diff {}%'.format(area, before_area, area_diff))
                            pred = bs_label
                            save = True
                            stats['bs_copied'] += 1


                    if not save:
                        depth_overlap = True
                        if len(np.unique(pred[depth != 0])) <= 1:
                            print('estimated depth does not overlap')
                            depth_overlap = False
                            stats['no_depth_overlap'] += 1

                        if depth_overlap:
                            s0 = pred.shape[0]
                            s1 = pred.shape[1]
                            cut0 = 30
                            cut1 = 50
                            if len(np.unique(pred[cut0:s0-cut0, cut1: s1-cut1])) > 1:
                                save = True
                            else:
                                print('pred not in center')
                                stats['not_in_center'] += 1

                    if save:
                        if d == 'extra':
                            stats['n_extra_samples'] += 1
                        else:
                            stats['n_samples'] += 1
                        label = Image.fromarray(pred)
                        # print('save new pred to ', os.path.join(label_path, '{}.new_pred.label.png'.format(id)))
                        label.save(os.path.join(label_path, '{}.new_pred.label.png'.format(id)))
                    else:
                        print('not saved', cls, d, id)
                        if os.path.exists(os.path.join(label_path, '{}.new_pred.label.png'.format(id))):
                            print('deleting old image')
                            os.remove(os.path.join(label_path, '{}.new_pred.label.png'.format(id)))

                        if os.path.exists(os.path.join(label_path, '{}.meta.json'.format(id))):
                            print('deleting old meta data')
                            os.remove(os.path.join(label_path, '{}.meta.json'.format(id)))

        times_new_pred.append(time.time()-t_new_pred)

        t_pc = time.time()
        n_viewpoints = 30
        min_friends = 20
        min_dist = 5
        nb_neighbors = 20

        threshold = 10
        voxel_size = 2
        voxel_size_out = 5

        l_arrow = 75
        global_regression = False
        icp_point2point = True
        icp_point2plane = False

        plot = False
        print('getting point cloud')
        save_dir = os.path.join(root, 'pc_reconstruction/data')
        load_point_cloud(cls,
                         save_dir,
                         root,
                         reference_point=reference_point,
                         mode=mode,
                         n_viewpoints=n_viewpoints,
                         min_friends=min_friends,
                         min_dist=min_dist,
                         nb_neighbors=nb_neighbors,
                         threshold=threshold,
                         voxel_size=voxel_size,
                         voxel_size_out=voxel_size_out,
                         l_arrow=l_arrow,
                         global_regression=global_regression,
                         icp_point2point=icp_point2point,
                         icp_point2plane=icp_point2plane,
                         plot=plot)
        times_pc.append(time.time()-t_pc)
        t_pose = time.time()
        print('getting pose label')
        create_pose_label(root,
                          cls,
                          global_regression,
                          icp_point2point,
                          icp_point2plane,
                          plot=plot,
                          view_label=plot,
                          with_extra=get_extra_labels)
        times_pose.append(time.time()-t_pose)

        print('_____________________')
        print('stats: {}'.format(stats))
        print('_____________________')
        elapsed_time = time.time() - t_start
        print('elapsed time {} sec, seg: {} sec, pc: {} sec, pose: {} sec'.format(np.round(elapsed_time, 2),
                                                                                  np.round(times_new_pred[-1], 2),
                                                                                  np.round(times_pc[-1], 2),
                                                                                  np.round(times_pose[-1], 2)))
        print('Avg. task times per object: seg: {}, pc: {}, pose: {}'.format(np.round(np.mean(times_new_pred), 2),
                                                                             np.round(np.mean(times_pc), 2),
                                                                             np.round(np.mean(times_pose), 2)))
        print('Total time per task: seg: {}, pc: {}, pose: {}'.format(np.round(np.sum(times_new_pred), 2),
                                                                      np.round(np.sum(times_pc), 2),
                                                                      np.round(np.sum(times_pose), 2)))
        print('_____________________')
        elapsed_times.append(elapsed_time)


    print('____________________________________________________________________________________________')
    print('Elapsed time for "{}" objects: {} sec, with a per object time of: {} sec'.format(
        len(classes), np.round(np.sum(elapsed_times), 2), np.round(np.mean(elapsed_times), 2)))

    print('____________________________________________________________________________________________')



def create_pose_label(root,
                      object_name,
                      global_regression,
                      icp_point2point,
                      icp_point2plane,
                      plot=False,
                      view_label=False,
                      with_extra=False):

    object_path = os.path.join(root, 'data_generation/data', object_name)
    dirs = os.listdir(object_path)
    background_path = os.path.join(object_path, 'background')
    pc_path = os.path.join(root, 'pc_reconstruction/data', object_name, '{}_out.ply'.format(object_name))
    pco_path = os.path.join(root, 'pc_reconstruction/data', object_name, '{}.ply'.format(object_name))


    try:
        i = dirs.index('background')
        del dirs[i]
    except:
        raise ValueError('background does not exist in object_path: {}'.format(object_path))

    if 'extra' in dirs:
        i = dirs.index('extra')
        del dirs[i]
        if with_extra:
            dirs.append('extra')

    if len(dirs) < 1:
        raise ValueError('no foreground')

    remember_pos_and_rot = []
    for d in dirs:
        pc_position = None
        pc_rotation = None
        data_path = os.path.join(object_path, d)
        label_path = os.path.join(root, 'label_generator/data', object_name, d)
        print('Getting pc position for {}'.format(d))
        if d != 'extra':
            source = o3d.io.read_point_cloud(pc_path)
            pc_position = pc_utils.get_my_source_center(source)

            # exchange with read correct from meta

            for meta_file in os.listdir(data_path):
                if meta_file[-5:] == '.json':
                    with open(os.path.join(data_path, meta_file)) as f:
                        meta = json.load(f)
                        pc_rotation = np.array(meta.get('object_pose')).reshape(4, 4)[:3, :3]
                    break

            old_rotation = np.rad2deg(transforms3d.euler.mat2euler(pc_rotation))
            if not np.array_equal(old_rotation, np.array([0.0, 0.0, 0.0])):

                # load point cloud
                pc_target_path = os.path.join(root, 'pc_reconstruction/data', object_name, '{}.ply'.format(d))

                target = o3d.io.read_point_cloud(pc_target_path)
                source = o3d.io.read_point_cloud(pc_path)


                #pc_utils.draw_registration_result(source, target, np.identity(4))
                # match pointcloud to current surface
                target, source, init_tf = pc_utils.icp_regression(target,
                                                                  source,
                                                                  voxel_size=5,
                                                                  threshold=10,
                                                                  global_regression=global_regression,
                                                                  icp_point2point=icp_point2point,
                                                                  icp_point2plane=icp_point2plane,
                                                                  plot=plot)



                print('requested rotation: {}'.format(old_rotation))
                pc_rotation = np.dot(pc_rotation, init_tf[:3, :3])
                diff_rotation = np.rad2deg(transforms3d.euler.mat2euler(init_tf[:3, :3]))
                euler = np.array(transforms3d.euler.mat2euler(pc_rotation))
                # set zeros where there should be no rotations
                for i, angle in enumerate(old_rotation):
                    if angle == 0.0:
                        euler[i] = 0.0
                        diff_rotation[i] = 0.0

                pc_rotation = transforms3d.euler.euler2mat(euler[0], euler[1], euler[2])
                new_rotation = np.rad2deg(transforms3d.euler.mat2euler(pc_rotation))
                print('___')
                print('found rotation: {}'.format(new_rotation))
                print('diff in rotation: {}'.format(diff_rotation))
                print('old pc_position: {}'.format(pc_position))
                print('new pc_position: {}'.format(np.array(source.get_center())))
                print('diff in pc_position: {}'.format(np.array(source.get_center()-pc_position)))
                print('___')
                pc_position = np.array(pc_utils.get_my_source_center(source))

            remember_pos_and_rot.append({
                'old_rotation': old_rotation,
                'pc_position': pc_position,
                'pc_rotation': pc_rotation
            })

        samples = list(os.listdir(data_path))
        samples = [d[:-10] for d in samples if '.color.png' in d]
        for id in samples:
            with open(os.path.join(data_path, '{}.meta.json'.format(id))) as f:
                meta = json.load(f)
                intr = meta.get('intr')
                robotEndEff2Cam = np.array(meta.get('hand_eye_calibration')).reshape((4, 4))
                robot2endEff_tf = np.array(meta.get('robot2endEff_tf')).reshape((4, 4))
                object_pose = np.array(meta.get('object_pose')).reshape(4, 4)[:3, :3]

            if d == 'extra':
                for rembered in remember_pos_and_rot:
                    object_rotation = np.rad2deg(transforms3d.euler.mat2euler(object_pose))
                    if np.array_equal(object_rotation, rembered['old_rotation']):
                        pc_position = rembered['pc_position']
                        pc_rotation = rembered['pc_rotation']
                        break

            # get the transformation from the camera to the object
            robot2object = np.identity(4)
            robot2object[:3, :3] = pc_rotation
            robot2object[:3, 3] = pc_position

            cam2robot = np.dot(np.linalg.inv(robotEndEff2Cam), np.linalg.inv(robot2endEff_tf))
            cam2object = np.dot(cam2robot, robot2object)

            object_position = cam2object[:3, 3]
            object_rotation = cam2object[:3, :3]

            pose_label = {'position': list(object_position),
                          'rotation': list(object_rotation.flatten()),
                          'cls_name': object_name,
                          'cam2robot': list(cam2robot.flatten()),
                          'robot2object': list(robot2object.flatten())}

            with open(os.path.join(label_path, '{}.meta.json'.format(id)), 'w')as f:
                json.dump(pose_label, f)

            if view_label:
                # paint the image
                with open(os.path.join(data_path, '{}.color.png'.format(id)), 'rb') as f:
                    image = np.array(Image.open(f).convert('RGB'))

                point_cloud = o3d.io.read_point_cloud(pco_path)
                point_cloud.transform(cam2object)
                image = pc_utils.pointcloud2image(image, point_cloud, 3, intr)
                plt.imshow(image)
                plt.show()


def create_labels(object_name, root, reference_point=np.array([]), plot=False, hsv=False, both=True):
    object_path = os.path.join(root, 'data_generation/data', object_name)
    dirs = os.listdir(object_path)
    background_path = os.path.join(object_path, 'background')
    n = int(len(os.listdir(background_path))/3)

    try:
        i = dirs.index('background')
        del dirs[i]
    except:
        raise ValueError('background does not exist in object_path: {}'.format(object_path))

    try:
        i = dirs.index('extra')
        del dirs[i]
    except:
        pass

    if len(dirs) < 1:
        raise ValueError('no foreground')


    ns = n*len(dirs)
    counter = 0
    for d in dirs:
        foreground_path = os.path.join(object_path, d)
        save_dir = os.path.join(root, 'label_generator/data', object_name, d)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)


        indexes = list(range(n))
        #indexes = [32,33]

        for idx in indexes:
            counter += 1
            print('number = {}/{}'.format(counter, ns))  # prints the progress in the terminal
            # load for the given index the background, object frame and ground truth
            with open(os.path.join(background_path, '{:06d}.color.png'.format(idx)), 'rb') as f:
                background_color_frame = np.array(Image.open(f).convert('RGB'))

            with open(os.path.join(foreground_path, '{:06d}.color.png'.format(idx)), 'rb') as f:
                foreground_color_frame = np.array(Image.open(f).convert('RGB'))

            with open(os.path.join(background_path, '{:06d}.depth.png'.format(idx)), 'rb') as f:
                background_depth_frame = np.array(Image.open(f), dtype=np.float)

            with open(os.path.join(foreground_path, '{:06d}.depth.png'.format(idx)), 'rb') as f:
                foreground_depth_frame = np.array(Image.open(f), dtype=np.float)

            if reference_point != np.array([]):
                with open(os.path.join(foreground_path, '{:06d}.meta.json'.format(idx))) as f:

                    meta = json.load(f)
                    robot2endEff_tf = np.array(meta.get('robot2endEff_tf')).reshape(4,4)
                    hand_eye_calibration = np.array(meta.get('hand_eye_calibration')).reshape(4, 4)
                    intr = meta['intr']

                    robot2cam = np.dot(robot2endEff_tf, hand_eye_calibration)
                    pos = robot2cam[:3, 3]
                    dist = np.linalg.norm(reference_point-pos)
            else:
                dist = None

            #if idx == 50:
            #    plot = True
            #else:
            #    plot = False
            label = createLabel_RGBD(background_color_frame,
                                     foreground_color_frame,
                                     background_depth_frame,
                                     foreground_depth_frame,
                                     plot=plot,
                                     threshold=30,
                                     hsv=hsv,
                                     both=both,
                                     intr=intr,
                                     close=6,
                                     open=6,
                                     measure_dist=dist,
                                     remove_one_std=True
                                     )

            if plot:
                plt.show()

            label = Image.fromarray(label)
            label.save(os.path.join(save_dir, '{:06d}.gen.label.png'.format(idx)))


if __name__ == '__main__':

    with open('./../hand_eye_calibration/data/handEye3_tf.json') as f:
        robotEndEff2Cam = json.load(f).get('tf')
        robotEndEff2Cam = np.array(robotEndEff2Cam).reshape((4, 4))

    n_viewpoints = 30
    min_friends = 30
    min_dist = 5
    nb_neighbors = 30
    n_points = 2000
    threshold = 20
    voxel_size = 2
    voxel_size_out = 5

    #n_viewpoints = 10
    #min_friends = 20
    #min_dist = 5
    #nb_neighbors = 20
    #n_points = 1000
    #threshold = 100
    #voxel_size = 1
    #voxel_size_out = 5

    l_arrow = 75
    voxel_size_selection = 100
    global_regression = False
    icp_point2point = True
    icp_point2plane = False
    both_half_way = False
    plot = False

    object_name = 'bluedude3'
    #print('getting labels')
    #create_labels(object_name)


    print('getting point cloud')
    save_dir = './../pc_reconstruction/data'
    load_point_cloud(object_name,
                     robotEndEff2Cam,
                     save_dir,
                     n_viewpoints=n_viewpoints,
                     min_friends=min_friends,
                     min_dist=min_dist,
                     nb_neighbors=nb_neighbors,
                     n_points=n_points,
                     threshold=threshold,
                     voxel_size=voxel_size,
                     voxel_size_out=voxel_size_out,
                     l_arrow=l_arrow,
                     voxel_size_selection=voxel_size_selection,
                     global_regression=global_regression,
                     icp_point2point=icp_point2point,
                     icp_point2plane=icp_point2plane,
                     both_half_way=both_half_way,
                     plot=plot)


    print('getting pose label')
    create_pose_label(object_name,
                      n_points,
                      min_friends,
                      min_dist,
                      nb_neighbors,
                      robotEndEff2Cam,
                      global_regression,
                      icp_point2point,
                      icp_point2plane,
                      plot=False,
                      view_label=False)


