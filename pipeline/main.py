from pipeline.utils import *
import data_generation.getData as data_gen
import os
from pathlib import Path
import json
from background_subtraction.utils import get_mask_prediction
import label_generator.create_labels as label_gen
import label_generator.make_train_and_test_dataset as data_set_gen
import segmentation
import segmentation.utils as segmentation_utils
import torch
import DenseFusion.tools.train as pose_estimation
import time
import random
import cv2
from robot_controller.TestController import RobotController
import transforms3d
import pipeline.grasping_utils as grasp_utils


root = str(Path(__file__).resolve().parent.parent)


def acquire_new_data_from_object():
    # create Depth Cam
    DC = data_gen.DepthCam(fps=30, height=480, width=640)
    #DC = None

    robot_path = 'viewpointsPath3.json'

    turn_selection = ['no turns', 'Turn once 180°', 'Turn 3 x 90°']
    names = list(os.listdir(os.path.join(root, 'data_generation/data')))
    turns = None
    symmetric = False
    runs = []
    continue_at = 0

    while True:
        print('____________________________________________________________________')
        name = input('Enter name of the new object: ')
        if name in names:
            print('A object with the name "{}" does already exist. Please find a different name.'.format(name))
            continue_selection = ['True', 'False']
            while True:
                continue_sel = input('do you want to continue from a given run?')
                if continue_sel not in continue_selection:
                    print('input "{}" is not valid.'.format(symmetric))
                    continue
                elif continue_sel == 'True':
                    while True:
                        continue_at = input('continue at run: ')
                        try:
                            continue_at = int(continue_at)
                            if continue_at<0:
                                print('continue at "{}" can not be negative.'.format(continue_at))
                                continue
                            break
                        except:
                            print('input "{}" is not valid'.format(continue_at))
                            continue
                    break
                else:
                    continue_at = 0
                    break

        selection = input(
            'New Object Name is: "{}", type "r" to rename, "b" to return, or hit any other key to continue.'.format(
                name))
        if selection == 'r':
            continue
        elif selection == 'b':
            return print('Returning to Main Menu')
        turns = get_selection(turn_selection, 'Select if and how the object is Turned')
        if not turns:
            continue
        else:
            symmetric_selection = ['True', 'False']
            while True:

                symmetric = input('Is the object symmetric? Type "True" or "False":')
                if symmetric not in symmetric_selection:
                    print('input "{}" is not valid.'.format(symmetric))
                    continue
                else:
                    symmetric = bool(symmetric)
                    hand_eye_calibs = os.listdir(os.path.join(root, 'hand_eye_calibration', 'data'))

                    hand_eye_calibs = [cal for cal in hand_eye_calibs if '.json' in cal]
                    if 'meta.json' in hand_eye_calibs:
                        index = hand_eye_calibs.index('meta.json')
                        del hand_eye_calibs[index]

                    hand_eye_calibration = get_selection(hand_eye_calibs, 'Select the current hand eye calibration')
                    if hand_eye_calibration:
                        with open(os.path.join(root, 'hand_eye_calibration', 'data', hand_eye_calibration)) as f:
                            hand_eye_calibration = json.load(f).get('tf')
                        break



            break

    if turns == turn_selection[0]:
        runs = [['background',
             {'x': 0, 'y': 0, 'z': 0, 'a': 0, 'b': 0, 'c': 0},
             'Getting Background. Clear the table.'],
            ['foreground',
             {'x': 0, 'y': 0, 'z': 0, 'a': 0, 'b': 0, 'c': 0},
             'Set the object into the table center.']]
    elif turns == turn_selection[1]:
        runs = [['background',
                 {'x': 0, 'y': 0, 'z': 0, 'a': 0, 'b': 0, 'c': 0},
                 'Getting Background. Clear the table.'],
                ['foreground',
                 {'x': 0, 'y': 0, 'z': 0, 'a': 0, 'b': 0, 'c': 0},
                 'Set the object into the table center.'],
                ['foreground180',
                 {'x': 0, 'y': 0, 'z': 0, 'a': 0, 'b': 0, 'c': 180},
                 'Rotate the object "180°" around its vertical axis.']]
    elif turns == turn_selection[2]:
        runs = [['background',
                 {'x': 0, 'y': 0, 'z': 0, 'a': 0, 'b': 0, 'c': 0},
                 'Getting Background. Clear the table.'],
                ['foreground',
                 {'x': 0, 'y': 0, 'z': 0, 'a': 0, 'b': 0, 'c': 0},
                 'Set the object into the table center.'],
                ['foreground90',
                 {'x': 0, 'y': 0, 'z': 0, 'a': 0, 'b': 0, 'c': 90},
                 'Rotate the object "90°" clock wise around its vertical axis. Total offset from start =  90°'],
                ['foreground180',
                 {'x': 0, 'y': 0, 'z': 0, 'a': 0, 'b': 0, 'c': 180},
                 'Rotate the object "90°" clock wise around its vertical axis. Total offset from start =  180°'],
                ['foreground270',
                 {'x': 0, 'y': 0, 'z': 0, 'a': 0, 'b': 0, 'c': 270},
                 'Rotate the object "90°" clock wise around its vertical axis. Total offset from start = 270°']]


    for run in runs[continue_at:]:
        print('____________________________________________________________________')
        print('Current name is: "{}"'.format('{}/{}'.format(name, run[0])))
        print('Instructions: {}.'.format(run[2]))
        input('Hit any key to continue if the instructions have been followed and leave the robots field of view.')
        data_gen.get_data(DC, robot_path, name, run[0], run[1], symmetric, hand_eye_calibration)
    pass


def create_labels():
    names = sorted(os.listdir(os.path.join(Path(__file__).resolve().parent.parent, 'data_generation/data')))
    while True:
        print('____________________________________________________________________')
        object_names = get_selection(names, 'Select the objects for which labels will be created. \n '
                                            'you can select multiple objects by separating them with a "," '
                                            '(e.g. "1,2")', multi_selection=True)
        if not object_names:
            return print('Returning to Main Menu')

        if not isinstance(object_names, list):
            object_names = [object_names]

        selection = input(
            'The Selected objects are: "{}", type "r" to reselect, or hit any other key to continue.'.format(object_names))
        if selection == 'r':
            continue
        else:
            break

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
    mode = 'new_pred'

    plot = False
    get_segmentation_labels = False
    get_target_point_cloud = True
    get_target_pose_label = False
    reference_point = np.array([-31, -823, -23])

    elapsed_times = []
    for object_name in object_names:
        t_start = time.time()
        print('Object name: {}'.format(object_name))
        if get_segmentation_labels:
            print('getting labels')
            if mode == 'pred':
                get_mask_prediction(object_name, root, reference_point=reference_point, plot=plot, use_cuda=False)
            elif mode == 'gen':
                label_gen.create_labels(object_name, root, reference_point, plot=plot)
            else:
                print('mode not supported')
                return False

        if get_target_point_cloud:
            print('getting point cloud')
            save_dir = os.path.join(root, 'pc_reconstruction/data')
            label_gen.load_point_cloud(object_name,
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

        if get_target_pose_label:
            print('getting pose label')
            label_gen.create_pose_label(root,
                                        object_name,
                                        global_regression,
                                        icp_point2point,
                                        icp_point2plane,
                                        plot=plot,
                                        view_label=plot,
                                        with_extra=False)
        elapsed_time = time.time()-t_start
        print('elapsed time {} sec'.format(np.round(elapsed_time, 2)))
        elapsed_times.append(elapsed_time)
    print('Elapsed time for "{}" objects: {} sec, with a per object time of: {} sec'.format(
        len(object_names), np.round(np.sum(elapsed_times), 2), np.round(np.mean(elapsed_times), 2)))

def create_pose_data():
    seg_path = os.path.join(root, 'label_generator/data_sets/segmentation')
    seg_names = sorted(os.listdir(seg_path))
    if not seg_names:
        print('There are not segmentation datasets. Please create and train a segmentation dataset first.')
        return False

    while True:
        print('____________________________________________________________________')
        ds_name = get_selection(seg_names, 'Select which segmentation dataset is transformed.')
        if not ds_name:
            return False

        classes = []
        input_file = open(os.path.join(seg_path, ds_name, 'classes.txt'))
        while 1:
            input_line = input_file.readline()[:-1]
            if not input_line:
                break
            classes.append(input_line)

        extra_available = True
        for cls in classes:
            if not 'extra' in os.listdir(os.path.join(os.path.join(root, 'data_generation/data', cls))):
                extra_available = False
                break

        while True:
            print('_______________________')
            use_new_pred, move_on = get_True_or_False('Use trained model to recompute segmentation '
                                                      'mask gained by background subtraction', default=True)
            if not move_on:
                break

            if extra_available:
                print('_______________________')
                get_extra_labels, move_on = get_True_or_False('Get Extra Labels', default=True)
                if not move_on:
                    continue
            else:
                get_extra_labels = False
            reference_point = np.array([-31, -823, -23])
            return label_gen.create_pose_data(root,
                                              classes,
                                              ds_name,
                                              reference_point=reference_point,
                                              new_pred=use_new_pred,
                                              get_extra_labels=get_extra_labels,
                                              plot=False,
                                              use_cuda=True)


def create_dataset():

    while True:
        print('____________________________________________________________________')
        data_set_type = get_selection(['segmentation', 'pose_estimation'], 'Select the data set type')
        if not data_set_type:
            return print('Returning to Main Menu')

        data_set_path = os.path.join(root, 'label_generator/data_sets', data_set_type)
        if not os.path.exists(data_set_path):
            os.makedirs(data_set_path)
        names = os.listdir(data_set_path)
        if data_set_type == 'segmentation':
            while True:
                print('____________________________________________________________________')
                name = input('Enter name of the new data set: ')
                if name in names:
                    print('A data set with the name "{}" does already exist. Please find a different name.'.format(name))
                    continue

                selection = input(
                    'The new data set name is: "{}", type "r" to rename, "b" to return, or hit any other key to continue.'.format(
                        name))
                if selection == 'r':
                    continue
                elif selection == 'b':
                    break

                path = os.path.join(root, 'label_generator/data')
                objects = sorted(os.listdir(path))
                while True:
                    print('____________________________________________________________________')
                    object_names = get_selection(objects, 'Select objects to include into the new dataset. '
                                                          '\n Select multiple objects by separating them with a comma. (e.g. "1,2")',
                                                 multi_selection=True)
                    if not object_names:
                        break

                    if isinstance(object_names, str):
                        object_names = [object_names]

                    data_set_gen.make_train_and_test_dataset(object_names, data_set_type, name)
                    print('____________________________________________________________________')
                    print('Created new "{}" data set "{}", with "{}" objects: '.format(data_set_type, name, len(object_names)))
                    for i, object_name in enumerate(object_names):
                        print('{}   : {}'.format(i+1, object_name))
                    return print('Returning to Main Menu')

        else:
            seg_path = os.path.join(root, 'label_generator/data_sets/segmentation')
            seg_names = sorted(os.listdir(seg_path))
            if not seg_names:
                print('There are not segmentation datasets. Please create and train a segmentation dataset first.')
                continue
            while True:
                print('____________________________________________________________________')
                ds_name = get_selection(seg_names, 'Select which segmentation dataset is transformed.')
                if not ds_name:
                    break

                classes = []
                input_file = open(os.path.join(seg_path, ds_name, 'classes.txt'))
                while 1:
                    input_line = input_file.readline()[:-1]
                    if not input_line:
                        break

                    classes.append(input_line)

                extra_available = True
                for cls in classes:
                    if not 'extra' in os.listdir(os.path.join(os.path.join(root, 'data_generation/data', cls))):
                        extra_available = False
                        break

                while True:
                    print('_______________________')
                    use_new_pred, move_on = get_True_or_False('Use masks generated by the segmentaiton model?', default=True)
                    if not move_on:
                        break


                    if extra_available:
                        print('_______________________')
                        get_extra_labels, move_on = get_True_or_False('Use Extra Labels', default=True)
                        if not move_on:
                            continue
                    else:
                        get_extra_labels = False


                    data_set_gen.make_train_and_test_dataset(classes,
                                                             data_set_type,
                                                             ds_name,
                                                             p_test=0.2,
                                                             use_extra_data=get_extra_labels)
                    print('____________________________________________________________________')
                    print('Created new "{}" data set "{}", with "{}" objects: '.format(data_set_type, ds_name,
                                                                                       len(classes)))
                    for i, object_name in enumerate(classes):
                        print('{}   : {}'.format(i + 1, object_name))
                    return print('Returning to Main Menu')


def train_segmentation():
    segmentation_data_sets_path = os.path.join(root, 'label_generator', 'data_sets', 'segmentation')
    if os.path.exists(segmentation_data_sets_path):
        segmentation_data_sets = sorted(os.listdir(segmentation_data_sets_path))
        if segmentation_data_sets:

            print('____________________________________________________________________')
            data_set = get_selection(segmentation_data_sets, 'Visualization Menu')
            if not data_set:
                return print('Returning to Main Menu')

            segmentation_config = {'name': 'Unet',
                                   'encoder_name': 'resnet34',
                                   'encoder_weights': 'imagenet',
                                   'activation': 'softmax'}
            training_config = {
                'epochs': 500,
                'batch_size': 4,
                'optimizer': 'Adam',
                'lr': 1e-4,
                'weight_decay': 0.0,
                'shuffle': True,
                'num_workers': 4,
                'momentum': 0.9,
                'dataset_name': data_set}

            print('Training {} model on the "{}" segmentation data set.'.format(segmentation_config['name'],
                                                                                data_set))
            segmentation.segmentation_training(training_config, segmentation_config)
            return print('Finished Training. Returning to Main Menu')
    return print('No segmentation data set available')


def train_pose_estimation():
    pose_estimation_data_sets_path = os.path.join(root, 'label_generator', 'data_sets', 'pose_estimation')
    if os.path.exists(pose_estimation_data_sets_path):
        pose_estimation_data_sets = sorted(os.listdir(pose_estimation_data_sets_path))
        if pose_estimation_data_sets:
            print('____________________________________________________________________')
            data_set = get_selection(pose_estimation_data_sets, 'Visualization Menu')
            if not data_set:
                return print('Returning to Main Menu')
            pose_estimation.main(data_set, root, p_viewpoints=1.0, p_extra_data=0.0, label_mode='new_pred',
                                 show_sample=False)
            return print('Finished Training. Returning to Main Menu')


color_dict = {
    'Angle': {'tag': 'red', 'value': [255, 0, 0]},
    'Assemblewall': {'tag': 'lime', 'value': [0, 255, 0]},
    'CameraStand': {'tag': 'blue', 'value': [0, 0, 255]},
    'Cylinder': {'tag': 'yellow', 'value': [255, 255, 0]},
    'Disk': {'tag': 'Cyan', 'value': [0, 255, 255]},
    'Edge': {'tag': 'Magenta', 'value': [255, 0, 255]},
    'Joint': {'tag': 'Maroon', 'value': [128, 0, 0]},
    'Motor': {'tag': 'Olive', 'value': [128, 128, 0]},
    'Plug': {'tag': 'Green', 'value': [0, 128, 0]},
    'Pole': {'tag': 'Purple', 'value': [128, 0, 128]},
    'Screw': {'tag': 'Teal', 'value': [0, 128, 128]},
    'Tube': {'tag': 'Navy', 'value': [0, 128, 0]}
}

def run_live_prediction():
    pose_estimation_data_sets_path = os.path.join(root, 'label_generator', 'data_sets', 'pose_estimation')
    if os.path.exists(pose_estimation_data_sets_path):
        pose_estimation_data_sets = sorted(os.listdir(pose_estimation_data_sets_path))
        if pose_estimation_data_sets:
            while True:
                print('____________________________________________________________________')
                data_set_name = get_selection(pose_estimation_data_sets, 'Visualization Menu')
                if not data_set_name:
                    return print('Returning to Main Menu')

                while True:
                    hand_eye_calibs = os.listdir(os.path.join(root, 'hand_eye_calibration', 'data'))

                    hand_eye_calibs = [cal for cal in hand_eye_calibs if '.json' in cal]
                    if 'meta.json' in hand_eye_calibs:
                        index = hand_eye_calibs.index('meta.json')
                        del hand_eye_calibs[index]

                    hand_eye_calibration = get_selection(hand_eye_calibs, 'Select the current hand eye calibration')
                    if not hand_eye_calibration:
                        break

                    with open(os.path.join(root, 'hand_eye_calibration', 'data', hand_eye_calibration)) as f:
                        end2cam = np.array(json.load(f).get('tf')).reshape((4,4))

                    pred_types = ['Camera Stream', 'Sample from test set']
                    while True:
                        print('____________________________________________________________________')
                        pred_type = get_selection(pred_types, 'What to predict')
                        if not pred_type:
                            break

                        segmentor, estimator, refiner, classes, to_tensor, normalize, point_clouds, device, cuda = get_prediction_models(
                            root, data_set_name)
                        data_path = os.path.join(root, 'data_generation', 'data')

                        data = []
                        input_file = open(os.path.join(pose_estimation_data_sets_path, data_set_name, 'test_data_list.txt'))
                        while 1:
                            input_line = input_file.readline()[:-1]
                            if not input_line:
                                break

                            data.append(input_line)
                        input_file.close()

                        print('____________________________________________________________________')
                        if pred_type == pred_types[0]:
                            DC = data_gen.DepthCam(fps=30, height=480, width=640)
                            controller = RobotController()
                            intr = DC.get_intrinsics()
                            meta = {'intr': {
                                'width': intr.width,
                                'height': intr.height,
                                'ppx': intr.ppx,
                                'ppy': intr.ppy,
                                'fx': intr.fx,
                                'fy': intr.fy,
                                'coeffs': intr.coeffs
                            }, 'depth_scale': DC.get_depth_scale()}

                            while True:
                                cam_data = DC.get_frames()
                                prediction = full_prediction(cam_data['image'], cam_data['depth'], meta, segmentor, estimator, refiner,  to_tensor, normalize, device,
                                                             cuda, color_dict, class_names=classes, point_clouds=point_clouds, plot=False,
                                                             color_prediction=True, bbox=True, put_text=True)
                                prediction = get_robot2object(prediction, controller, end2cam)
                                cv2.imshow("Pose Estimation", cv2.cvtColor(np.hstack((prediction['segmented_prediction'],
                                                                                      prediction['pose_prediction'])
                                                                                     ), cv2.COLOR_RGB2BGR))
                                if cv2.waitKey(1) == 27:
                                    break
                                #print(prediction['predictions'])

                                print('fps', 1/prediction['elapsed_times']['total'])

                            print('not implemented')
                            continue#
                        if pred_type == pred_types[1]:

                            while True:
                                sample_path = os.path.join(data_path, random.sample(data, 1)[0])
                                with open(os.path.join(sample_path + '.color.png'), 'rb') as f:
                                    image = Image.open(f).convert('RGB')

                                with open(os.path.join(sample_path+'.depth.png'), 'rb') as f:
                                    depth = Image.open(f)
                                    depth = np.array(depth)

                                with open(os.path.join(sample_path+'.meta.json')) as json_file:
                                    meta = json.load(json_file)

                                prediction = full_prediction(image, depth, meta, segmentor, estimator, refiner,  to_tensor, normalize, device,
                                                             cuda, color_dict, class_names=classes, point_clouds=point_clouds, plot=False,
                                                             color_prediction=False)

                                print(prediction['predictions'].keys())
                                print('fps', 1/prediction['elapsed_times']['total'])
                        else:
                            raise NotImplementedError


        else:
            return print('No pose estimation data set available')
    else:
        return print('No pose estimation data set available')


def visualise():
    v = {'Point Cloud': visualise_pointcloud,
         'Segmentation Mask': visualise_segmentation_maks,
         'Pose Label': visualise_pose_label}
    while True:
        print('____________________________________________________________________')
        selection = get_selection(list(sorted(v.keys())), 'Visualization Menu', with_return=True)
        if not selection:
            print('Returning to Main Menu')
            break
        v[selection](root)

def create_extra_labels_with_segmentation_model():
    path = os.path.join(root, 'label_generator', 'data', 'data_sets', 'segmentation')
    if not os.path.exists(path):
        return print('No data set available')

    dirs = sorted(os.listdir(path))
    if not dirs:
        return print('No data set available')

    while True:
        print('____________________________________________________________________')
        data_set = get_selection(dirs, 'Select for which data set extra labels will be generated for the extra data')
        if not data_set:
            return print('Returning to Main Menu')
        segmentation_model_path = os.path.join(root, 'segmentation', 'trained_models', data_set)
        if not os.path.exists(segmentation_model_path):
            return print('A Segmentation Model for the "{}" segmentation data set has not been trained yet.'.format(data_set))

        trained_models = sorted(os.listdir(segmentation_model_path))
        if not trained_models:
            return print('No Segmentaion Model Trained for the "{}" segmentation data set.'.format(data_set))

        while True:
            if len(trained_models) < 1:
                model_name = trained_models[0]
                print('Selected the trained model "{}", since it is the only option.'.format(model_name))
            else:

                model_name = get_selection(trained_models, 'Select trained Model')
                if not model_name:
                    break

                model_cp = torch.load(os.path.join(segmentation_model_path, model_name))
                model = segmentation_utils.get_model(model_cp['name'], model_cp['segmentation_config'])
                model.load_state_dict(model_cp['state_dict'])
                del model_cp['state_dict']


def teach_grasping():
    pose_estimation_data_sets_path = os.path.join(root, 'label_generator', 'data_sets', 'pose_estimation')
    if os.path.exists(pose_estimation_data_sets_path):
        pose_estimation_data_sets = sorted(os.listdir(pose_estimation_data_sets_path))
        if pose_estimation_data_sets:
            while True:
                print('____________________________________________________________________')
                data_set_name = get_selection(pose_estimation_data_sets, 'Visualization Menu')
                if not data_set_name:
                    break

                while True:
                    hand_eye_calibs = os.listdir(os.path.join(root, 'hand_eye_calibration', 'data'))

                    hand_eye_calibs = [cal for cal in hand_eye_calibs if '.json' in cal]
                    if 'meta.json' in hand_eye_calibs:
                        index = hand_eye_calibs.index('meta.json')
                        del hand_eye_calibs[index]

                    hand_eye_calibration = get_selection(hand_eye_calibs, 'Select the current hand eye calibration')
                    if not hand_eye_calibration:
                        break

                    with open(os.path.join(root, 'hand_eye_calibration', 'data', hand_eye_calibration)) as f:
                        end2cam = np.array(json.load(f).get('tf')).reshape((4,4))

                    segmentor, estimator, refiner, classes, to_tensor, normalize, point_clouds, device, cuda = get_prediction_models(
                        root, data_set_name)

                    data = []
                    input_file = open(os.path.join(pose_estimation_data_sets_path, data_set_name, 'test_data_list.txt'))
                    while 1:
                        input_line = input_file.readline()[:-1]
                        if not input_line:
                            break

                        data.append(input_line)
                    input_file.close()

                    print('Get camera and robot controller')
                    DC = data_gen.DepthCam(fps=30, height=480, width=640)
                    #print('got camera')
                    controller = RobotController()
                    #print('got robot controller')
                    intr = DC.get_intrinsics()
                    meta = {'intr': {
                        'width': intr.width,
                        'height': intr.height,
                        'ppx': intr.ppx,
                        'ppy': intr.ppy,
                        'fx': intr.fx,
                        'fy': intr.fy,
                        'coeffs': intr.coeffs
                    }, 'depth_scale': DC.get_depth_scale()}

                    print('got all required objects')
                    save_path = os.path.join(root, 'pipeline', 'data')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    meta_path = os.path.join(save_path, '{}_grasping_deltas.json'.format(data_set_name))

                    if os.path.exists(meta_path):
                        try:
                            with open(meta_path) as f:
                                save_meta = json.load(f)
                        except:
                            save_meta = {}
                    else:
                        save_meta = {}

                    print('meta: {}'.format(save_meta))
                    i = 0
                    while 0 <= i < len(classes):
                        cls = classes[i]
                        teach, move_on = get_True_or_False('Teach object the object "{}"?'.format(cls))
                        if not move_on:
                            i -= 1
                            continue
                        else:
                            i += 1

                        if not teach:
                            continue

                        if cls not in list(meta.keys()):
                            save_meta[cls] = {}

                        while True:
                            cam_data = DC.get_frames()
                            prediction = full_prediction(cam_data['image'], cam_data['depth'], meta, segmentor, estimator,
                                                         refiner, to_tensor, normalize, device,
                                                         cuda, color_dict, class_names=classes, point_clouds=point_clouds,
                                                         plot=False,
                                                         color_prediction=True, bbox=True, put_text=True)
                            prediction = get_robot2object(prediction, controller, end2cam)
                            stop = False
                            cv2.imshow("Pose Estimation", cv2.cvtColor(np.hstack((prediction['segmented_prediction'],
                                                                                  prediction['pose_prediction'])
                                                                                 ), cv2.COLOR_RGB2BGR))
                            if cv2.waitKey(1) == 27:
                                stop = True

                            preds = list(prediction['predictions'].keys())
                            if len(preds) > 1 or cls not in preds:
                                print('Include only the object "{}" into the scene. Found objects: "{}"'.format(cls, preds))
                                continue
                            elif stop:
                                pos = prediction['predictions'][cls]['position']
                                rot = transforms3d.euler.quat2euler(prediction['predictions'][cls]['rotation'])
                                c_rot = np.rad2deg(rot[2])
                                print('pos: {}, c rotation: {}'.format(pos, c_rot))
                                retake, move_on = get_True_or_False('Retake Pose?', default=False)
                                if not retake and move_on:
                                    print('Now teach the grasping Pose')
                                    print('Move the Robot to the grasping pose')
                                    while True:
                                        input('press any key too teach current position')
                                        teach_position, move_on = get_True_or_False('Teach position?', default=True)
                                        if not teach_position or not move_on:
                                            continue

                                        pose = controller.get_pose(return_mm=False)
                                        r = [pose['a'], pose['b'], pose['c']]
                                        anlge = np.linalg.norm(r)
                                        axis = r / anlge
                                        robot2end = np.zeros((4, 4))
                                        robot2end[3, 3] = 1
                                        robot2end[:3, :3] = transforms3d.axangles.axangle2mat(axis, anlge)
                                        robot2end[:3, 3] = [pose['x'], pose['y'], pose['z']]

                                        robot_rot = transforms3d.euler.mat2euler(robot2end[:3, :3])
                                        c_rot_robot = np.rad2deg(robot_rot[2])
                                        robot_pos = robot2end[:3, 3]

                                        pos_diff = robot_pos-pos
                                        rot_diff = c_rot_robot-c_rot
                                        print('robot pos: {}, robot c rotation: {}'.format(robot_pos, c_rot_robot))
                                        print('pos diff: {}, c rotation diff: {}'.format(pos_diff, rot_diff))

                                        save_meta[cls]['delta_x'] = float(pos_diff[0])
                                        save_meta[cls]['delta_y'] = float(pos_diff[1])
                                        save_meta[cls]['delta_z'] = float(pos_diff[2])
                                        save_meta[cls]['delta_c'] = float(rot_diff)

                                        with open(os.path.join(save_path,
                                                               '{}_grasping_deltas.json'.format(data_set_name)),
                                                  'w') as f:
                                            json.dump(save_meta, f)
                                        break
                                    break
                            else:
                                print('Press esc in the the Pose Estimation window to capture the pose.')

                    if i == len(classes):
                        return print('Finished teaching')
    return print('no pose estimation dataset')


def grasp():
    pose_estimation_data_sets_path = os.path.join(root, 'label_generator', 'data_sets', 'pose_estimation')
    if os.path.exists(pose_estimation_data_sets_path):
        pose_estimation_data_sets = sorted(os.listdir(pose_estimation_data_sets_path))
        if pose_estimation_data_sets:
            while True:
                print('____________________________________________________________________')
                data_set_name = get_selection(pose_estimation_data_sets, 'Visualization Menu')
                if not data_set_name:
                    break

                while True:
                    hand_eye_calibs = os.listdir(os.path.join(root, 'hand_eye_calibration', 'data'))

                    hand_eye_calibs = [cal for cal in hand_eye_calibs if '.json' in cal]
                    if 'meta.json' in hand_eye_calibs:
                        index = hand_eye_calibs.index('meta.json')
                        del hand_eye_calibs[index]

                    hand_eye_calibration = get_selection(hand_eye_calibs, 'Select the current hand eye calibration')
                    if not hand_eye_calibration:
                        break

                    with open(os.path.join(root, 'hand_eye_calibration', 'data', hand_eye_calibration)) as f:
                        end2cam = np.array(json.load(f).get('tf')).reshape((4,4))

                    segmentor, estimator, refiner, classes, to_tensor, normalize, point_clouds, device, cuda = get_prediction_models(
                        root, data_set_name)

                    data = []
                    input_file = open(os.path.join(pose_estimation_data_sets_path, data_set_name, 'test_data_list.txt'))
                    while 1:
                        input_line = input_file.readline()[:-1]
                        if not input_line:
                            break

                        data.append(input_line)
                    input_file.close()

                    DC = data_gen.DepthCam(fps=30, height=480, width=640)
                    controller = RobotController()
                    intr = DC.get_intrinsics()
                    meta = {'intr': {
                        'width': intr.width,
                        'height': intr.height,
                        'ppx': intr.ppx,
                        'ppy': intr.ppy,
                        'fx': intr.fx,
                        'fy': intr.fy,
                        'coeffs': intr.coeffs
                    }, 'depth_scale': DC.get_depth_scale()}

                    save_path = os.path.join(root, 'pipeline', 'data')
                    meta_path = os.path.join(save_path, '{}_grasping_deltas.json'.format(data_set_name))

                    try:
                        with open(meta_path) as f:
                            save_meta = json.load(f)
                    except:
                        return print('Grasping has not been teached for the objects of the dataset "{}"'.format(
                            data_set_name))
                    print('Taught objects: {}'.format(list(save_meta.keys())))

                    vel = 0.6
                    grasping_vel = 0.1
                    # get object position
                    if not grasp_utils.move_to_grasp_position(controller, vel=vel):
                        return print('could not move to grasp position')

                    selections = ['View Predictions', 'Get Predictions', 'Grasp_object']

                    predictions = {}
                    prediction_dict = {
                        'meta': meta,
                        'segmentor': segmentor,
                        'estimator': estimator,
                        'refiner': refiner,
                        'to_tensor':to_tensor,
                        'normalize': normalize,
                        'device': device,
                        'cuda': cuda,
                        'color_dict':color_dict,
                        'class_names': classes,
                        'point_clouds': point_clouds,
                        'plot': True,
                        'color_prediction': True,
                        'put_text': True
                    }
                    try:
                        while True:
                            print('______________________________________________________')
                            task = get_selection(selections, 'Select what to do')
                            if not task:
                                break

                            elif task == selections[0]:
                                print(predictions)
                            elif task == selections[1]:
                                success, predictions = grasp_utils.get_predictions(controller, DC, end2cam, prediction_dict, vel=vel)
                                if not success:
                                    print('cloud not get the predictions')
                                    continue
                                print('got predictions:')
                                print(predictions)

                            elif task == selections[2]:
                                if not predictions:
                                    print('No objects found')
                                    continue
                                # list objects
                                # select object
                                cls = get_selection(list(predictions.keys()), 'Select object to grasp')
                                if not cls:
                                    continue

                                pos = predictions[cls]['position']
                                print('ori pos: {}'.format(pos))
                                pos[0] += save_meta[cls]['delta_x']
                                pos[1] += save_meta[cls]['delta_y']
                                pos[2] += save_meta[cls]['delta_z']
                                print('moved pos: {}'.format(pos))

                                rotation = transforms3d.quaternions.quat2mat(predictions[cls]['rotation'])
                                eulers = np.rad2deg(transforms3d.euler.mat2euler(rotation))
                                print('eulers', eulers)
                                print('moved z ', eulers[2] + save_meta[cls]['delta_c'])
                                rotation = transforms3d.euler.euler2mat(0,
                                                                         0,
                                                                         np.deg2rad(
                                                                             eulers[2] + save_meta[cls]['delta_c']))


                                pose = controller.get_pose(return_mm=False)
                                r = [pose['a'], pose['b'], pose['c']]
                                anlge = np.linalg.norm(r)
                                axis = r / anlge
                                robot2end = transforms3d.axangles.axangle2mat(axis, anlge)

                                rotation = np.dot(rotation, robot2end)
                                vec, theta = transforms3d.axangles.mat2axangle(rotation)
                                rotation = vec * theta
                                print(rotation, vec, theta)

                                # approach object
                                print('approach object')
                                if not grasp_utils.approach_object(pos, rotation, controller, vel=vel):
                                    print('could not approach object')
                                    continue

                                # move down
                                print('move down')
                                if not grasp_utils.move_down(pos, rotation, controller, vel=grasping_vel):
                                    print('could not move down')
                                    continue

                                # grasp
                                print('grasp')
                                controller.close_gripper()
                                time.sleep(1)

                                # move up
                                print('move up')
                                if not grasp_utils.approach_object(pos, rotation, controller, vel=grasping_vel, moveType='l'):
                                    print('could not approach object')
                                    continue

                                # move down
                                if not grasp_utils.move_down(pos, rotation, controller, vel=grasping_vel):
                                    print('could not move down')
                                    continue

                                # release
                                controller.open_gripper()
                                time.sleep(1)

                                # move up
                                if not grasp_utils.approach_object(pos, rotation, controller, vel=grasping_vel, moveType='l'):
                                    print('could not approach object')
                                    continue

                                # move to start position
                                if not grasp_utils.return_2_grasp_position(controller, vel=vel):
                                    print('could not return to grasp position')
                                    continue
                                print('not implemented')
                            else:
                                break
                    except ValueError:
                        print(ValueError)


                    # move  home
                    if not grasp_utils.move_home(controller, vel=vel):
                        return print('could not move home')

                    return print('returning to main menu')



    return print('no pose estimation dataset')

def main():
    s = {'Acquire New Data from Object': acquire_new_data_from_object,
         'Create Labels': create_labels,
         'Create Pose labels': create_pose_data,
         'Create Data Set': create_dataset,
         'Train Segmentation Model': train_segmentation,
         'Train Pose Estimation Model': train_pose_estimation,
         'Run Live Prediction': run_live_prediction,
         'Visualise': visualise,
         'Teach Grasping': teach_grasping,
         'Grasp objects': grasp}

    while True:
        print('____________________________________________________________________')
        selection = get_selection(list(sorted(s.keys())), 'Main Menu', with_exit=True, with_return=False)
        if selection == 'exit':
            break
        else:
            s[selection]()


if __name__ == '__main__':
    main()
