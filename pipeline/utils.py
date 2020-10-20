import numpy as np
import os
import open3d as o3d
from PIL import Image
import matplotlib.pyplot as plt
import copy
import json
import pc_reconstruction.open3d_utils as pc_utils
from threading import Thread
import torch
from label_generator.create_labels import get_default_model
from torchvision import transforms
from DenseFusion.lib.network import PoseNet, PoseRefineNet
from DenseFusion.datasets.myDatasetAugmented.dataset import get_bbox
from torch.nn import functional as F
import cv2
import numpy.ma as ma
from DenseFusion.tools.utils import my_estimator_prediction, my_refined_prediction, get_new_points
from DenseFusion.lib.transformations import quaternion_matrix
import time
import transforms3d


def get_selection(options_list,
                  text,
                  multi_selection=False,
                  select_from_range=False,
                  with_return=True,
                  with_exit=False):
    """Enables navigation in a list of options.

    Args:
        options_list (list): list of options
        text (string): text shown in TUI to guide selection

    Returns:
        string: some selection of the given list
    """
    selection_string = ''
    if select_from_range:
        selection_string += '{}-{}  : select from range\n'.format(options_list[0], options_list[-1])
    else:
        for i, a in enumerate(options_list):
            selection_string += '{}   : {}\n'.format(i + 1, a)

    if with_exit:
        selection_string += 'exit   : exit program\n'
        options_list.append('exit')


    while True:
        # List options and get user input
        if with_return:
            selection = input(text + ':\n0   : return\n' + selection_string)
        else:
            selection = input(text + ':\n' + selection_string)

        if not multi_selection:
            single = True
        else:
            try:
                selection = list(selection)
                if len(selection) == 1:
                    single = True
                    selection = selection[0]
                else:
                    single = False
                    selections = []
                    n = ''
                    for s in selection:
                        if s in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:
                            n += s
                        elif s == ',':
                            selections.append(int(n))
                            n = ''
                    if len(n)>0:
                        selections.append(int(n))

                    if 0 in selections:
                        return None
                    uniques, counts = np.unique(selections, return_counts=True)
                    if len(uniques) != np.sum(counts):
                        print('Found the value for {} multiple times. Try again.'.format([x for x in uniques[counts>1]]))
                        continue
                    out_of_scope = [s for s in selections if s < 0 or s > len(options_list)]
                    if out_of_scope:
                        print('Found value out of scope for {}'.format(out_of_scope))
                        continue
                    else:
                        return [options_list[s - 1] for s in selections]

            except ValueError:
                print('That is not a valid option. Try again.')
                continue

        if single:
            if with_exit:
                if selection == 'exit':
                    return selection
            try:
                # Convert user input to index of options list
                selection = int(selection)
            except ValueError:
                # This is raised when there is input that cannot be converted to an integer
                print('That is not a valid option. Try again.')
                continue

            # If user input is 0, return None
            if selection == 0:
                return None

            # If user input is valid, return chosen option
            elif 0 < selection <= len(options_list):
                name = options_list[selection - 1]
                return name

            # If user input is not valid, let the user try again
            else:
                print('That is not a valid option. Try again.')


def visualise_pointcloud(root):
    path = os.path.join(root, 'pc_reconstruction/data')
    objects = sorted(os.listdir(path))
    if len(objects) == 0:
        return print('No Point Cloud available')

    while True:
        print('____________________________________________________________________')
        selection = get_selection(objects, 'Select Object')
        if not selection:
            break
        selection_path = os.path.join(path, selection)
        pcds = sorted([pcd for pcd in os.listdir(selection_path) if pcd[-4:] == '.pcd'])
        if len(pcds) == 0:
            print('No point cloud available for the object "{}"'.find(selection))
            continue

        while True:
            print('____________________________________________________________________')
            pc_selection = get_selection(pcds, 'Select the point cloud to visualize')
            if not pc_selection:
                break

            pc_path = os.path.join(path, selection, pc_selection)
            pcd = o3d.io.read_point_cloud(pc_path)

            #pcd, out = pcd.remove_radius_outlier(nb_points=10, radius=5)
            points = np.array(pcd.points)
            length = np.round(np.max(points[:, 0]) - np.min(points[:, 0]), 2)
            width = np.round(np.max(points[:, 1]) - np.min(points[:, 1]), 2)
            height = np.round(np.max(points[:, 2]) - np.min(points[:, 2]), 2)

            print('name: {}         | x: {}, y: {}, z: {}'.format(selection, length, width, height))
            o3d.visualization.draw_geometries([pcd])

    return print('Returning to Visualization Menu')


def get_True_or_False(text, default=None):
    while True:
        if default is None:
            pt_string = input('{}: True or False, "b" to return: '.format(text))
        else:
            if default:
                pt_string = input('{}: True(default) or False, "b" to return: '.format(text))
            else:
                pt_string = input('{}: True or False(default), "b" to return: '.format(text))
        if pt_string == 'True':
            return True, True
        elif pt_string == 'False':
            return False, True
        elif pt_string == 'b':
            return None, False
        elif default is not None:
            return default, True
        elif pt_string == '':
            print('Please provide some input.')
            continue
        else:
            print('That is not a valid option write "True" or "False". Try again.')
            continue


class CancellationToken:
    def __init__(self):
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True

def stop_visualise(cancellationToken):
    input('press any key to stop visualization')
    cancellationToken.cancel()
    plt.close('all')
    print('stopping')


def visualise_segmentation_maks(root):
    path = os.path.join(root, 'label_generator/data')

    objects = sorted(os.listdir(path))
    if len(objects) == 0:
        return print('No Labels available')

    while True:
        print('____________________________________________________________________')
        selection = get_selection(objects, 'Select Object')
        if not selection:
            break
        object_path = os.path.join(path, selection)
        foregrounds = sorted(os.listdir(object_path))
        if not foregrounds:
            print('The object "{}" has no foregrounds'.format(selection))
            continue

        while True:
            print('____________________________________________________________________')
            foreground = get_selection(foregrounds, 'Select the foreground')
            if not foreground:
                break
            foreground_path = os.path.join(object_path, foreground)
            label_dirs = list(os.listdir(foreground_path))
            if not label_dirs:
                print('The foreground "{}" has no labels'.format(foreground))
                continue
            modes = []
            for d in label_dirs:
                if '.gen.label.png' in d and 'gen' not in modes:
                    modes.append('gen')
                if '.pred.label.png' in d and 'pred' not in modes:
                    modes.append('pred')
                if '.new_pred.label.png' in d and 'new_pred' not in modes:
                    modes.append('new_pred')

            if not modes:
                print('No mode was found')
                continue
            while True:
                if len(modes) == 1:
                    mode = modes[0]
                else:
                    print('____________________________________________________________________')
                    modes = sorted(modes)
                    mode = get_selection(modes, 'Select which label mode is shown')
                if not mode:
                    break

                l = len('.{}.label.png'.format(mode))
                dirs = np.array([d[:-l] for d in label_dirs if '.{}.label.png'.format(mode) in d])
                np.random.shuffle(dirs)

                data_path = os.path.join(root, 'data_generation/data')
                data_path = os.path.join(data_path, selection, foreground)

                cancellationToken = CancellationToken()
                thread = Thread(target=stop_visualise, args=[cancellationToken])
                thread.daemon = False
                thread.start()

                for d in dirs:
                    #index = d[:6]
                    with open(os.path.join(data_path, '{}.color.png'.format(d)), 'rb') as f:
                        image = np.array(Image.open(f).convert('RGB'), dtype=np.uint8)

                    with open(os.path.join(foreground_path, '{}.{}.label.png'.format(d, mode)), 'rb') as f:
                        label = np.array(Image.open(f), dtype=np.uint8)

                    plt.cla()
                    plt.subplot(1, 3, 1)
                    plt.imshow(image)
                    plt.title('Image')
                    plt.axis('off')

                    plt.subplot(1, 3, 2)
                    plt.imshow(label)
                    plt.title('Label')
                    plt.axis('off')

                    added = copy.deepcopy(image)
                    red = np.zeros(image.shape)
                    red[ :, :, 0] = 255
                    added[label != 0] = added[label != 0]*0.7 + red[label != 0]*0.3
                    added[added < 0] = 0
                    added[added > 255] = 255
                    added = np.array(added, dtype=np.uint8)
                    plt.subplot(1, 3, 3)
                    plt.imshow(added)
                    plt.title('Added')
                    plt.axis('off')
                    plt.show()

                    if cancellationToken.is_cancelled:
                        break

                if len(modes) == 1:
                    break

    return print('Returning to Visualization Menu')


def visualise_pose_label(root):
    path = os.path.join(root, 'label_generator/data')

    objects = sorted(os.listdir(path))
    if len(objects) == 0:
        return print('No Labels available')

    while True:
        print('____________________________________________________________________')
        selection = get_selection(objects, 'Select Object')
        if not selection:
            break
        object_path = os.path.join(path, selection)
        foregrounds = sorted(os.listdir(object_path))
        if not foregrounds:
            print('The object "{}" has no foregrounds'.format(selection))
            continue

        while True:
            print('____________________________________________________________________')
            foreground = get_selection(foregrounds, 'Select the foreground')
            if not foreground:
                break
            foreground_path = os.path.join(object_path, foreground)
            label_dirs = list(os.listdir(foreground_path))
            if not label_dirs:
                print('The foreground "{}" has no labels'.format(foreground))
                continue

            l = len('.meta.json')
            dirs = np.array([d[:-l] for d in label_dirs if '.meta.json' in d])
            np.random.shuffle(dirs)

            data_path = os.path.join(root, 'data_generation/data', selection, foreground)
            pc_path = os.path.join(root, 'pc_reconstruction/data', selection, '{}.ply'.format(selection))

            cancellationToken = CancellationToken()
            thread = Thread(target=stop_visualise, args=[cancellationToken])
            thread.daemon = False
            thread.start()

            for d in dirs:
                with open(os.path.join(data_path, '{}.color.png'.format(d)), 'rb') as f:
                    image = np.array(Image.open(f).convert('RGB'), dtype=np.uint8)

                with open(os.path.join(data_path, '{}.meta.json'.format(d))) as f:
                    image_meta = json.load(f)
                    intr = image_meta.get('intr')

                with open(os.path.join(foreground_path, '{}.meta.json'.format(d))) as f:
                    meta = json.load(f)

                    object_position = np.array(meta.get('position'))
                    object_rotation = np.array(meta.get('rotation')).reshape(3, 3)

                cam2object = np.identity(4)
                cam2object[:3, 3] = object_position
                cam2object[:3, :3] = object_rotation
                point_cloud = o3d.io.read_point_cloud(pc_path)
                point_cloud.transform(cam2object)
                added = pc_utils.pointcloud2image(copy.deepcopy(image), point_cloud, 3, intr)

                plt.cla()
                plt.subplot(1, 2, 1)
                plt.imshow(image)
                plt.title('Image')
                plt.axis('off')
                plt.subplot(1, 2, 2)
                plt.imshow(added)
                plt.title('Added')
                plt.axis('off')
                plt.show()

                if cancellationToken.is_cancelled:
                    break

    return print('Returning to Visualization Menu')


def get_robot2object(prediction, controller, end2cam):

    if len(prediction['predictions'].keys()) > 0:
        pose = controller.get_pose(return_mm=True)
        r = [pose['a'], pose['b'], pose['c']]
        anlge = np.linalg.norm(r)
        axis = r / anlge
        robot2end = np.zeros((4, 4))
        robot2end[3, 3] = 1
        robot2end[:3, :3] = transforms3d.axangles.axangle2mat(axis, anlge)
        robot2end[:3, 3] = [pose['x'], pose['y'], pose['z']]
        robot2cam = np.dot(robot2end, end2cam)
        for cls in prediction['predictions']:
            cam2obj = np.zeros((4, 4))
            cam2obj[3, 3] = 1
            try:
                cam2obj[:3, :3] = transforms3d.quaternions.quat2mat(prediction['predictions'][cls]['rotation'])
            except:
                print(prediction['predictions'][cls])
            cam2obj[:3, 3] = prediction['predictions'][cls]['position']*1000

            robot2obj = np.dot(robot2cam, cam2obj)
            prediction['predictions'][cls]['position'] = robot2obj[:3, 3]/1000
            prediction['predictions'][cls]['rotation'] = transforms3d.quaternions.mat2quat(robot2obj[:3, :3])
            print('found object "{}" at position "{}" with rotation "{}"'.format(cls,
                                                                                 prediction['predictions'][cls]['position'],
                                                                                 prediction['predictions'][cls]['rotation']))
    return prediction

def full_prediction(image, depth, meta, segmentor, estimator, refiner, to_tensor, normalize, device, cuda, color_dict,
                    class_names=None, point_clouds=None, plot=False, color_prediction=False, bbox=False, put_text=False):

    start_time = time.time()
    output_dict = {'predictions': {},
                   'elapsed_times': {}}

    if color_prediction:
        output_dict['segmented_prediction'] = np.array(copy.deepcopy(image), dtype=np.float)
        output_dict['pose_prediction'] = np.array(copy.deepcopy(image), dtype=np.float)

    # preprocesses input
    x = copy.deepcopy(image)
    x = to_tensor(x)
    x = normalize(x)

    x = x.to(device)
    x = x.unsqueeze(0)
    # get segmetation label
    pred = segmentor.predict(x)
    pred = F.softmax(pred, dim=1)
    if cuda:
        pred = pred.cpu()
    pred = pred[0]
    # crop and preproceses label to pass into the pose estimation
    pred_arg = torch.argmax(pred, dim=0).numpy()

    found_cls, counts = np.unique(pred_arg, return_counts=True)

    if 0 in found_cls:
        start = 1
    else:
        start = 0

    for i, cls in enumerate(found_cls[start:]):
        if counts[i+start] > 100:
            cls_pred_arg = copy.deepcopy(pred_arg)
            cls_pred_arg[cls_pred_arg != cls] = 0
            cls_pred = cls_pred_arg * pred[cls].numpy()

            ret, labels = cv2.connectedComponents(np.array(cls_pred_arg, dtype=np.uint8), connectivity=8)

            biggest = 1
            biggest_score = 0
            unique = np.unique(labels)
            if 0 in unique:
                start2 = 1
            else:
                start2 = 0
            for u in unique[start2:]:
                score = np.mean(cls_pred[labels == u])
                if score > biggest_score:
                    biggest_score = score
                    biggest = u

            cls_pred[labels != biggest] = 0
            cls_pred[cls_pred != 0] = 255
            cls_pred = np.array(cls_pred, dtype=np.uint8)

            output_dict['predictions'][class_names[cls-1]] = {'mask': cls_pred}

            if color_prediction:
                for c, color_value in enumerate(color_dict[class_names[cls-1]]['value']):
                    c_mask = np.zeros(cls_pred.shape)
                    c_mask[cls_pred != 0] = color_value
                    output_dict['segmented_prediction'][:, :, c][cls_pred != 0] = \
                        output_dict['segmented_prediction'][:, :, c][cls_pred != 0] * 0.7 + c_mask[cls_pred != 0] * 0.3

                if bbox:
                    bbox = get_bbox(cls_pred)
                    cv2.rectangle(output_dict['segmented_prediction'],
                                  (bbox[2], bbox[0]),
                                  (bbox[3], bbox[1]),
                                  color_dict[class_names[cls-1]]['value'],
                                  2)
                if put_text:
                    bbox = get_bbox(cls_pred)
                    try:
                        cv2.putText(output_dict['segmented_prediction'],
                                    'Segmentation',
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 0, 0),
                                    2,
                                    cv2.LINE_AA)

                        cv2.putText(output_dict['segmented_prediction'],
                                    class_names[cls - 1],
                                    (bbox[2] + 10, bbox[0] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    color_dict[class_names[cls-1]]['value'],
                                    2,
                                    cv2.LINE_AA)
                    except:
                        pass



    if color_prediction:
        output_dict['segmented_prediction'][output_dict['segmented_prediction'] < 0] = 0
        output_dict['segmented_prediction'][output_dict['segmented_prediction'] > 255] = 255
        output_dict['segmented_prediction'] = np.array(output_dict['segmented_prediction'], dtype=np.uint8)

    output_dict['elapsed_times']['segmentation'] = time.time()-start_time

    start_time_pose = time.time()
    xmap = np.array([[j for i in range(640)] for j in range(480)])
    ymap = np.array([[i for i in range(640)] for j in range(480)])
    num_points = 1000
    # estimate the pose
    for cls in output_dict['predictions']:
        #print('cls name', cls, class_names.index(cls))
        mask_label = ma.getmaskarray(ma.masked_equal(output_dict['predictions'][cls]['mask'], 255))
        rmin, rmax, cmin, cmax = get_bbox(mask_label)
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask = mask_label * mask_depth
        # select some points on the object
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) == 0:
            continue

        if len(choose) > num_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:num_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

        # select the choosen points
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        # compute the cartesian space position of each pixel and sample to point cloud

        pt2 = depth_masked * meta['depth_scale']
        pt0 = (ymap_masked - meta['intr']['ppx']) * pt2 / meta['intr']['fx']
        pt1 = (xmap_masked - meta['intr']['ppy']) * pt2 / meta['intr']['fy']

        points = np.concatenate((pt0, pt1, pt2), axis=1)
        #np_points = copy.deepcopy(points)

        points = torch.from_numpy(points.astype(np.float32)).unsqueeze(0).to(device)
        choose = torch.LongTensor(choose.astype(np.int32)).unsqueeze(0).to(device)

        img = np.transpose(np.array(image)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]
        img = normalize(torch.from_numpy(img.astype(np.float32))).unsqueeze(0).to(device)
        idx = torch.LongTensor([int(class_names.index(cls))]).unsqueeze(0).to(device)


        pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)
        new_points = get_new_points(pred_r, pred_t, pred_c, points)
        _, my_r, my_t = my_estimator_prediction(pred_r, pred_t, pred_c, num_points, 1, points)

        # refine the pose
        for ite in range(0, 2):
            pred_r, pred_t = refiner(new_points, emb, idx)
        _, my_r, my_t = my_refined_prediction(pred_r, pred_t, my_r, my_t)

        output_dict['predictions'][cls]['position'] = my_t
        output_dict['predictions'][cls]['rotation'] = my_r

        if color_prediction:
            my_r = quaternion_matrix(my_r)[:3, :3]
            np_pred = np.dot(point_clouds[class_names.index(cls)], my_r.T)
            np_pred = np.add(np_pred, my_t)

            output_dict['pose_prediction'] = pc_utils.pointcloud2image(output_dict['pose_prediction'],
                                                                       np_pred,
                                                                       3,
                                                                       meta['intr'],
                                                                       color=color_dict[cls]['value'])

            if put_text:
                try:
                    cv2.putText(output_dict['pose_prediction'],
                                'Pose Estimation',
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 0),
                                2,
                                cv2.LINE_AA)
                except:
                    pass

    if color_prediction:
        output_dict['pose_prediction'][output_dict['pose_prediction'] < 0] = 0
        output_dict['pose_prediction'][output_dict['pose_prediction'] > 255] = 255
        output_dict['pose_prediction'] = np.array(output_dict['pose_prediction'], dtype=np.uint8)

    output_dict['elapsed_times']['pose_estimation'] = time.time() - start_time_pose

    if plot and color_prediction:
        title = ''
        for cls in output_dict['predictions']:
            title += '{}: {}, '.format(color_dict[cls]['tag'], cls)
        fig, axs = plt.subplots(1, 2, constrained_layout=True)
        fig.suptitle(title)
        plt.subplot(1, 2, 1)
        plt.imshow(output_dict['segmented_prediction'])
        plt.axis('off')
        plt.title('segmented_prediction')
        plt.subplot(1, 2, 2)
        plt.imshow(output_dict['pose_prediction'])
        plt.axis('off')
        plt.title('pose_prediction')
        plt.show()

    del_keys = []
    for cls in output_dict['predictions'].keys():
        #print(output_dict['predictions'][cls].keys())
        if 'position' not in output_dict['predictions'][cls]:
            del_keys.append(cls)
        elif 'rotation' not in output_dict['predictions'][cls]:
            del_keys.append(cls)
        elif 'mask' not in output_dict['predictions'][cls]:
            del_keys.append(cls)

    for cls in del_keys:
        print('Deleting cls "{}"'.format(cls))
        del output_dict['predictions'][cls]

    output_dict['elapsed_times']['total'] = time.time() - start_time

    # return dict with found objects and their pose. also return the painted image

    return output_dict

def get_prediction_models(root, data_set_name):

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        cuda = True
    else:
        cuda = False
        device = torch.device('cpu')

    print(cuda, device)

    to_meter = True
    class_file = open(os.path.join(root, 'label_generator', 'data_sets', 'segmentation', data_set_name, 'classes.txt'))
    class_id = 0
    classes = []
    cld = {}
    while 1:
        class_input = class_file.readline()[:-1]
        if not class_input:
            break
        classes.append(class_input)
        meta_path = os.path.join(root, 'data_generation', 'data', class_input)
        meta_path = os.path.join(meta_path, list(os.listdir(meta_path))[0])

        input_file = open(os.path.join(root, 'pc_reconstruction/data/', class_input, '{}.xyz'.format(class_input)))
        cld[class_id] = []
        while 1:
            input_line = input_file.readline()[1:-2]
            if not input_line:
                break

            input_line = input_line[:-1].split(' ')
            xyz = []
            for number in input_line:
                if number != '':
                    if to_meter:
                        xyz.append(float(number) / 1000)
                    else:
                        xyz.append(float(number))

            cld[class_id].append([xyz[0], xyz[1], xyz[2]])
        cld[class_id] = np.array(cld[class_id])
        input_file.close()
        class_id += 1


    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean, std)

    print('create segmentor')
    segmentor = get_default_model(root, data_set_name, len(classes) + 1)
    segmentor.to(device)
    segmentor.eval()

    print('create estimator and refiner models')
    pose_path = os.path.join(root, 'DenseFusion', 'trained_models', data_set_name)
    num_points = 1000
    num_objects = len(classes)
    estimator = PoseNet(num_points=num_points, num_obj=num_objects)
    refiner = PoseRefineNet(num_points=num_points, num_obj=num_objects)
    loading_path = os.path.join(pose_path, 'pose_model.pth')
    pretrained_dict = torch.load(loading_path, map_location=torch.device('cpu'))
    estimator.load_state_dict(pretrained_dict)
    loading_path = os.path.join(pose_path, 'pose_refine_model.pth')
    pretrained_dict = torch.load(loading_path, map_location=torch.device('cpu'))
    refiner.load_state_dict(pretrained_dict)

    estimator.to(device)
    refiner.to(device)
    estimator.eval()
    refiner.eval()


    return segmentor, estimator, refiner, classes, to_tensor, normalize, cld, device, cuda






