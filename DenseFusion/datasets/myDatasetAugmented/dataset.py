import torch.utils.data as data
from PIL import Image
import os
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import numpy.ma as ma
import json
import matplotlib.pyplot as plt
import pc_reconstruction.open3d_utils as pc_utils
import random
import transforms3d

class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, noise_trans, refine, data_set_name, root, show_sample=False,
                 to_meter=True, label_mode='new_pred', p_extra_data=0.0, p_viewpoints=1.0):
        if mode == 'train':
            self.path = os.path.join(root, 'label_generator/data_sets/pose_estimation', data_set_name, 'train_data_list.txt')
            self.extra_path = os.path.join(root, 'label_generator/data_sets/pose_estimation', data_set_name, 'extra_train_data_list.txt')
        elif mode == 'test':
            self.path = os.path.join(root, 'label_generator/data_sets/pose_estimation', data_set_name, 'test_data_list.txt')

        self.mode = mode
        self.to_meter = to_meter
        self.num_pt = num_pt
        self.root = os.path.join(root, 'data_generation/data')
        self.label_root = os.path.join(root, 'label_generator/data')
        self.add_noise = add_noise
        self.noise_trans = noise_trans
        self.show_sample = show_sample
        self.label_mode = label_mode
        self.p_extra_data = p_extra_data
        self.p_viewpoints = p_viewpoints

        # load training data
        self.list = []
        input_file = open(self.path)
        while 1:
            input_line = input_file.readline()[:-1]
            if not input_line:
                break

            self.list.append(input_line)
        input_file.close()

        self.n_extra_samples = 0
        if mode == 'train':
            # select p viewpoints
            start_l = self.list[0].split('/')[1]
            viewpoint_ids = []
            for l in self.list:
                if l.split('/')[1] != start_l:
                    break
                else:
                    viewpoint_ids.append(l[-6:])
            viewpoint_ids = np.array(viewpoint_ids)
            np.random.shuffle(viewpoint_ids)
            viewpoints = viewpoint_ids[:int(len(viewpoint_ids)*self.p_viewpoints)]
            remove_list_ids = []
            for i, l in enumerate(self.list):
                if l[-6:] not in viewpoints:
                    remove_list_ids.append(i)
            self.list = np.delete(self.list, remove_list_ids, axis=0)

            # load extra data if wanted
            if self.p_extra_data >= 0:
                viewpoint_ids = []
                for v in viewpoints:
                    viewpoint_ids.append(int(v))

                self.extra_data = []
                input_file = open(self.extra_path)
                while 1:
                    input_line = input_file.readline()[:-1]
                    if not input_line:
                        break
                    with open('{0}/{1}.meta.json'.format(self.root, input_line)) as json_file:
                        extra_meta = json.load(json_file)
                        if extra_meta['view_point_id'] in viewpoint_ids:
                            self.extra_data.append(input_line)
                input_file.close()

                self.len_extra_data = len(self.extra_data)
                self.extra_data_ids = np.arange(self.len_extra_data)
                np.random.shuffle(self.extra_data_ids)
                self.extra_data_index = 0
                self.n_extra_samples = int(len(self.list) * p_extra_data)

        self.len_data = len(self.list)
        self.length = self.len_data + self.n_extra_samples
        #self.length = 16

        class_file = open(os.path.join(root, 'label_generator/data_sets/pose_estimation', data_set_name, 'classes.txt'))
        class_id = 0
        self.class_id_names = []
        self.cld = {}
        self.symmetry_obj_idx = []
        while 1:
            class_input = class_file.readline()[:-1]
            if not class_input:
                break
            self.class_id_names.append(class_input)
            meta_path = os.path.join(self.root, class_input)
            meta_path = os.path.join(meta_path, list(os.listdir(meta_path))[0])
            for md in list(os.listdir(meta_path)):
                if'.meta.json' in md:
                    with open(os.path.join(meta_path, md)) as metafile:
                        class_meta = json.load(metafile)
                        if bool(class_meta['symmetric']):
                            self.symmetry_obj_idx.append(class_id)
                    break
            input_file = open(os.path.join(root, 'pc_reconstruction/data/', class_input, '{}.xyz'.format(class_input)))
            self.cld[class_id] = []
            while 1:
                input_line = input_file.readline()[1:-2]
                if not input_line:
                    break

                input_line = input_line[:-1].split(' ')
                xyz = []
                for number in input_line:
                    if number != '':
                        if self.to_meter:
                            xyz.append(float(number)/1000)
                        else:
                            xyz.append(float(number))

                self.cld[class_id].append([xyz[0], xyz[1], xyz[2]])
            self.cld[class_id] = np.array(self.cld[class_id])
            input_file.close()
            
            class_id += 1
        self.num_classes = len(self.class_id_names)

        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.rotate_angles = None
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.num_pt_mesh = 1000
        self.refine = refine
        self.front_num = 2


    def __getitem__(self, index):

        # load data and meta data
        if index < self.len_data:
            if self.show_sample:
                print('view point sample')
            # load view point data
            img = Image.open('{0}/{1}.color.png'.format(self.root, self.list[index]))
            depth = Image.open('{0}/{1}.depth.png'.format(self.root, self.list[index]))
            with open('{0}/{1}.meta.json'.format(self.root, self.list[index])) as json_file:
                image_meta = json.load(json_file)

            label = Image.open('{0}/{1}.{2}.label.png'.format(self.label_root, self.list[index], self.label_mode))
            with open('{0}/{1}.meta.json'.format(self.label_root, self.list[index])) as json_file:
                meta = json.load(json_file)
        elif index < self.length:
            if self.show_sample:
                print('extra sample')
            # load extra data
            img = Image.open('{0}/{1}.color.png'.format(self.root, self.extra_data[self.extra_data_index]))
            depth = Image.open('{0}/{1}.depth.png'.format(self.root, self.extra_data[self.extra_data_index]))
            with open('{0}/{1}.meta.json'.format(self.root, self.extra_data[self.extra_data_index])) as json_file:
                image_meta = json.load(json_file)

            label = Image.open('{0}/{1}.new_pred.label.png'.format(self.label_root, self.extra_data[self.extra_data_index]))
            with open('{0}/{1}.meta.json'.format(self.label_root, self.extra_data[self.extra_data_index])) as json_file:
                meta = json.load(json_file)
            self.extra_data_index += 1
            if self.extra_data_index >= self.len_extra_data:
                self.extra_data_ids = np.arange(self.len_extra_data)
                np.random.shuffle(self.extra_data_ids)
                self.extra_data_index = 0
        else:
            raise ValueError


        # set the camera intrinsitcs
        cam_cx = image_meta['intr']['ppx']
        cam_cy = image_meta['intr']['ppy']
        cam_fx = image_meta['intr']['fx']
        cam_fy = image_meta['intr']['fy']

        # find the object class id
        obj = self.class_id_names.index(meta['cls_name'])

        # augment the data
        if self.add_noise:
            # change color
            img = self.trancolor(img)

            # rotate
            angle = random.uniform(-180, 180)
            augment_rotation = np.identity(4)
            augment_rotation[:3, :3] = transforms3d.euler.euler2mat(0, 0, np.deg2rad(angle))
            img = img.rotate(angle)
            label = label.rotate(angle)
            depth = depth.rotate(angle)

        # load tf from the camera to the robot
        cam2robot = np.array(meta['cam2robot']).reshape((4, 4))

        # if the image is rotated we need to rotate the camera before computing the target object pose
        if self.add_noise:
            cam2robot = np.dot(np.linalg.inv(augment_rotation), cam2robot)

        # compute object pose
        robot2object = np.array(meta['robot2object']).reshape((4, 4))
        cam2object = np.dot(cam2robot, robot2object)
        target_r = cam2object[:3, :3]
        target_t = cam2object[:3, 3]
        if self.to_meter:
            target_t = target_t/1000


        #convert input numpy
        img = np.array(img)
        label = np.array(label)
        depth = np.array(depth)


        # get masks and bounding box around the object
        mask_label = ma.getmaskarray(ma.masked_equal(label, 255))
        rmin, rmax, cmin, cmax = get_bbox(mask_label)
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask = mask_label * mask_depth


        # add some offset noise to the target position
        if self.add_noise:
            add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])

        # select some points on the object
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

        # select the choosen points
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        choose = np.array([choose])

        # compute the cartesian space position of each pixel and sample to point cloud
        cam_scale = image_meta['depth_scale']

        pt2 = depth_masked * cam_scale
        if not self.to_meter:
            pt2 = pt2*1000

        pt0 = (ymap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (xmap_masked - cam_cy) * pt2 / cam_fy

        cloud = np.concatenate((pt0, pt1, pt2), axis=1)

        if self.add_noise:
            cloud = np.add(cloud, add_t)

        # if needed delete some points from the target model points to have the same number all the time
        dellist = [j for j in range(0, len(self.cld[obj]))]
        dellist = random.sample(dellist, len(self.cld[obj]) - self.num_pt_mesh)
        model_points = np.delete(self.cld[obj], dellist, axis=0)

        # rotate and move the target into the image
        target = np.dot(model_points, target_r.T)
        if self.add_noise:
            target = np.add(target, target_t + add_t)
        else:
            target = np.add(target, target_t)

        # show the created sample for debug (if wanted)
        if self.show_sample:
            fig, axs = plt.subplots(1, 2, constrained_layout=True)
            fig.suptitle('{}, {}'.format(meta['cls_name'], obj), fontsize=16)
            plt.subplot(1,2,1)
            image_cloud = pc_utils.pointcloud2image(img.copy(), cloud, 3, image_meta['intr'])
            plt.imshow(image_cloud)
            plt.title('cloud')
            plt.subplot(1,2,2)
            plt.title('target')
            image_target = pc_utils.pointcloud2image(img.copy(), target, 3, image_meta['intr'])
            plt.imshow(image_target)
            plt.show()

        # crop and convert image into torch image
        img_masked = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))[:, rmin:rmax, cmin:cmax]

        # return sample as torch
        if self.mode == 'test':
            return torch.from_numpy(cloud.astype(np.float32)), \
                   torch.LongTensor(choose.astype(np.int32)), \
                   self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
                   torch.from_numpy(target.astype(np.float32)), \
                   torch.from_numpy(model_points.astype(np.float32)), \
                   torch.LongTensor([int(obj)]), \
                   image_meta['intr'],\
                   img.copy()

        else:
            return torch.from_numpy(cloud.astype(np.float32)), \
                   torch.LongTensor(choose.astype(np.int32)), \
                   self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
                   torch.from_numpy(target.astype(np.float32)), \
                   torch.from_numpy(model_points.astype(np.float32)), \
                   torch.LongTensor([int(obj)])


    def __len__(self):
        return self.length

    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        return self.num_pt_mesh

border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]
img_width = 480
img_length = 640

def get_bbox(label):
    rows = np.any(label, axis=1)
    cols = np.any(label, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    rmax += 1
    cmax += 1
    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


def crop_and_zoom(image,
                  label,
                  depth,
                  intr,
                  distance2object,
                  output_size=320,
                  bbox_increase=1.1,
                  to_small=0.8,
                  to_big=1.2,
                  max_l=480,
                  min_l=320,
                  size=(480, 640)):
    # get some variables
    extreme_points = get_extreme_points(np.array(label))
    h, w, c = get_size(extreme_points)
    h_ratio = float(h) / float(output_size)
    w_ratio = float(w) / float(output_size)
    h_w_ratio = h_ratio / w_ratio
    ls = [h, w]
    bigger = 0
    if w_ratio > h_ratio:
        bigger = 1

    # create bbox
    print(ls[bigger]*bbox_increase)
    bbox_size = ls[bigger]*bbox_increase
    if bbox_size < min_l:
        bbox_size = min_l
    elif bbox_size > max_l:
        bbox_size = max_l

    bbox_size = random.uniform(bbox_size, max_l)
    bbox = get_bbox_my(c, bbox_size)
    bbox = resize_bbox_to_max_zoom(bbox, max_l, min_l)
    bbox_h, bbox_w, bbox_c = get_size(bbox)  # height = width

    # adapt bbox
    if h_w_ratio <= to_big and h_w_ratio >= to_small:
        # case: square
        if bbox_h <= size[0] and bbox_w <= size[0]:
            # if the bbox is not to big, ensure that it is inside the image
            bbox = move_bbox_inside(bbox, size)
        else:
            # if the box is to big, slide random along the horizontal axis of the bbox and then create it as big as
            # possible and ensure that it is inside the image
            bbox_c[1] = int(bbox_c[1] - (w / 2)) + np.random.randint(0, w)
            bbox = get_bbox_my(bbox_c, size[0] - 2)
            bbox = move_bbox_inside(bbox, size)
    else:
        # case: rectangular
        # slide the bbox randomly along the bigger axis
        bbox_c[bigger] = int(bbox_c[bigger] - (ls[bigger] / 2)) + np.random.randint(0, ls[bigger])
        bbox = get_bbox_my(bbox_c, bbox_h)
        bbox_h, bbox_w, bbox_c = get_size(bbox)  # height = width

        if bbox_h <= size[0] and bbox_w <= size[0]:
            # if the bbox is not to big, ensure that it is inside the image
            bbox = move_bbox_inside(bbox, size)
        else:
            bbox = get_bbox_my(bbox_c, size[0] - 2)
            bbox = move_bbox_inside(bbox, size)

    # compute the x and y displacement
    _, _, [cy, cx] = get_size(extreme_points)

    dx = (cx - intr['ppx']) * distance2object / intr['fx']
    dy = (cy - intr['ppy']) * distance2object / intr['fy']
    intr['ppx'] = int(min_l/2)
    intr['ppy'] = int(min_l / 2)
    
    a1 = 240*distance2object/intr['fx']
    a2 = (bbox_size/2)*distance2object/intr['fx']
    h1 = np.sqrt(np.power(a1, 2) + np.power(distance2object, 2))
    alpha = np.arccos(distance2object/h1)
    h2 = a2/np.cos(alpha)
    d2 = np.sqrt(np.power(h2, 2) + np.power(a2, 2))
    dz = distance2object-d2
    delta_t = [dx, dy, dz]

    image = image.resize((output_size, output_size), box=(bbox[2], bbox[0], bbox[3], bbox[1]))
    label = label.resize((output_size, output_size), box=(bbox[2], bbox[0], bbox[3], bbox[1]))
    depth = depth.resize((output_size, output_size), box=(bbox[2], bbox[0], bbox[3], bbox[1]))



    return image, label, depth, delta_t, intr




def get_extreme_points(label):
    label_pos = np.where(label == 255)
    label_x = label_pos[0]
    label_y = label_pos[1]
    arg_max_x = np.argmax(label_x)
    arg_max_y = np.argmax(label_y)
    arg_min_x = np.argmin(label_x)
    arg_min_y = np.argmin(label_y)
    extreme_points = np.array(
        [label_x[arg_min_x], label_x[arg_max_x],
         label_y[arg_min_y], label_y[arg_max_y]])  # used for plotting [up, down, left, right]


    return extreme_points

def get_size(extreme_points):
    h = extreme_points[1] - extreme_points[0]
    w = extreme_points[3] - extreme_points[2]
    c = [extreme_points[0] + int(h / 2), extreme_points[2] + int(w / 2)]  # [height, width] (x,y)
    return h, w, c

def get_bbox_my(c, l):
    half = int(l / 2)
    bbox = [c[0] - half, c[0] + half, c[1] - half, c[1] + half]  # [up, down, left, right]
    return bbox

def move_bbox_inside(bbox, size):
    move = [0, 0]
    if bbox[0] < 0:
        move[0] = bbox[0]
    elif bbox[1] > size[0]:
        move[0] = bbox[1] - size[0]

    if bbox[2] < 0:
        move[1] = bbox[2]
    elif bbox[3] > size[1]:
        move[1] = bbox[3] - size[1]
    bbox = [bbox[0] - move[0], bbox[1] - move[0], bbox[2] - move[1], bbox[3] - move[1]]
    return bbox

def resize_bbox_to_max_zoom(bbox, max_l, min_l):
    bbox_h, bbox_w, bbox_c = get_size(bbox)
    if bbox_h > max_l:
        bbox = get_bbox_my(bbox_c, max_l)
    elif bbox_h < min_l:
        bbox = get_bbox_my(bbox_c, min_l)
    return bbox



