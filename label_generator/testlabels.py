import numpy as np
import transforms3d
import json
import os
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt

object_name = 'bluedude3'
path = os.path.join('./data', object_name)
img_path = os.path.join('./../data_generation/data', object_name)
dirs = os.listdir(path)
n_dirs = np.array(os.listdir(os.path.join(path, dirs[0])))
n_dirs = [d for d in n_dirs if 'meta' not in d]
n = len(n_dirs)

pc_path = os.path.join('./../pc_reconstruction/data', object_name, '{}.ply'.format(object_name))

counter = 0
ns = n*len(dirs)
point_size = 5
mark = np.zeros((point_size, point_size, 3))
mark[:, :, 0] = 255



for d in dirs[1:]:
    print(d)
    label_path = os.path.join(path, d)
    foreground_path = os.path.join(img_path, d)
    for idx in range(n):
        counter += 1
        print('number = {}/{}'.format(counter, ns))  # prints the progress in the terminal

        with open(os.path.join(label_path, '{:06d}.meta.json'.format(idx))) as f:
            pose_meta = json.load(f)

        with open(os.path.join(foreground_path, '{:06d}.color.png'.format(idx)), 'rb') as f:
            image = np.array(Image.open(f).convert('RGB'))

        with open(os.path.join(foreground_path, '{:06d}.meta.json'.format(idx))) as f:
            meta = json.load(f)
            intr = meta.get('intr')



        rotation = transforms3d.quaternions.quat2mat(np.array(pose_meta.get('quaternions')))

        translation = np.array(pose_meta.get('position'))
        tf = np.identity(4)
        tf[:3, :3] = rotation
        tf[:3, 3] = translation

        #point_cloud.rotate(R=rotation, center=True)
        #point_cloud.translate(translation=translation, relative=True)
        point_cloud = o3d.io.read_point_cloud(pc_path)
        point_cloud.transform(tf)

        points = np.array(point_cloud.points)
        pixels = []
        for point in points:
            x, y, z = point
            p0 = (x/(z / intr.get('fx')))
            x = int(p0 + intr.get('ppx'))
            p1 = (y/(z / intr.get('fy')))
            y = int(p1 + intr.get('ppy'))
            pixels.append([y, x])
            step = int((point_size-1)/2)
            step = int((point_size - 1) / 2)
            image[y-step:y+step+1, x-step:x+step+1, :] = mark

        plt.imshow(image)
        plt.show()












