import matplotlib.pyplot as plt
import json
import os
import numpy as np
from PIL import Image
import pyrealsense2 as rs
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import open3d as o3d
import pc_reconstruction.open3d_utils as utils
import copy



def load_robot_path_pos(robot_path):
    path = os.path.join('./../robot_controller/robot_path', robot_path)
    with open(path) as f:
        data = json.load(f)

    x = []
    y = []
    z = []
    robot_home = [data['cart_pose'][0]['x']*1000, data['cart_pose'][0]['y']*1000, data['cart_pose'][0]['z']*1000]
    for i, pose in enumerate(data['cart_pose']):
        if data['via_points'][i] == '0':
            x.append(pose['x']*1000)
            y.append(pose['y']*1000)
            z.append(pose['z']*1000)

    return [x, y, z], robot_home


def plot_robot_viewpoints():
    robot_path = 'viewpointsPath.json'
    [x, y, z], robot_home = load_robot_path_pos(robot_path)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='r', marker='o')
    ax.scatter(robot_home[0],robot_home[1],robot_home[2],c='g', marker='^')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def get_view_distribution(data_path, d, n, n_viewpoints, plot=False, l_arrow=30,
                     reference_point=np.array([0,0,0])):
    points = []
    arrows = []
    for idx in range(n):
        with open(os.path.join(data_path, d, '{:06d}.meta.json'.format(idx))) as f:
            meta = json.load(f)

        robotEndEff2Cam = np.array(meta.get('hand_eye_calibration')).reshape(4, 4)
        intr = meta.get('intr')
        robot2endEff_tf = meta.get('robot2endEff_tf')
        robot2endEff_tf = np.array(robot2endEff_tf).reshape((4, 4))
        depth_intrinsic = rs.intrinsics()
        depth_intrinsic.width = intr.get('width')
        depth_intrinsic.height = intr.get('height')
        depth_intrinsic.ppx = intr.get('ppx')
        depth_intrinsic.ppy = intr.get('ppy')
        depth_intrinsic.fx = intr.get('fx')
        depth_intrinsic.fy = intr.get('fy')
        depth_intrinsic.model = rs.distortion.inverse_brown_conrady
        depth_intrinsic.coeffs = intr.get('coeffs')
        robot2Cam_ft = np.dot(robot2endEff_tf, robotEndEff2Cam)
        points.append(list(robot2Cam_ft[:3,3]))

        next_arrow = []
        for i in range(3):
            new_arrow = np.identity(4)
            new_arrow[i, 3] = l_arrow
            new_arrow = np.dot(robot2Cam_ft, new_arrow)
            next_arrow.append(new_arrow[:3, 3])
        arrows.append(next_arrow)

    points = np.array(points)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    voxel_size = np.inf
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i != j:
                dist = int(np.linalg.norm(p2-p1))
                if dist < voxel_size:
                    voxel_size = dist

            if voxel_size == 0:
                break
        if voxel_size == 0:
            break

    while True:
        c_point_cloud = copy.deepcopy(point_cloud)
        c_point_cloud = c_point_cloud.voxel_down_sample(voxel_size=voxel_size)
        l = len(np.array(c_point_cloud.points))
        if l == n_viewpoints:
            selected_points = np.array(c_point_cloud.points)
            break
        elif l < n_viewpoints:
            voxel_size -= 1
            c_point_cloud = copy.deepcopy(point_cloud)
            c_point_cloud = c_point_cloud.voxel_down_sample(voxel_size=voxel_size)
            selected_points = np.random.choice(np.arange(len(np.array(c_point_cloud.points))), replace=False, size=n_viewpoints)
            selected_points = np.array(c_point_cloud.points)[selected_points]
            break
        else:
            voxel_size += 1

    selection = []
    for i, p1 in enumerate(selected_points):
        min_dist = np.inf
        min_dist_index = 0
        for j, p2 in enumerate(points):
            dist = np.linalg.norm(p2-p1)
            if dist < min_dist:
                min_dist = dist
                min_dist_index = j
        selection.append(min_dist_index)

    points = points[selection]

    new_selection = []
    new_selection.append(np.argmin(np.linalg.norm(points, axis=1)))
    while len(new_selection) != n_viewpoints:
            min_dist = np.inf
            min_dist_index = 0
            for j, p in enumerate(points):
                if j not in new_selection:
                    dist = np.linalg.norm(p-points[new_selection[-1]])
                    if dist < min_dist:
                        min_dist = dist
                        min_dist_index = j
            new_selection.append(min_dist_index)

    selection = np.array(selection)
    new_selection = selection[new_selection]


    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        robot_home = [0.46051215031184256, -193.22331249713898, 1000.9851455688477]
        ax.scatter(robot_home[0], robot_home[1], robot_home[2], c='g', marker='^')
        ax.scatter(reference_point[0], reference_point[1], reference_point[2], c='b', marker='o')
        vXs = points[:, 0]
        vYs = points[:, 1]
        vZs = points[:, 2]
        ax.scatter(vXs, vYs, vZs, c='r', marker='x')
        arrows = np.array(arrows)
        arrows = arrows[selection]
        for i in range(len(arrows)):
            cax, cay, caz = arrows[i]
            cx = vXs[i]
            cy = vYs[i]
            cz = vZs[i]
            #a = Arrow3D([cx, cax[0]], [cy, cax[1]], [cz, cax[2]], color='r')
            #ax.add_artist(a)
            #a = Arrow3D([cx, cay[0]], [cy, cay[1]], [cz, cay[2]], color='b')
            #ax.add_artist(a)
            a = Arrow3D([cx, caz[0]], [cy, caz[1]], [cz, caz[2]], color='g')
            ax.add_artist(a)

        ax.set_xlim([-700, 500])
        ax.set_ylim([-1200, 0])
        ax.set_zlim([-200, 1000])
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        ax.set_zlabel('Z [mm]')
        plt.show()

    return new_selection






def load_point_cloud(object_name,
                     save_dir,
                     root,
                     reference_point=np.array([0,0,0]),
                     mode='gen',
                     n_viewpoints=10,
                     min_friends=10,
                     voxel_size=5,
                     voxel_size_out=10,
                     threshold=50,
                     min_dist=10,
                     nb_neighbors=5,
                     l_arrow=30,
                     global_regression=False,
                     icp_point2point=True,
                     icp_point2plane=True,
                     plot=False):

    object_label_path = os.path.join(root, 'label_generator/data', object_name)

    dirs = os.listdir(object_label_path)
    # remove extra labels if they exist
    try:
        i = dirs.index('extra')
        del dirs[i]
    except:
        pass

    if not dirs:
        raise ValueError('no labels obtained yet')

    data_path = os.path.join(root, 'data_generation/data', object_name)

    pcd_path = os.path.join(save_dir, object_name)
    if not os.path.exists(pcd_path):
        os.makedirs(pcd_path)

    counter = 0
    n_dirs = np.array(os.listdir(os.path.join(object_label_path, dirs[0])))
    n_dirs = [d for d in n_dirs if '.{}.label.png'.format(mode) in d]
    n = len(n_dirs)
    ns = len(dirs)*n_viewpoints

    point_clouds = []
    point_cloud = o3d.geometry.PointCloud()

    for d in dirs:
        idx_selection = get_view_distribution(data_path, d, n, n_viewpoints, plot=plot, l_arrow=l_arrow, reference_point=reference_point)
        label_path = os.path.join(object_label_path, d)
        first_surface = True

        for idx in idx_selection:
            counter += 1
            print('number = {}/{}, idx = {}'.format(counter, ns, idx))  # prints the progress in the terminal
            # load for the given index the background, object frame and ground truth

            with open(os.path.join(data_path, d, '{:06d}.meta.json'.format(idx))) as f:
                meta = json.load(f)

            intr = meta.get('intr')

            robotEndEff2Cam = np.array(meta.get('hand_eye_calibration')).reshape(4, 4)
            robot2endEff_tf = meta.get('robot2endEff_tf')
            robot2endEff_tf = np.array(robot2endEff_tf).reshape((4, 4))
            robot2Cam_ft = np.dot(robot2endEff_tf, robotEndEff2Cam)
            point_cloud_tf = meta.get('object_pose')
            point_cloud_tf = np.array(point_cloud_tf, dtype=np.float64).reshape(4, 4)[:3, :3]

            with open(os.path.join(data_path, d, '{:06d}.depth.png'.format(idx)), 'rb') as f:
                depth_frame = np.array(Image.open(f), dtype=np.float)

            with open(os.path.join(label_path, '{:06d}.{}.label.png'.format(idx, mode)), 'rb') as f:
                label = np.array(Image.open(f), dtype=np.uint8)

            if plot:
                with open(os.path.join(data_path, d, '{:06d}.color.png'.format(idx)), 'rb') as f:
                    color_frame = np.array(Image.open(f).convert('RGB'), dtype=np.float)

                plt.subplot(1, 3, 1)
                plt.imshow(np.array(color_frame,dtype=np.uint8))
                plt.subplot(1, 3, 2)
                plt.imshow(label)

                added = copy.deepcopy(color_frame)
                red = np.zeros(color_frame.shape)
                red[ :, :, 0] = 255
                added[label!=0] = added[label!=0]*0.7 + red[label!=0]*0.3
                added[added < 0] = 0
                added[added > 255] = 255
                added = np.array(added, dtype=np.uint8)
                plt.subplot(1, 3, 3)
                plt.imshow(added)
                plt.show()

            # get the surface pointcloud of the label and depthframe
            source = utils.get_surface(label,
                                       depth_frame,
                                       intr,
                                       robot2Cam_ft,
                                       min_friends,
                                       min_dist,
                                       nb_neighbors,
                                       voxel_size=voxel_size)
            #print('n points source', len(np.array(source.points)))
            if len(np.array(source.points)) == 0:
                continue

            if first_surface:
                first_surface = False
                # transformation from the source camera to the target camera
                point_cloud.points = copy.deepcopy(source.points)
                if plot:
                    o3d.visualization.draw_geometries([point_cloud])

            else:
                # convert source and target numpy pc to open3d pc
                target, source, init_tf = utils.icp_regression(point_cloud,
                                                               source,
                                                               voxel_size=voxel_size,
                                                               threshold=threshold,
                                                               global_regression=global_regression,
                                                               icp_point2point=icp_point2point,
                                                               icp_point2plane=icp_point2plane,
                                                               plot=plot)

                # transform source
                source = source.transform(init_tf)

                point_cloud.points = o3d.utility.Vector3dVector(
                    np.concatenate((np.array(source.points), np.array(target.points))))

                point_cloud = point_cloud.voxel_down_sample(voxel_size=voxel_size)

                if plot:
                    print('new pc')
                    o3d.visualization.draw_geometries([point_cloud])
            print('n points', len(np.array(point_cloud.points)))


        point_cloud.rotate(R=point_cloud_tf, center=True)

        #save point cloud
        print('save to: {}'.format(pcd_path))
        o3d.io.write_point_cloud(os.path.join(pcd_path, '{}.pcd'.format(d)), point_cloud)
        o3d.io.write_point_cloud(os.path.join(pcd_path, '{}.ply'.format(d)), point_cloud)

        # append to point cloud list and reset
        point_clouds.append(copy.deepcopy(point_cloud))
        point_cloud = o3d.geometry.PointCloud()

    point_cloud = utils.align_point_clouds(point_clouds,
                                           min_friends=min_friends,
                                           min_dist=min_dist,
                                           nb_neighbors=nb_neighbors,
                                           plot=plot,
                                           global_regression=global_regression,
                                           icp_point2point=icp_point2point,
                                           icp_point2plane=icp_point2plane,
                                           voxel_size=voxel_size,
                                           threshold=threshold)


    # save pc with coords out in the robot world
    o3d.io.write_point_cloud(os.path.join(pcd_path, '{}_out.pcd'.format(object_name)), point_cloud)
    o3d.io.write_point_cloud(os.path.join(pcd_path, '{}_out.ply'.format(object_name)), point_cloud)
    # move the pointcloud into the robot origin and save

    point_cloud_down = point_cloud.voxel_down_sample(voxel_size=voxel_size_out)
    # get points for ptl plot


    point_cloud_down.translate(translation=-utils.get_my_source_center(point_cloud_down))
    o3d.io.write_point_cloud(os.path.join(pcd_path, '{}.pcd'.format(object_name)), point_cloud_down)
    o3d.io.write_point_cloud(os.path.join(pcd_path, '{}.ply'.format(object_name)), point_cloud_down)

    # output for training with dense fusion
    point_cloud_big_out = copy.deepcopy(point_cloud)
    point_cloud_big_out.translate(translation=-utils.get_my_source_center(point_cloud_big_out))

    if plot:
        o3d.visualization.draw_geometries([point_cloud_down])



    while True:
        voxel_size += 0.1
        pc_store = point_cloud_big_out.voxel_down_sample(voxel_size=voxel_size)
        n_points = len(np.array(pc_store.points))
        if n_points < 1000:
            point_cloud_big_out = point_cloud_big_out.voxel_down_sample(voxel_size=voxel_size-0.1)
            break

    points = np.array(point_cloud_big_out.points)
    with open(os.path.join(pcd_path, '{}.xyz'.format(object_name)), 'w') as f:
        for item in points:
            f.write("%s\n" % item)

    return point_cloud

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

def plot_point_cloud(object_name):
    save_dir = './data'
    #with open('./../hand_eye_calibration/data/handEye3_tf.json') as f:
    #    robotEndEff2Cam = json.load(f).get('tf')
    #    robotEndEff2Cam = np.array(robotEndEff2Cam).reshape((4, 4))
    robotEndEff2Cam = None
    [Xs, Ys, Zs], [vXs, vYs, vZs], arrows = load_point_cloud(object_name,
                                                             robotEndEff2Cam,
                                                             save_dir,
                                                             n_viewpoints=10,
                                                             min_friends=20,
                                                             min_dist=5,
                                                             nb_neighbors=20,
                                                             n_points=1000,
                                                             threshold=10,
                                                             voxel_size=1,
                                                             voxel_size_out=5,
                                                             l_arrow=75,
                                                             voxel_size_selection=100,
                                                             global_regression=False,
                                                             icp_point2point=True,
                                                             icp_point2plane=False,
                                                             both_half_way=False,
                                                             plot=False,
                                                             matplot=True)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    robot_home = [0.46051215031184256, -193.22331249713898, 1000.9851455688477]
    ax.scatter(robot_home[0], robot_home[1], robot_home[2], c='g', marker='^')
    ax.scatter(vXs, vYs, vZs, c='r', marker='x')
    ax.scatter(Xs, Ys, Zs, c='b', marker='o')

    for i in range(len(arrows)):
        cax, cay, caz = arrows[i]
        cx = vXs[i]
        cy = vYs[i]
        cz = vZs[i]
        a = Arrow3D([cx, cax[0]], [cy, cax[1]], [cz, cax[2]], color='r')
        ax.add_artist(a)
        a = Arrow3D([cx, cay[0]], [cy, cay[1]], [cz, cay[2]], color='b')
        ax.add_artist(a)
        a = Arrow3D([cx, caz[0]], [cy, caz[1]], [cz, caz[2]], color='g')
        ax.add_artist(a)


    ax.set_xlim([-700, 500])
    ax.set_ylim([-1200, 0])
    ax.set_zlim([-200, 1000])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


if __name__ == '__main__':
    object_name = 'bluedude3'
    #plot_point_cloud(object_name, )


    save_dir = './data'
    #with open('./../hand_eye_calibration/data/handEye3_tf.json') as f:
    #    robotEndEff2Cam = json.load(f).get('tf')
    #    robotEndEff2Cam = np.array(robotEndEff2Cam).reshape((4, 4))
    robotEndEff2Cam = None
    point_cloud = load_point_cloud(object_name,
                                   robotEndEff2Cam,
                                   save_dir,
                                   n_viewpoints=30,
                                   min_friends=30,
                                   min_dist=5,
                                   nb_neighbors=30,
                                   n_points=3000,
                                   threshold=10,
                                   voxel_size=2,
                                   voxel_size_out=5,
                                   l_arrow=75,
                                   min_friends_pc=3,
                                   min_dist_pc=10,
                                   nb_neighbors_pc=2,
                                   voxel_size_selection=100,
                                   global_regression=False,
                                   icp_point2point=True,
                                   icp_point2plane=False,
                                   both_half_way=False,
                                   plot=False,
                                   matplot=False
                                   )
    o3d.visualization.draw_geometries([point_cloud])
