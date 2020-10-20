import copy
import numpy as np
import open3d as o3d
from mathutils.geometry import intersect_line_line
from PIL import Image
import os
import json


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size):
    #print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    #print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    #print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    #print(":: RANSAC registration on downsampled point clouds.")
    #print("   Since the downsampling voxel size is %.3f," % voxel_size)
    #print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False), 4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result

def refine_registration(source, target, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    #print(":: Point-to-plane ICP registration is applied on original point")
    #print("   clouds to refine the alignment. This time we use a strict")
    #print("   distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    return result



def icp_regression(target,
                   source,
                   voxel_size=5,
                   threshold=100,
                   global_regression=False,
                   icp_point2point=True,
                   icp_point2plane=True,
                   plot=False):

    target, target_fpfh = preprocess_point_cloud(copy.deepcopy(target), voxel_size)
    source, source_fpfh = preprocess_point_cloud(source, voxel_size)
    init_tf = np.identity(4)

    criteria = o3d.registration.ICPConvergenceCriteria(relative_fitness=1e-2,
                                                       relative_rmse=1e-2,
                                                       max_iteration=100,)

    if plot:
        print('init', init_tf)
        draw_registration_result(source, target, init_tf)

    if global_regression:
        # global regression
        init_tf = execute_global_registration(source,
                                              target,
                                              source_fpfh,
                                              target_fpfh,
                                              voxel_size
                                              ).transformation
        if plot:
            print('global_regression', init_tf)
            draw_registration_result(source, target, init_tf)

    if icp_point2point:
        # iterative closest point PointToPoint regression
        init_tf = o3d.registration.registration_icp(source,
                                                    target,
                                                    threshold,
                                                    init_tf,
                                                    o3d.registration.TransformationEstimationPointToPoint(),
                                                    criteria
                                                    ).transformation
        if plot:
            print('point2point', init_tf)
            draw_registration_result(source, target, init_tf)

    if icp_point2plane:
        # iterative closest point PointToPlane regression
        init_tf = o3d.registration.registration_icp(source,
                                                    target,
                                                    threshold,
                                                    init_tf,
                                                    o3d.registration.TransformationEstimationPointToPlane(),
                                                    criteria
                                                    ).transformation
        if plot:
            print('point2plane', init_tf)
            draw_registration_result(source, target, init_tf)

    return target, source, init_tf


def align_point_clouds(point_clouds,
                       min_friends,
                       min_dist,
                       nb_neighbors,
                       plot=False,
                       global_regression=False,
                       icp_point2point=True,
                       icp_point2plane=False,
                       voxel_size=5,
                       threshold=50):

    target = point_clouds[0]
    for k, source in enumerate(point_clouds[1:]):
        t_center = np.array(target.get_center())
        s_center = np.array(source.get_center())
        diff = s_center-t_center # -800 -810 -10 + 20
        if diff[1] > -30:
            move = [0, -30-diff[1], 0]
            source.translate(translation=move)

        target, source, init_tf = icp_regression(target,
                                                 source,
                                                 voxel_size=voxel_size,
                                                 threshold=threshold,
                                                 global_regression=global_regression,
                                                 icp_point2point=True,
                                                 icp_point2plane=False,
                                                 plot=plot)
        # transform source
        source = source.transform(init_tf)
        target.points = o3d.utility.Vector3dVector(
            np.concatenate((np.array(source.points), np.array(target.points))))

        target = target.voxel_down_sample(voxel_size=voxel_size)

        # post process
        target, out = target.remove_radius_outlier(nb_points=min_friends,
                                                   radius=min_dist)
        dist = np.array(target.compute_mahalanobis_distance())
        std_ratio = np.std(dist)
        target, out = target.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                        std_ratio=std_ratio)

    return target


def get_surface(label, depth_frame, intr, robot2Cam_ft, min_friends, min_dist, nb_neighbors, voxel_size):
    points = []
    pos = []
    x, y = np.where(label != 0)
    for i in range(len(x)):
        pos.append([x[i], y[i]])
    pos = np.array(pos)

    for p in pos:
        p2 = depth_frame[p[0], p[1]]
        if p2 != 0:
            px = p[1]
            py = p[0]
            ppx = px - intr.get('ppx')
            ppy = py - intr.get('ppy')

            p0 = ppx * p2 / intr.get('fx')
            p1 = ppy * p2 / intr.get('fy')
            cam2obj = np.identity(4)
            cam2obj[0:3, 3] = [p0, p1, p2]
            robot2obj_tf = np.dot(robot2Cam_ft, cam2obj)
            points.append(robot2obj_tf[:3, 3])


    surface = o3d.geometry.PointCloud()
    surface.points = o3d.utility.Vector3dVector(np.array(points))

    surface = surface.voxel_down_sample(voxel_size=voxel_size)

    dist = np.abs(np.array(surface.compute_mahalanobis_distance()))
    std_ratio = np.abs(np.std(dist))
    #print(std_ratio, min_friends, min_dist)

    # post process
    surface, out = surface.remove_radius_outlier(nb_points=min_friends, radius=min_dist)
    #print('after remove radius outliners', len(np.array(surface.points)))
    dist = np.abs(np.array(surface.compute_mahalanobis_distance()))
    std_ratio = np.abs(np.std(dist))
    #print(std_ratio, nb_neighbors)
    surface, out = surface.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                                      std_ratio=std_ratio)
    #print('after remove stat outliners', len(np.array(surface.points)))
    return surface

def pixels2points(pixels, depth, intr):
    points = []
    for p in pixels:

        p2 = depth[p[0], p[1]]
        #print(p, p2)
        if p2 != 0:
            px = p[1]
            py = p[0]
            ppx = px - intr.get('ppx')
            ppy = py - intr.get('ppy')

            p0 = ppx * p2 / intr.get('fx')
            p1 = ppy * p2 / intr.get('fy')
            points.append([p0, p1, p2])

    return points

def points2pixel(points, intr):
    pixels = []
    for point in points:
        x, y, z = point
        p1 = (x / (z / intr.get('fx')))
        p1 = int(p1 + intr.get('ppx'))
        p0 = (y / (z / intr.get('fy')))
        p0 = int(p0 + intr.get('ppy'))
        pixels.append([p0, p1])

    return pixels


def pointcloud2image(image, point_cloud, point_size, intr, color=None):

    step = int((point_size - 1) / 2)

    mark = np.zeros((point_size, point_size, 3))
    if not color:
        mark[:, :, 0] = 255
    else:
        for i, c in enumerate(color):
            mark[:, :, i] = c

    if isinstance(point_cloud, np.ndarray):
        points = point_cloud
    else:
        points = np.array(point_cloud.points)
    pixels = points2pixel(points, intr)

    for x, y in pixels:
        try:
            image[x - step:x + step + 1, y - step:y + step + 1, :] = mark * 0.3 + image[x - step:x + step + 1, y - step:y + step + 1, :] * 0.7
        except:
            pass


    return image


def get_my_source_center(source):
    source_points = np.array(source.points)
    x_min = np.min(source_points[:, 0])
    x_max = np.max(source_points[:, 0])

    y_min = np.min(source_points[:, 1])
    y_max = np.max(source_points[:, 1])

    z_min = np.min(source_points[:, 2])
    z_max = np.max(source_points[:, 2])

    source_center = np.array(source.get_center())
    my_source_center = np.array([x_min + (x_max - x_min) / 2,
                                 y_min + (y_max - y_min) / 2,
                                 z_min + (z_max - z_min) / 2])

    #print('source_center: {}'.format(source_center))
    #print('my_source_center: {}'.format(my_source_center))
    #print('diff from source of: {}'.format(source_center-my_source_center))
    return my_source_center


def get_new_position(position_vectors, source):

    source_points = np.array(source.points)
    x_min = np.min(source_points[:, 0])
    x_max = np.max(source_points[:, 0])

    y_min = np.min(source_points[:, 1])
    y_max = np.max(source_points[:, 1])

    z_min = np.min(source_points[:, 2])
    z_max = np.max(source_points[:, 2])

    source_center = np.array(source.get_center())
    my_source_center = np.array([x_min + (x_max - x_min) / 2,
                                 y_min + (y_max - y_min) / 2,
                                 z_min + (z_max - z_min) / 2])

    shift2mycenter = np.subtract(source_center, my_source_center)

    print('my_source_center: {}'.format(my_source_center))
    print('diff from source of: {}'.format(shift2mycenter))

    points = []
    for i, line0 in enumerate(position_vectors[:-1]):
        for line1 in position_vectors[i+1:]:
            p0, p1 = intersect_line_line(line0[0], line0[1], line1[0], line1[1])
            p0 = np.array(p0)
            p1 = np.array(p1)
            center_point = p0 + (np.subtract(p1, p0)/2)
            points.append(center_point)
    points = np.array(points)
    position = np.mean(points, axis=0)
    position = position+shift2mycenter

    return position


def get_surface_positions(root,
                          object_name,
                          d,
                          min_friends,
                          min_dist,
                          nb_neighbors,
                          mode='gen',
                          voxel_size=5):

    object_label_path = os.path.join(root, 'label_generator/data', object_name)
    dirs = os.listdir(object_label_path)
    if not dirs:
        raise ValueError('no labels obtained yet')

    data_path = os.path.join(root, 'data_generation/data', object_name)

    n_dirs = np.array(os.listdir(os.path.join(object_label_path, dirs[0])))
    n_dirs = [d for d in n_dirs if '.{}.label.png'.format(mode) in d]

    n = len(n_dirs)
    label_path = os.path.join(object_label_path, d)
    positions = []
    for idx in range(n):
        # load for the given index the background, object frame and ground truth

        with open(os.path.join(data_path, d, '{:06d}.meta.json'.format(idx))) as f:
            meta = json.load(f)

        intr = meta.get('intr')
        robotEndEff2Cam = np.array(meta.get('hand_eye_calibration')).reshape((4, 4))
        robot2endEff_tf = np.array(meta.get('robot2endEff_tf')).reshape((4, 4))
        robot2Cam_ft = np.dot(robot2endEff_tf, robotEndEff2Cam)


        with open(os.path.join(data_path, d, '{:06d}.depth.png'.format(idx)), 'rb') as f:
            depth_frame = np.array(Image.open(f), dtype=np.float)

        with open(os.path.join(label_path, '{:06d}.{}.label.png'.format(idx, mode)), 'rb') as f:
            label = np.array(Image.open(f), dtype=np.uint8)

        # get the surface pointcloud of the label and depthframe
        source = get_surface(label,
                             depth_frame,
                             intr,
                             robot2Cam_ft,
                             min_friends,
                             min_dist,
                             nb_neighbors,
                             voxel_size=voxel_size)
        positions.append([list(source.get_center()), list(robot2Cam_ft[:3, 3])])
    return np.array(positions)

