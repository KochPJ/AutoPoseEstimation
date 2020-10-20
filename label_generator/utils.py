import matplotlib.pyplot as plt
import cv2
import numpy as np
import copy


def contrast_stretching(one_channel_image):

    flat = np.ndarray.flatten(one_channel_image)
    min_i = np.min(flat)
    max_i = np.max(flat)
    min_0 = 0
    max_0 = 255

    flat = (flat-min_i) * ((max_0 - min_0) / (max_i - min_i) + min_0)
    one_channel_image = np.reshape(flat, one_channel_image.shape)

    return np.array(one_channel_image)


def connectedComponents(img):
    ret, labels = cv2.connectedComponents(np.array(img, dtype=np.uint8), connectivity=8)
    return labels


def smoothing(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size*kernel_size)
    dst = cv2.filter2D(img, -1, kernel)

    return dst


def opening(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return opening


def closing(img, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return closing


def createLabel_RGBD(background,
                     foreground,
                     background_depth,
                     foreground_depth,
                     threshold=100,
                     intr=None,
                     p=None,
                     min_size=100,
                     open=3,
                     close=9,
                     hsv=True,
                     both=False,
                     plot=False,
                     do_cca=True,
                     remove_one_std=False,
                     measure_dist=None):

    if p is None:
        if hsv:
            p = [0.08026211175912534, 1.2577782150904344, 1.9483549172969372, 1.392821046939864]
        #p = [0.08026211175912534, 1.2577782150904344, 1.09483549172969372, 1.392821046939864]
        elif both:
            p = [0.8, 0.6, 0.1, 0.3, 0.3, 0.5, 0.5]
        else:
            p = [0.5, 0.5, 0.5, 1]

    if plot:
        x = 4
        y = 4

        plt.subplot(x, y, 1)
        plt.title('(1) Background Image')
        plt.imshow(background)
        plt.axis('off')
        plt.subplot(x, y, 2)
        plt.title('(2) Foreground Image')
        plt.imshow(foreground)
        plt.axis('off')



    if hsv:
        # convert image to hsv
        background = cv2.cvtColor(background, cv2.COLOR_RGB2HSV)
        foreground = cv2.cvtColor(foreground, cv2.COLOR_RGB2HSV)
    elif both:
        background_hsv = cv2.cvtColor(background, cv2.COLOR_RGB2HSV)
        foreground_hsv = cv2.cvtColor(foreground, cv2.COLOR_RGB2HSV)
        background = np.concatenate((background_hsv, background), axis=2)
        foreground = np.concatenate((foreground_hsv, foreground), axis=2)


    if p[-1] > 0:
        # remove objects far away, are not of interest and might cause issues due to measuring errors
        if not measure_dist:
            max_measure_dist = int(1200) # 1.2 m
        else:
            max_measure_dist = measure_dist + 150
            min_measure_dist = measure_dist - 150

        foreground_depth[foreground_depth > max_measure_dist] = 0
        background_depth[background_depth > max_measure_dist] = 0
        foreground_depth[foreground_depth < min_measure_dist] = 0
        background_depth[background_depth < min_measure_dist] = 0

        h, w = background_depth.shape
        h_p = 0.3
        h_w = 0.3

        center = background_depth[int(h/2 - h*h_p):int(h/2 + h*h_p), int(w/2 - w*h_w):int(w/2 + w*h_w)]
        pos = np.where(center != 0)
        pos = np.array([[x1, y1] for x1, y1 in zip(pos[0], pos[1])])
        if pos != []:


            lowest = np.where(pos[:, 0] == np.max(pos[:, 0]))[0]
            uppest = np.where(pos[:, 0] == np.min(pos[:, 0]))[0]
            rightest = np.where(pos[:, 1] == np.max(pos[:, 1]))[0]
            uppest = uppest[int(len(uppest)/2)]
            if len(lowest)>100:
                lowest = np.sort(lowest)
                pos = np.array([pos[lowest[0]], pos[uppest], pos[lowest[-1]]])
            else:
                lowest = lowest[int(len(lowest) / 2)]
                rightest = rightest[int(len(rightest) / 2)]
                pos = np.array([pos[lowest], pos[uppest], pos[rightest]])
            points = []
            for point in pos:
                points.append([point[0], point[1], center[point[0], point[1]]])

            p1 = np.array(points[0])
            p2 = np.array(points[1])
            p3 = np.array(points[2])

            # These two vectors are in the plane
            v1 = p3 - p1
            v2 = p2 - p1

            # the cross product is a vector normal to the plane
            cp = np.cross(v1, v2)
            a, b, c = cp

            # This evaluates a * x3 + b * y3 + c * z3 which equals d
            d = np.dot(cp, p3)

            pos = np.where(center!=-1)

            Z = (d - a * pos[0] - b * pos[1]) / c
            dist_plane = np.concatenate((np.expand_dims(pos[0], axis=1),
                                         np.expand_dims(pos[1], axis=1),
                                         np.expand_dims(Z, axis=1)), axis=1)
            dist_plane = np.array([np.linalg.norm(point) for point in dist_plane]).reshape(center.shape)
            dist_plane[center!=0] = center[center!=0]
            dist_plane = smoothing(dist_plane, kernel_size=5)
            background_depth[int(h/2 - h*h_p):int(h/2 + h*h_p), int(w/2 - w*h_w):int(w/2 + w*h_w)] = dist_plane


        #background_depth[background_depth == 0] = measure_dist
        #background_depth = spyn.maximum_filter(background_depth, size=20)

        foreground_depth[background_depth == 0] = 0
        background_depth[foreground_depth == 0] = 0

        if plot:
            plt.subplot(x, y, 3)
            plt.title('(3) Background Depth Image')
            plt.imshow(np.array(background_depth / 2000 * 255, dtype=np.uint8), cmap='gray')
            plt.axis('off')
            plt.subplot(x, y, 4)
            plt.title('(4) Foreground Depth Image')
            plt.imshow(np.array(foreground_depth / 2000 * 255, dtype=np.uint8), cmap='gray')
            plt.axis('off')


        # create the depth mask via background subtraction
        depth_mask = np.abs(foreground_depth - background_depth)
        # set range to 0-100
        depth_mask[depth_mask > 100] = 100
        if plot:
            plt.subplot(x, y, 5)
            plt.title('(5) Depth Mask')
            plt.axis('off')
            plt.imshow(depth_mask)



    # convert frames to allow negative numbers
    object_np = np.array(foreground, dtype=np.float64)
    background_np = np.array(background, dtype=np.float64)

    # subtract images and get the abs to also get negative changes
    mask_np = object_np - background_np
    mask_np = np.abs(mask_np)

    if hsv or both:
        mask_np[:, :, 0] *= 256 / 180  # equal strong

    # set range to 0-100
    mask_np[mask_np > 100] = 100

    if plot:
        plt.subplot(x, y, 6)
        plt.axis('off')
        plt.title('(6) HSV Mask')
        plt.imshow(np.array(mask_np[:, :, :3], dtype=np.uint8))

        if both:
            plt.subplot(x, y, 7)
            plt.axis('off')
            plt.title('(7) RGB Mask')
            plt.imshow(np.array(mask_np[:, :, 3:6], dtype=np.uint8))


    # weight channels
    for c in range(int(mask_np.shape[2])):
        mask_np[:, :, c] *= p[c]

    if plot:
        plt.subplot(x, y, 8)
        plt.axis('off')
        plt.title('(8) Weighted HSV Mask')
        plt.imshow(np.array(mask_np[:, :, :3], dtype=np.uint8))

        if both:
            plt.subplot(x, y, 9)
            plt.axis('off')
            plt.title('(9) Weighted RGB Mask')
            plt.imshow(np.array(mask_np[:, :, 3:6], dtype=np.uint8))

    if p[-1] > 0:
        depth_mask *= p[-1]
    # sum image to get binary 2d version.
    mask_np = np.sum(mask_np, axis=2)

    mask_np_color = copy.deepcopy(mask_np)
    if p[-1] > 0:
        mask_np += depth_mask  # adding weighted depthchannel

    if plot:
        plt.subplot(x, y, 10)
        plt.title('(10) Summed Mask')
        plt.axis('off')
        plt.imshow(mask_np)


    # threshold mask
    mask_np[mask_np < threshold] = 0
    if plot:
        plt.subplot(x, y, 11)
        plt.title('(11) Thresholded Mask')
        plt.axis('off')
        plt.imshow(mask_np, cmap='gray')

    if open > 0:
        mask_np = opening(mask_np, open)

    if close > 0:
        mask_np = closing(mask_np, close)

    if plot:
        plt.subplot(x, y, 12)
        plt.title('(12) Opened&Closed mask')
        plt.axis('off')
        plt.imshow(mask_np, cmap='gray')


    if do_cca:  # do connected component analyses

        # pick the component with the hightest average score, if it is above a certain size.
        labeled = connectedComponents(mask_np)
        if plot:
            average_score = np.zeros(mask_np.shape)

        uni, counts = np.unique(labeled, return_counts=True)


        j = 0
        score = 0
        if len(uni > 1):
            for i, u in enumerate(uni[1:]):
                if counts[i+1] > min_size:
                    current_score = int(np.mean(mask_np[labeled == u]))
                    if current_score > score:
                        j = i+1
                        score = current_score
                    if plot:
                        average_score[labeled == u] = current_score


        #mask_np = np.zeros(mask_np.shape, dtype=np.uint8)
        #mask_np[labeled == uni[j]] = 255
        mask_np = mask_np_color
        mask_np[labeled != uni[j]] = 0

        if plot:
            plt.subplot(x, y, 13)
            plt.title('(13) Average Score per Part')
            plt.axis('off')
            plt.imshow(average_score)

        if plot:
            plt.subplot(x, y, 14)
            plt.title('(14) Part with Highest Average Score')
            plt.axis('off')
            plt.imshow(mask_np)

        if remove_one_std:
            mean = np.mean(mask_np[mask_np != 0])
            std = np.std(mask_np[mask_np != 0])
            #print(mean, std, mean - std)
            mask_np[mask_np < mean - std] = 0

            if plot:
                plt.subplot(x, y, 15)
                plt.title('(15) Remove 1 std')
                plt.axis('off')
                plt.imshow(mask_np)


        if open > 0:
            mask_np = opening(mask_np, open)

        if close > 0:
            mask_np = closing(mask_np, close)

        if do_cca:  # do connected component analyses

            # pick the component with the hightest average score, if it is above a certain size.
            labeled = connectedComponents(mask_np)
            if plot:
                average_score = np.zeros(mask_np.shape)

            uni, counts = np.unique(labeled, return_counts=True)

            j = 0
            score = 0
            if len(uni > 1):
                for i, u in enumerate(uni[1:]):
                    if counts[i + 1] > min_size:
                        #current_score = int(np.mean(mask_np[labeled == u]))
                        current_score = int(np.sum(np.array([labeled == u])))
                        if current_score > score:
                            j = i + 1
                            score = current_score
                        if plot:
                            average_score[labeled == u] = current_score

            mask_np = mask_np_color
            mask_np[labeled != uni[j]] = 0

        mask_np[mask_np != 0] = 255
        if plot:
            plt.subplot(x, y, 16)
            plt.title('(16) Connected Component with highest avg score')
            plt.axis('off')
            plt.imshow(mask_np)

    binary_2d = np.array(mask_np, dtype=np.uint8)

    return np.array(binary_2d, dtype=np.uint8)
