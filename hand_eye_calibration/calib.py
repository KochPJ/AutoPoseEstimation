import cv2
from cv2 import aruco
from depth_camera.DepthCam import DepthCam
import numpy as np
import time
import os
import matplotlib.pyplot as plt


def get_board(intr):
    squaresX = 6
    squaresY = 7
    squareLength = 15
    markerLength = 10
    aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)
    board = cv2.aruco.CharucoBoard_create(squaresX, squaresY, squareLength, markerLength, aruco_dict)
    distCoeffs = np.zeros(5)
    cameraMatrix = np.array([[intr.fx , 0., intr.ppx],
                             [0., intr.fy, intr.ppy],
                             [0., 0., 1.]])
    return aruco_dict, board, distCoeffs, cameraMatrix

def stream_cam_pose(stop_after_image=False):
    cam = DepthCam(depth_frame_height=720, depth_frame_width=1280, color_frame_height=720, color_frame_width=1280)
    intr = cam.get_intrinsics()
    aruco_dict, board, distCoeffs, cameraMatrix = get_board(intr)

    fps = 2
    while True:
        start = time.time()
        intr = cam.get_intrinsics()
        out = cam.get_frames()
        gray = cv2.cvtColor(out['image'], cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        imsize = gray.shape


        charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, gray, board)
        rvec = None
        tvec = None
        retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix, distCoeffs, rvec, tvec)
        rmat = cv2.Rodrigues(rvec)[0]
        trans_mat = np.zeros((4,4))
        trans_mat[3,3] = 1
        trans_mat[:3, :3] = rmat
        trans_mat[:3, 3] = tvec[:, 0]
        print(trans_mat)
        eps = time.time()-start
        if eps < 1/fps:
            time.sleep(1/fps - eps)
        if stop_after_image:
            input()


def get_cam_poses(images, intr, save_path=None):
    aruco_dict, board, distCoeffs, cameraMatrix = get_board(intr)
    cam_poses = []

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners)>0:
            charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids, gray, board)
            rvec = None
            tvec = None
            retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix, distCoeffs,
                                                                rvec, tvec)

            rmat = cv2.Rodrigues(rvec)[0]
            trans_mat = np.zeros((4,4))
            trans_mat[3,3] = 1
            trans_mat[:3, :3] = rmat
            trans_mat[:3, 3] = tvec[:, 0]
            cam_poses.append(trans_mat.flatten())


    cam_poses = np.array(cam_poses)
    if not save_path:
        print(cam_poses)
    else:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path = os.path.join(save_path, 'cam_poses.yaml')
        fs_write = cv2.FileStorage(save_path, cv2.FILE_STORAGE_WRITE)
        fs_write.write("poses", cam_poses)
        fs_write.release()


    return cam_poses

def read_chessboards(images, intr):
    """
    Charuco base pose estimation.
    """
    aruco_dict, board, distCoeffs, cameraMatrix = get_board(intr)
    allCorners = []
    allIds = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)

    for im in images:
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict)

        if len(corners)>0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize = (3,3),
                                 zeroZone = (-1,-1),
                                 criteria = criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(corners,ids,gray,board)
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])

        decimator += 1

    imsize = gray.shape
    return allCorners, allIds, imsize

def calibrate_camera(allCorners, allIds, imsize, intr):
    """
    Calibrates the camera using the dected corners.
    """
    aruco_dict, board, distCoeffs, cameraMatrix = get_board(intr)


    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_ASPECT_RATIO)
    #flags = (cv2.CALIB_RATIONAL_MODEL)
    (ret, camera_matrix, distortion_coefficients0,
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrix,
                      distCoeffs=distCoeffs,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors


def calibrate(path, intr):
    dirs = os.listdir(path)
    images = [os.path.join(path, p) for p in dirs]
    allCorners, allIds, imsize = read_chessboards(images, intr)
    ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors = calibrate_camera(allCorners, allIds, imsize, intr)
    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors

def apply_calibration(image, camera_matrix, distortion_coefficients0):
    image = cv2.undistort(image, camera_matrix, distortion_coefficients0, None)
    return image


if __name__ == '__main__':

    cam = DepthCam()
    intr = cam.get_intrinsics()

    path = './test'
    stream_cam_pose(stop_after_image=True)

    poses = get_cam_poses(path, intr)
    aruco_dict, board, distCoeffs, cameraMatrix = get_board(intr)

    frame = cv2.imread('./test/test3_Color.png')

    img_undist = apply_calibration(frame, cameraMatrix, distCoeffs)


    plt.subplot(1,2,1)
    plt.imshow(frame)
    plt.title("Raw image")
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(img_undist)
    plt.title("Corrected image")
    plt.axis("off")
    plt.show()
