import os
import cv2
import numpy as np


class UnDistort:
    def __init__(self, gird_imgs_dir, grid):
        imgs_ori = self._get_images_by_dir(gird_imgs_dir)
        self.object_points, self.img_points = self._calibrate(imgs_ori, grid)

    def repaire(self, img_dis):
        imgs_undis = undistort._cal_undistort(img_dis)
        return imgs_undis

    def _get_images_by_dir(self, dirname):
        img_names = os.listdir(dirname)
        img_paths = [dirname + '/' + img_name for img_name in img_names]
        imgs = [cv2.imread(path) for path in img_paths]
        return imgs

    def _calibrate(self, images, grid=(9, 6)):
        object_points = []
        img_points = []
        for img in images:
            object_point = np.zeros((grid[0] * grid[1], 3), np.float32)
            object_point[:, :2] = np.mgrid[0:grid[0], 0:grid[1]].T.reshape(-1, 2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, grid, None)
            if ret:
                object_points.append(object_point)
                img_points.append(corners)
        return object_points, img_points

    def _cal_undistort(self, img):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.object_points, self.img_points, img.shape[1::-1], None, None)
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        return dst

    def warp_img(self, img):
        src = np.float32([[(203, 720), (585, 460), (695, 460), (1127, 720)]])  # 原图坐标
        dst = np.float32([[(320, 720), (320, 0), (960, 0), (960, 720)]])  # 变换坐标
        M = cv2.getPerspectiveTransform(src, dst)
        trans = cv2.warpPerspective(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        cv2.imshow('img_test', trans)
        cv2.waitKey()


gridpath = '/home/dell/lg/code/lg_pro_sets/datasets/used_images/camera_cal'
undistort = UnDistort(gridpath, (9, 6))
imgs = undistort._get_images_by_dir(gridpath)
for img in imgs:
    cv2.imshow('imgraw', img)
    img_undist = undistort.repaire(img)
    cv2.imshow('imgrepaire', img_undist)
    cv2.waitKey()
