import math
import cv2
import numpy as np
import os


class IQAScore:
    '''
    PSNR:XXXX
    '''

    def __init__(self, use_torch_mse=True):
        self.use_MSE = use_torch_mse

    def ssim(self, img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def run_image(self, path_image1, path_image2):
        img1 = cv2.imread(path_image1).astype(np.float32) / 255.0
        img2 = cv2.imread(path_image2)
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_CUBIC)
            # cv2. INTER_LINEAR，双线性插值（缺省）；cv2.INTER_NEAREST，最近邻域插值；cv2. INTER_CUBIC，4x4像素邻域的双三次插值；
            # cv2. INTER_LANCZOS4，8x8像素邻域的Lanczos插值；cv2. INTER_AREA， 像素关系重采样
        img2 = img2.astype(np.float32) / 255.0

        if self.use_MSE:
            import torch
            self.mse = torch.nn.MSELoss()
            mse = self.mse(torch.FloatTensor(img1), torch.FloatTensor(img2))

        else:
            shape = img1.shape
            mse = np.sum((img1 - img2) ** 2 / (shape[0] * shape[1] * shape[2]))

        psnr = 10 * math.log10(1.0 / mse.item())
        return psnr

    def video2images(self, video_path):
        cap = cv2.VideoCapture(video_path)
        i = 0
        ret = True
        while ret:
            ret, frame = cap.read()
            if not ret: break
            write_path = os.path.dirname(video_path)
            # cv2.imshow('img', frame)
            # cv2.waitKey()
            cv2.imwrite(os.path.join(write_path, str(i) + '.jpg'), frame)
            i += 1

    def _read_train_test_dataset(self, idx_stores_dir):
        f = open(idx_stores_dir, 'r')
        test_set = [line.strip().split(';') for line in f.readlines()]
        return test_set


if __name__ == '__main__':
    # path_1 = 'E:/datasets/kitti/training/images/000001.png'
    # path_2 = 'E:/datasets/kitti/training/images/000002.png'
    idx_store_dir = '../tmp/idx_stores/test_set.txt'
    # video_path = 'F:/datasets/SR_video/youku_test/5/5.mp4'
    #
    images_raw_file = 'F:/datasets/SR/youku/youku_00000_00149_h_GT/Youku_00000_h_GT/'  # F:/datasets/SR_video/youku_test
    images_rec_file = 'F:/datasets/SR_video/youku_test/4/'  # 28.6
    images_low_file = 'F:/datasets/SR/youku/youku_00000_00149_l/Youku_00000_l'  # 30.46

    psnr = IQAScore(use_torch_mse=False)
    # psnr.video2images(video_path)
    out = []
    from_txt = True
    if from_txt:
        raw_files = psnr._read_train_test_dataset(idx_store_dir)
    else:
        raw_files = os.listdir(images_raw_file)

    for i, file in enumerate(raw_files):
        if from_txt:
            image1, image2 = file[1], file[2]
        else:
            image1 = os.path.join(images_raw_file, file)
            image2 = os.path.join(images_low_file, file.replace('h_GT', 'l'))
        if (not os.path.isfile(image1)) or (not os.path.isfile(image2)):
            print(image1, image2)
        else:
            _psnr = psnr.run_image(image1, image2)
            out.append(_psnr)
            score = np.asarray(out).mean()
            print(i, '/', len(raw_files), '  [PSNR]: ', _psnr, '[Average]: ', score)
