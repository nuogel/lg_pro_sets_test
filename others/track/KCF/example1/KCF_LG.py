# import tracker
import numpy as np
import cv2
from lgdet.util.util_HOG import HOG
import os
import time
from argparse import ArgumentParser


class Kcftracker(object):
    def __init__(self, img, start_pos, padding=2, HOG_flag=0, dataformat=0, resize=0):

        self.HOG_flag = HOG_flag
        self.padding = padding
        self.dataformat = dataformat
        self.resize = resize
        self.img_size = img.shape[0], img.shape[1]

        if self.dataformat:
            w, h = start_pos[2] - start_pos[0], start_pos[3] - start_pos[1]
            self.pos = start_pos[0], start_pos[1], w, h
        else:
            self.pos = start_pos

        if self.resize:
            self.pos = tuple([ele / 2 for ele in self.pos])
            self.img_size = img.shape[0] // 2, img.shape[1] // 2
            img = cv2.resize(img, self.img_size[::-1])

        object_size = self.pos[2:]
        if self.HOG_flag:
            self.target_size = 32, 32
            self.l = 0.0001
            self.sigma = 0.6
            self.f = 0.012
        else:
            self.target_size = object_size[0] * self.padding, object_size[1] * self.padding
            self.l = 0.0001
            self.sigma = 0.2
            self.f = 0.02
        output_sigma_factor = 1 / float(8)
        # # 用output_sigma_factor，cell_sz和跟踪框尺寸计算高斯标签的带宽output_sigma
        output_sigma = np.sqrt(np.prod(self.target_size)) * output_sigma_factor
        ## 生成汉宁窗cos_window，尺寸与yf相同，即floor(window_sz / cell_size)，检测范围window_sz中cell的个数。对信号进行傅里叶变换时，为了减少频谱泄漏，通常在采样后对信号加窗。
        self.cos_window = np.outer(np.hanning(self.target_size[0]), np.hanning(self.target_size[1]))
        self.y = self.generate_gaussian(self.target_size, output_sigma)
        x = self.get_window(img, self.pos, self.padding)
        x = self.getFeature(x, self.cos_window, self.HOG_flag)
        self.alpha = self.train(x, self.y, self.sigma, self.l)
        self.z = x

    def updateTracker(self, img):
        if self.resize:
            img = cv2.resize(img, self.img_size[::-1])
        x = self.get_window(img, self.pos, self.padding, 1, self.target_size)
        x = self.getFeature(x, self.cos_window, HOG_flag=self.HOG_flag)
        response = self.detect(self.alpha, x, self.z, self.sigma)
        new_pos = self.update_tracker(response, self.img_size, self.pos, HOG_flag=self.HOG_flag, scale_factor=1)
        x = self.get_window(img, new_pos, self.padding, 1, self.target_size)
        x = self.getFeature(x, self.cos_window, HOG_flag=self.HOG_flag)
        new_alpha = self.train(x, self.y, self.sigma, self.l)
        self.alpha = self.f * new_alpha + (1 - self.f) * self.alpha
        new_z = x
        self.z = (1 - self.f) * self.z + self.f * new_z
        self.pos = new_pos
        output_pos = self.pos
        if self.resize:
            output_pos = tuple([ele * 2 for ele in self.pos])
        if self.dataformat:
            output_pos = output_pos[0], output_pos[1], output_pos[0] + output_pos[2], output_pos[1] + output_pos[3]
        return output_pos

    def crop_img(self, img, bbox):
        (x, y, w, h) = bbox
        return img[x:x + w, y:y + h]

    def generate_gaussian(self, win_size, sigma):
        '''
        # 先用gaussian_shaped_labels生成回归标签，然后进行傅里叶变换转换到频域上的yf
        （1）岭回归里面样本对应的输出y是什么？
        答：y不是（0，1）标签哦，是高斯加权后的值。初始目标的位置在padding后的search window的中心，
        循环移位得到的多个样本反应的是背景信息，而且离中心越远，就越不是目标，所以我们对标签进行高斯加权就刚好可以体现这种可能性准则。
        KCF里的输出是一个二维response矩阵，里面元素的大小代表该位置下的目标为预测目标的可能性，因此，在训练的时候就是输入是特征，而输出是一个gaussian_shaped_label，
        一般分类的标签是二值的，或者多值离散的，但是这个高斯标签反应的是由初始目标移位采样形成的若干样本距离初识样本越近可能性越大的准则，
        在代码中，高斯的峰值被移动到了左上角（于是四个角的值偏大），
        原因是“after computing a cross-correlation between two images in the Fourier domain and converting back to the spatial domain,
        it is the top-left element of the result that corresponds to a shift of zero”，
        也就是说目标零位移对应的是左上角的值。这样一来，我们在预测目标位置的时候，只需要pos=pos+find(response==max(response(:)))就好。
        :param win_size:
        :param sigma:
        :return:
        '''
        h, w = win_size
        rx = np.arange(w / 2)
        ry = np.arange(h / 2)
        x = np.hstack((rx, rx[::-1]))  # [1...n_1,n-1,...1]
        y = np.hstack((ry, ry[::-1]))
        xx, yy = np.meshgrid(x, y)
        y_reg = np.exp(-1 * (xx ** 2 + yy ** 2) / (sigma ** 2))  # e^{-(xx^2+yy^2)/sigma^2}
        return y_reg

    def fft(self, img):
        f = np.fft.fft2(img, axes=(0, 1))
        return f

    def cor_fft(self, x1, x2, sigma):
        dist11 = np.sum(np.square(x1))
        dist22 = np.sum(np.square(x2))
        if len(x1.shape) == 2:
            c = np.fft.ifft2((np.conj(self.fft(x1)) * self.fft(x2)))
        else:
            c = np.fft.ifft2(np.sum(np.conj(self.fft(x1)) * self.fft(x2), 2))
        dist = dist11 - 2 * c + dist22
        cor = np.exp(-1 * dist / (sigma ** 2 * x1.size))  # 采用计算高斯核,核矩阵的快速计算
        cor = np.real(cor)
        return cor

    def train(self, x, y, sigma, lambda_):
        '''
         # 岭回归进行训练,x=原box的padding倍, y怎么来的?
        :param x:
        :param y:
        :param sigma:
        :param lambda_:
        :return:
        '''
        k = self.cor_fft(x, x, sigma)
        alpha = self.fft(y) / (self.fft(k) + lambda_)
        return alpha

    def detect(self, alpha, x, last_x, sigma):
        k = self.cor_fft(last_x, x, sigma)
        response = np.real(np.fft.ifft2(alpha * self.fft(k)))  # np.real为元素的实部
        return response

    def update_tracker(self, response, img_size, pos, HOG_flag, scale_factor=1):
        start_w, start_h = response.shape
        w, h = img_size
        px, py, ww, wh = pos
        print('repsponse:', response.max())
        res_pos = np.unravel_index(response.argmax(), response.shape)  # 找出最大值的索引
        scale_w = 1.0 * scale_factor * (ww * 2) / start_w
        scale_h = 1.0 * scale_factor * (wh * 2) / start_h
        move = list(res_pos)
        if not HOG_flag:
            px_new = [px + 1.0 * move[0] * scale_w, px - (start_w - 1.0 * move[0]) * scale_w][1 if (move[0] > start_w / 2) else 0]
            py_new = [py + 1.0 * move[1] * scale_h, py - (start_h - 1.0 * move[1]) * scale_h][1 if (move[0] > start_w / 2) else 0]
            px_new = np.int(px_new)
            py_new = np.int(py_new)
        else:
            move[0] = np.floor(res_pos[0] / 32.0 * (2 * ww))
            move[1] = np.floor(res_pos[1] / 32.0 * (2 * wh))
            px_new = [px + move[0], px - (2 * ww - move[0])][move[0] > ww]
            py_new = [py + move[1], py - (2 * wh - move[1])][move[1] > wh]
        if px_new < 0: px_new = 0
        if px_new > w: px_new = w - 1
        if py_new < 0: py_new = 0
        if py_new > h: py_new = h - 1
        ww_new = np.ceil(ww * scale_factor)
        wh_new = np.ceil(wh * scale_factor)
        new_pos = (px_new, py_new, ww_new, wh_new)
        return new_pos

    def get_window(self, img, bbox, padding, scale_factor=1, rez_shape=None):
        '''
         # 原box的padding倍；get_subwindow获得图像中第一帧的检测的区域patch，
         如果超过图像尺寸会加以修正。作者的修正方法是认为超出部分的值都与边界的值相同。
        :param img:
        :param bbox:
        :param padding:
        :param scale_factor:
        :param rez_shape:
        :return:
        '''
        (x, y, w, h) = bbox
        ix, iy = img.shape[0], img.shape[1]
        center_x = np.int(x + np.floor(w / 2.0))
        center_y = np.int(y + np.floor(h / 2.0))
        w = np.floor(1.0 * w * scale_factor)
        h = np.floor(1.0 * h * scale_factor)
        x_min, x_max = center_x - np.int(w * padding / 2.0), center_x + np.int(w * padding / 2.0)
        y_min, y_max = center_y - np.int(h * padding / 2.0), center_y + np.int(h * padding / 2.0)
        if (x_max - x_min) % 2 != 0:
            x_max += 1
        if (y_max - y_min) % 2 != 0:
            y_max += 1
        lx = 0 if x_min >= 0 else x_min * -1
        ly = 0 if y_min >= 0 else y_min * -1
        rx = 0 if x_max <= ix else x_max - ix
        ry = 0 if y_max <= iy else y_max - iy
        x_min = x_min if lx == 0 else 0
        y_min = y_min if ly == 0 else 0
        x_max = x_max if rx == 0 else ix
        y_max = y_max if ry == 0 else iy
        ww, hh = x_max - x_min, y_max - y_min
        window = (x_min, y_min, ww, hh)
        img_crop = self.crop_img(img, window)
        if lx == 0 and rx == 0 and ly == 0 and ry == 0:
            if rez_shape is not None:
                return cv2.resize(img_crop, rez_shape[::-1])
            else:
                return img_crop
        else:  # get_subwindow获得图像中第一帧的检测的区域patch，如果超过图像尺寸会加以修正。作者的修正方法是认为超出部分的值都与边界的值相同。
            if len(img_crop.shape) == 3:
                img_crop = np.pad(img_crop, ((lx, rx), (ly, ry), (0, 0)), 'edge')
            else:
                img_crop = np.pad(img_crop, ((lx, rx), (ly, ry)), 'edge')
            if rez_shape is not None:
                return cv2.resize(img_crop, rez_shape[::-1])
            else:
                return img_crop

    def process_cos(self, img, cos_window):
        if len(img.shape) == 3:
            channel = img.shape[2]
            cos_mc = np.tile(cos_window, (channel, 1, 1))
            cos_window_out = np.transpose(cos_mc, [1, 2, 0])
        else:
            cos_window_out = cos_window
        return img * cos_window_out

    def prewhiten(self, x):  # (x-mean)/std  ---- 高斯分布
        mean = np.mean(x)
        std = np.std(x)
        a = np.sqrt(x.size)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)  # subtract 两数相减
        return y

    def getFeature(self, x, cos_window, HOG_flag=0):
        '''
        # 利用get_feature获得第一帧patch的特征矩阵， 并加窗；
        :param x:
        :param cos_window:
        :param HOG_flag:
        :return:
        '''
        if HOG_flag:
            x = HOG(x)
        else:
            x = x.astype('float64')
            x = self.prewhiten(x)
        x = self.process_cos(x, cos_window)
        return x


if __name__ == '__main__':

    def display_tracker_lg(img, bbox):
        img = cv2.imread(img)
        visual(img, bbox)


    def visual(img, bbox):
        (x, y, w, h) = bbox
        x1 = int(x)
        y1 = int(y)
        w = int(w)
        h = int(h)
        pt1, pt2 = (x1, y1), (x1 + w, y1 + h)
        img_rec = cv2.rectangle(img, pt1, pt2, (0, 255, 255), 2)
        cv2.imshow('window', img_rec)
        cv2.waitKey()


    def load_bbox(ground_file, resize, dataformat=0):
        f = open(ground_file)
        lines = f.readlines()
        bbox = []
        for line in lines:
            if line:
                pt = line.strip().split(',')
                pt_int = [float(ii) for ii in pt]
                bbox.append(pt_int)
        bbox = np.array(bbox)
        if resize:
            bbox = (bbox.astype('float32') / 2).astype('int')
        else:
            bbox = bbox.astype('float32').astype('int')
        if dataformat:
            bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
            bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
        return bbox


    def load_imglst(img_dir):
        file_lst = [pic for pic in os.listdir(img_dir) if '.jpg' in pic]
        img_lst = [os.path.join(img_dir, filename) for filename in file_lst]
        return img_lst


    def main(args):
        # Load  arg
        dataset = args.dataset_descriptor
        save_directory = args.save_directory
        show_result = args.show_result
        padding = 2
        dataformat = 1

        # Load dataset information and get start position
        title = dataset.split('/')
        title = [t for t in title if t][-1]
        img_lst = load_imglst(dataset + '/img/')
        bbox_lst = load_bbox(os.path.join(dataset + '/groundtruth_rect.txt'), 0, dataformat=0)
        px1, py1, px2, py2 = bbox_lst[0]
        pos = (px1, py1, px2, py2)
        frames = len(img_lst)
        # Attention: the original data format is (y,x,h,w), so the  code above translate
        # the data to (x1,y1,x2,y2) format

        # Create file to record the result
        tracker_bb = []
        result_file = os.path.join(save_directory, title + '_' + 'result_KCF.txt')
        file = open(result_file, 'w')
        start_time = time.time()

        # Tracking
        for i in range(frames):
            img = cv2.imread(img_lst[i])
            if i == 0:
                # Initialize trakcer, img 3 channel, pos(x1,y1,x2,y2)
                kcftracker = Kcftracker(img, pos, padding, HOG_flag=1, dataformat=0)
            else:
                # Update position and traking
                pos = kcftracker.updateTracker(img)

            # Write the position
            out_pos = pos  # [pos[1], pos[0], pos[3] - pos[1], pos[2] - pos[0]]
            win_string = [str(p) for p in out_pos]
            win_string = ",".join(win_string)
            tracker_bb.append(win_string)
            file.write(win_string + '\n')

            if show_result:
                display_tracker_lg(img_lst[i], out_pos)

        duration = time.time() - start_time
        fps = int(frames / duration)
        print('each frame costs %3f second, fps is %d' % (duration / frames, fps))
        file.close()

        result = load_bbox(result_file, 0)

        # # Show the result with bbox
        # if show_result:
        #     display_tracker(img_lst, result, save_flag=0)


    def parse_arguments():
        parser = ArgumentParser()
        parser.add_argument('--dataset_descriptor', type=str, default='E:/datasets/TRACK/OTB100/BlurCar1/',
                            help='The directory of video and groundturth file')
        parser.add_argument('--save_directory', type=str, default='./',
                            help='The directory of result file')
        parser.add_argument('--show_result', type=int,
                            help='Show result or not', default=1)

        return parser.parse_args()


    cfg = parse_arguments()
    main(cfg)
