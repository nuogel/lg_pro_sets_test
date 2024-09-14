import numpy as np
import torch
import torch.nn as nn


class Anchors(nn.Module):
    '''
    :return anchors: [x y w h]
    '''

    def __init__(self, pyramid_levels=None, strides=None, base_size=None, ratios=None, scales=None):
        super(Anchors, self).__init__()
        self.pyramid_levels = pyramid_levels
        self.strides = strides
        self.base_size = base_size
        self.ratios = ratios
        self.scales = scales
        self.base_size_ratio = 2  # lg: for small target to fit better anchors
        if self.pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6, 7]
        if self.strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if self.base_size is None:
            self.base_size = [2 ** (x + 2) for x in self.pyramid_levels]
        if self.ratios is None:
            self.ratios = np.array([0.5, 1, 2])
        if self.scales is None:
            self.scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])  # ])  #

    def forward(self, image):
        image_shape = image.shape[2:]
        n, c, h, w = image.shape
        image_shape = np.array(image_shape)
        # 计算feature map 的size.
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        # compute anchors over all pyramid levels
        all_anchors = np.zeros((0, 4)).astype(np.float32)

        for idx, p in enumerate(self.pyramid_levels):
            anchors = self._generate_anchors(base_size=self.base_size[idx] / self.base_size_ratio, ratios=self.ratios, scales=self.scales)
            shifted_anchors = self._shift(image_shapes[idx], self.strides[idx], anchors)
            all_anchors = np.append(all_anchors, shifted_anchors, axis=0)

        # all_anchors = np.expand_dims(all_anchors, axis=0)
        all_anchors_xywh = torch.from_numpy(
            all_anchors.astype(np.float32) / np.asarray([w, h, w, h], dtype=np.float32)).to(image.device)
        all_anchors_xywh = all_anchors_xywh.unsqueeze(0).repeat([image.shape[0], 1, 1])  # for multi GPU
        return all_anchors_xywh

    def _generate_anchors(self, base_size=16, ratios=None, scales=None):
        """
        Generate anchor (reference) windows by enumerating aspect ratios X
        scales w.r.t. a reference window.
        """

        if ratios is None:
            ratios = np.array([0.5, 1, 2])

        if scales is None:
            scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

        num_anchors = len(ratios) * len(scales)

        # initialize output anchors
        anchors = np.zeros((num_anchors, 4))

        # scale base_size
        anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]

        # correct for ratios
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

        # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        # anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        # anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

        return anchors

    def _shift(self, shape, stride, anchors):
        shift_x = (np.arange(0, shape[1]) + 0.5) * stride
        shift_y = (np.arange(0, shape[0]) + 0.5) * stride

        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        # shifts = np.vstack((
        #     shift_x.ravel(), shift_y.ravel(),
        #     shift_x.ravel(), shift_y.ravel()
        # )).transpose()
        shifts = np.vstack((
            shift_x.ravel(), shift_y.ravel(), np.zeros(shift_x.ravel().shape), np.zeros(shift_y.ravel().shape)
        )).transpose()
        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = anchors.shape[0]
        K = shifts.shape[0]
        all_anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
        all_anchors = all_anchors.reshape((K * A, 4))

        return all_anchors


if __name__ == '__main__':
    anchor = Anchors()
