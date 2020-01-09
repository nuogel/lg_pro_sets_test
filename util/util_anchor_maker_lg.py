import torch
import numpy as np
from itertools import product as product


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source feature map.
    """

    def __init__(self, image_shape=None, pyramid_levels=None, strides=None, sizes=None, scales=None):
        super(PriorBox, self).__init__()
        self.pyramid_levels = pyramid_levels
        self.strides = strides
        self.sizes = sizes
        self.scales = scales
        self.image_shape = np.array(image_shape)  # H,W
        if self.pyramid_levels is None:
            self.pyramid_levels = [3, 4, 5, 6]  # [0, 1, 2...]
        if self.strides is None:
            self.strides = [2 ** x for x in self.pyramid_levels]
        if self.sizes is None:
            self.min_sizes = [2 ** (x + 2) for x in self.pyramid_levels]
        if self.scales is None:
            self.scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]  # [1, 1.25, 1.5]
        self.feature_maps = [(self.image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]

        # self.image_size = 512
        # number of priors for feature map location (either 4 or 6)
        # self.feature_maps = [64, 32, 16, 8]
        # self.min_sizes = [32, 64, 128, 256]
        # self.max_sizes = []
        # self.strides = [8, 16, 32, 64]
        self.clip = True

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            iddd = product(range(f[0]), range(f[1]))
            for i, j in iddd:
                f_k = self.image_shape / self.strides[k]
                # unit center x,y
                cx = (j + 0.5) / f_k[1]
                cy = (i + 0.5) / f_k[0]

                # aspect_ratio: 1
                # rel size: min_size
                s_k_w = self.min_sizes[k] / self.image_shape[1]
                s_k_h = self.min_sizes[k] / self.image_shape[0]
                mean += [cx, cy, s_k_w, s_k_h]

                # aspect_ratio: 1
                # rel size: torch.sqrt(s_k * s_(k+1))

                # rest of aspect ratios
                for s in self.scales:
                    ratio = np.sqrt(2)
                    mean += [cx, cy, s_k_w * ratio * s, s_k_h / ratio * s]
                    mean += [cx, cy, s_k_w / ratio * s, s_k_h * ratio * s]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(min=0, max=1)
        return output


if __name__ == '__main__':
    p = PriorBox(pyramid_levels=[3, 4, 5, 6], scales=[1])
    z = p.forward()
    a = 0
