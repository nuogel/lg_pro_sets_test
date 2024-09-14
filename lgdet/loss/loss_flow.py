"""Loss calculation based on yolo."""
import torch
import torch.nn.functional as F
from lgdet.util.util_show_FLOW import _viz_flow


class FlowLoss:
    # pylint: disable=too-few-public-methods
    """Calculate loss."""

    def __init__(self, cfg):
        """Init."""
        #
        self.cfg = cfg
        self.batch_size = cfg.TRAIN.BATCH_SIZE

    def EPE(self, input_flow, target_flow, sparse=False, mean=True):
        EPE_map = torch.norm(target_flow - input_flow, 2, 1)
        batch_size = EPE_map.size(0)
        if sparse:
            # invalid flow is defined with both flow coordinates to be exactly 0
            mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)

            EPE_map = EPE_map[~mask]
        if mean:
            return EPE_map.mean()
        else:
            return EPE_map.sum() / batch_size

    def sparse_max_pool(self, input, size):
        '''Downsample the input by considering 0 values as invalid.

        Unfortunately, no generic interpolation mode can resize a sparse map correctly,
        the strategy here is to use max pooling for positive values and "min pooling"
        for negative values, the two results are then summed.
        This technique allows sparsity to be minized, contrary to nearest interpolation,
        which could potentially lose information for isolated data points.'''

        positive = (input > 0).float()
        negative = (input < 0).float()
        output = F.adaptive_max_pool2d(input * positive, size) - F.adaptive_max_pool2d(-input * negative, size)
        return output

    def multiscaleEPE(self, network_output, target_flow, weights=None, sparse=False):
        def one_scale(output, target, sparse):

            b, _, h, w = output.size()

            if sparse:
                target_scaled = self.sparse_max_pool(target, (h, w))
            else:
                target_scaled = F.interpolate(target, (h, w), mode='area')
            return self.EPE(output, target_scaled, sparse, mean=False)

        if type(network_output) not in [tuple, list]:
            network_output = [network_output]
        if weights is None:
            weights = [0.005, 0.01, 0.02, 0.08, 0.32]  # as in original article
        assert (len(weights) == len(network_output))

        loss = 0
        for output, weight in zip(network_output, weights):
            loss += weight * one_scale(output, target_flow, sparse)
        return loss

    def realEPE(self, output, target, sparse=False):

        b, _, h, w = target.size()
        upsampled_output = F.interpolate(output, (h, w), mode='bilinear', align_corners=False)
        epe = self.EPE(upsampled_output, target, sparse, mean=True)
        if self.cfg.TRAIN.SHOW_PREDICTED:
            flow2bgr1 = output[0].permute(1, 2, 0)
            h, w, _ = flow2bgr1.shape
            downsampled_target = F.interpolate(target, (h, w), mode='bilinear', align_corners=False)
            flow2bgr2 = downsampled_target[0].permute(1, 2, 0)
            _viz_flow(inputs=None, flow=flow2bgr1, predicted=flow2bgr2, show_time=self.cfg.TRAIN.SHOW_PREDICTED)
        return epe

    def Loss_Call(self, network_output, dataset, losstype=''):
        inputimages, target_flow, datainfo = dataset
        if self.cfg.TRAIN.MODEL in ['flow_fgfa']:
            loss = self.realEPE(network_output, target_flow)
        else:
            loss = self.multiscaleEPE(network_output, target_flow)

        return loss, None
