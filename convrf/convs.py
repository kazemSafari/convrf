import torch
# import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules.conv import _ConvNd
from .convrf import FilterBank
import numpy as np
# from hdf5storage import loadmat


class Conv2dS(_ConvNd):
    """borrowed from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
     the not learnable weights of the module of shape
     (out_channels, in_channels\groups, kernel_size[0], kernel_size[1])    """

    def __init__(self, num_filters, in_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, padding_mode='zeros', bias=False, fbank_type="frame"):

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        groups = in_channels
        self.kernel_size = kernel_size

        if 1 in self.kernel_size:
            raise ValueError("Cannot have any of kernel dimensions equal to 1.")

        self.fbank_type = fbank_type
        if self.fbank_type not in ["nn_bank", "frame", "pframe"]:
            raise ValueError(f"fbank_type values must be one of the following: 'nn_bank', 'frame', 'pframe' "
                             f"but is input as {self.fbank_type}.")

        fbank = np.float32(self.get_filterbank())
        assert num_filters <= len(fbank)
        fbank = fbank[0:num_filters]
        out_channels = num_filters*in_channels

        super(Conv2dS, self).__init__(
            in_channels, out_channels, kernel_size, stride,
                padding, dilation, False, _pair(0),
                groups, bias, padding_mode)

        fbank = np.repeat(fbank[np.newaxis], in_channels, axis=0)
        fbank = np.reshape(fbank, [out_channels, 1, *kernel_size])
        # fbank = np.repeat(fbank[np.newaxis], in_channels//groups, axis=0).transpose((1, 0, 2, 3))
        fbank = torch.as_tensor(fbank)
        self.register_buffer("kernels", fbank)

    def get_filterbank(self, ):
        return getattr(FilterBank, self.fbank_type)[FilterBank.shape2str(self.kernel_size)]

    def _conv_forward(self, input):

        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            self.kernels.detach(), self.bias, self.stride,
                            _pair(0),
                            self.dilation, self.groups)
        return F.conv2d(input, self.kernels.detach(), self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self._conv_forward(input)


# kernel_size, fbank_type, in_channels = (3, 3), "frame", 2
# nf = getattr(FilterBank, fbank_type)[FilterBank.shape2str(kernel_size)][0:3]
# print(nf.shape)
# print(nf)
# nft = np.repeat(nf[np.newaxis], in_channels, axis=0).transpose((1, 0, 2, 3))
# print(nft.shape)
# print(nft)


# data_path = '/media/kazem/ssd_1tb/datasets/uh2013/contest_uh_data.mat'
# UH2013 = loadmat(data_path)
# IMAGE = UH2013['contest_uh_casi'].transpose((2, 0, 1))
# IMAGE = np.expand_dims(IMAGE, axis=0)
# IMAGE = np.float32(IMAGE)
# IMAGE = torch.as_tensor(IMAGE)
# print(IMAGE.shape)
# conv = Conv2dS(IMAGE.shape[1], 3, padding=1)
# x = conv(IMAGE)
# print(x.size())
