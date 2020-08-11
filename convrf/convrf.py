import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules.conv import _ConvNd
import torch.nn.init as init
from torch.nn.init import calculate_gain, _calculate_correct_fan, _calculate_fan_in_and_fan_out
from .parseval import Parseval
# import sys
# from pathlib import Path
# from time import time

# from random import sample
# from numpy.random import default_rng
# rng = default_rng()
# rng does not generalize to distributed training since the initial state
# of the different processes are not identical even when fixing the random seed using
# np.random.seed(seed) # We use np.random.choice instead which does not have such issues.

"""
Written and copywrite by Kazem Safari. 
Class 'Parseval' is borrowed from Nikolaos Karantzas' Github.
Thanks to Mozahid Haque for his insightful comments.
"""

class _FilterBank(object):
    def __init__(self):

        self.frame = self.get_fb("frame", [(3, 3), (5, 5), (7, 7)])
        self.pframe = self.get_fb("pframe", [(3, 3), (5, 5), (7, 7)])
        self.nn_bank = self.get_fb("nn_bank", [(3, 3), (5, 5), (7, 7)])

    def get_fb(self, fbank_type, shapes):
        return {
            self.shape2str(item):
                np.float32(Parseval(
                    shape=item,
                    low_pass_kernel='gauss',
                    first_order=True,
                    second_order=True,
                    bank=fbank_type).fbank())
            for item in shapes
        }

    def shape2str(self, shape):
        return 'x'.join([f'{dim}' for dim in shape])


FilterBank = _FilterBank()
# print(FilterBank.pframe['7x7'].shape)
# print(FilterBank.frame)
# print(sys.getrefcount(FilterBank))
# print(sys.getrefcount(FilterBank.nn_bank))




def kaiming_uniform_mod(tensor, a=0, gain=1, mode='fan_in', nonlinearity='leaky_relu'):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
        used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """
    gain_mod = gain  # The "gain_mod" mutiplier is my only modification
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = (gain / math.sqrt(fan)) * gain_mod
    # print(gain, gain_mod)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class Conv2d_(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', gain=1):


        super(Conv2d_, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        self.reset_parameters(gain)
    def reset_parameters(self, gain=1):
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        super(Conv2d_, self).reset_parameters()
        kaiming_uniform_mod(self.weight,
                            a=math.sqrt(5),
                            gain=gain,
                            mode='fan_in',
                            nonlinearity='leaky_relu')
        if self.bias is not None:
            # the gain multiplier is my only modification
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = gain*(1 / math.sqrt(fan_in))
            init.uniform_(self.bias, -bound, bound)



class _ConvNdRF(_ConvNd):
    """borrowed from https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py
    The only new argument that is necessary to be added to _ConvNd's arguments is kernel_drop_rate.

    The kernel_span tensor defined in forward replaces the nn._ConvNd's weight tensor by
    a linear combination of predefined filters in each of its convolutional channels.

    Therefore there are two ingrdients to it: self.weight and self.kernels

    1) self.weight is a tensor that defines the coefficients used in such linear combinations.
    2) self.kernels is another tensor that defines the vectors (filters).

    Now there are two cases when writing such self.kernel_span:
    1) All the filters present in fbank are used in each linear combination per convolutional channel.
    2) A random subset of fbank are used.

    The 'kernels' tensor is a non-trainable parameter that should be saved and restored in the state_dict,
    therefore we register them as buffers. Buffers won’t be returned in model.parameters()

    # According to ptrblck's comments,
    .detach() prevents cpu memory leak from the "self.kernels" buffer in "forward".

    The following links were helpful and used in building this package:
    for memory leak issues:
    https://github.com/pytorch/pytorch/issues/20275
    https://discuss.pytorch.org/t/how-does-batchnorm-keeps-track-of-running-mean/40084/15

    difference between .data and .detach:
    https://github.com/pytorch/pytorch/issues/6990

    for masking the gradient:
    https://discuss.pytorch.org/t/update-only-sub-elements-of-weights/29101/2
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, kernel_drop_rate=0, fbank_type="frame", gain=1):

        super(_ConvNdRF, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, transposed, output_padding,
            groups, bias, padding_mode)
        self.kernel_drop_rate = kernel_drop_rate
        self.fbank_type = fbank_type
        self.gain = gain
        # print(self.gain)
        if self.fbank_type not in ["nn_bank", "frame", "pframe"]:
            raise ValueError(f"fbank_type values must be one of the following: 'nn_bank', 'frame', 'pframe' "
                             f"but is input as {self.fbank_type}.")
        if self.kernel_drop_rate >= 1 or self.kernel_drop_rate < 0:
            raise ValueError(f"Can't drop all kernel. "
                             f"kernel_drop_rate must be a value strictly less than 1, "
                             f"But found {self.kernel_drop_rate}.")
        if 1 in self.kernel_size:
            raise ValueError("Cannot have any of kernel dimensions equal to 1.")

        if self.kernel_drop_rate == 0:
            self.get_all_kernels()
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, self.total_kernels))
        else:
            self.get_some_kernels()
            self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, self.num_kernels))

        self.reset_parameters(self.gain)

    def reset_parameters(self, gain=1):
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        super(_ConvNdRF, self).reset_parameters()
        kaiming_uniform_mod(self.weight,
                            a=math.sqrt(5),
                            gain=gain,
                            mode='fan_in',
                            nonlinearity='leaky_relu')
        if self.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            bound = gain*(1 / math.sqrt(fan_in))
            init.uniform_(self.bias, -bound, bound)

    def get_all_kernels(self, ):
        """(num_filters, height, width)"""
        # this will load a custom pre-designed kernel_size filter-bank
        fbank = np.float32(self.get_filterbank())
        assert fbank.ndim == 3, "dimensions has to be 3, but found {}".format(fbank.ndim)
        self.total_kernels = fbank.shape[0]
        self.num_kernels = int((1-self.kernel_drop_rate) * self.total_kernels)
        # print(f"nk: {self.num_kernels}")
        # torch.tensor() always copies data. To avoid copying the numpy array, use torch.as_tensor() instead.
        fbank = torch.as_tensor(fbank)
        self.register_buffer("kernels", fbank)

    def get_some_kernels(self, ):
        fbank = np.float32(self.get_filterbank())
        self.total_kernels = fbank.shape[0]
        assert fbank.ndim == 3, "dimensions has to be 3, but found {}".format(fbank.ndim)
        self.num_kernels = int((1-self.kernel_drop_rate) * self.total_kernels)
        # print(f"nk: {self.num_kernels}")
        total = self.out_channels * (self.in_channels // self.groups)
        # select random indices "total" number of times
        indices = np.array(list(map(lambda x: np.random.choice(self.total_kernels, self.num_kernels, replace=False),
                                    np.zeros(total, dtype=np.uint8))))
        # Get the kernels from fbank that correspond to "indices"
        kernels = np.take(fbank, indices, axis=0)
        # reshape kernels to match the dimensions of a convolutional weights
        kernels = np.reshape(kernels,
                             (self.out_channels,
                              self.in_channels//self.groups,
                              self.num_kernels,
                              *self.kernel_size))

        kernels = torch.as_tensor(kernels)
        self.register_buffer("kernels", kernels)

    def get_filterbank(self, ):
        return getattr(FilterBank, self.fbank_type)[FilterBank.shape2str(self.kernel_size)]
        # return Parseval(
        #     shape=self.kernel_size,
        #     low_pass_kernel='gauss',
        #     first_order=True,
        #     second_order=True,
        #     bank=self.fbank_type).fbank()


class Conv2dRF(_ConvNdRF):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', kernel_drop_rate=0, fbank_type="frame", gain=1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        if 1 in kernel_size:
            raise ValueError("All kernel dimension values must be greater than 1.")

        super(Conv2dRF, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode, kernel_drop_rate, fbank_type, gain)


    def _conv_forward(self, input, weight):

        kernel_span = self.get_kernel_span(weight, self.kernels)
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            kernel_span, self.bias, self.stride,
                            _pair(0),
                            self.dilation, self.groups)
        return F.conv2d(input, kernel_span, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self._conv_forward(input, self.weight)

    def get_kernel_span(self, weight, kernels):
        if self.kernel_drop_rate == 0:
            kernel_span = torch.einsum("ijk, klm -> ijlm", weight, kernels.detach())
        else:
            kernel_span = torch.einsum("ijk, ijklm -> ijlm", weight, kernels.detach())
        return kernel_span
