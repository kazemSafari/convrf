import math
import numpy as np
import torch
import torch.nn as nn
from random import sample
import torch.nn.functional as F
from numpy.random import default_rng
from ..parseval import Parseval
from ..rf_args import rf_args
rng = default_rng()


# TODO: needs to be completely rewritten and cleaned up based on conv_rf.py
class Conv2dRF_v1(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 kernel_drop_rate=0
                 ):
        super(Conv2dRF_v1, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        # get the filterbank kernels as buffers
        self.get_kernels(kernel_size, kernel_drop_rate)
        # initialize the coefficients for the linear combinations
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, self.total_kernels))

        # when using less than the total number of kernels available in the filterbank, create a mask to mask out
        # the filters indices you do not need
        if self.num_kernels < self.total_kernels:
            self.mask = self.create_mask()
            print("mask: ", self.mask)
            # set the gradient of self.weights on self.mask to be always zero.
            # self.weight.register_hook(self.hook_fn)

        self.reset_parameters()

        if self.num_kernels < self.total_kernels:
            self.weight.data[self.mask.data] = 0  # set the initial values self.weights on self.mask to zero as well.

    def get_kernels(self, kernel_size, kernel_drop_rate):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if 1 in kernel_size:
            raise AssertionError("Cannot have any of kernel dimensions equal to 1.")
        # this will load a custom pre-designed kernel_size filter-bank
        kernels = np.float32(self.get_filterbank())
        print(kernels.dtype)

        assert kernels.ndim == 3, \
            "number of dimensions of kernels numpy array has to be 3, it's shape is: (num_filters, height, width)"
        self.total_kernels, self.kernel_size = kernels.shape[0], kernels.shape[1:]
        self.num_kernels = int(self.total_kernels*(1-kernel_drop_rate))
        print(self.num_kernels)
        assert 1 <= self.num_kernels <= self.total_kernels

        kernels = torch.as_tensor(kernels)
        self.register_buffer("kernels", kernels)

    def random_selection(self, arr):
        # this method is faster for smaller values of self.total_kernels but still creates lists in memory!!!
        # arr[random.sample(range(0, self.total_kernels), self.total_kernels-self.num_kernels)] = True

        # this method is faster for large values of self.total_kernels and small values of self.num_kernels,
        # and does not create a list in memory!!!! Therefore this method is more memory efficient.
        arr[rng.choice(self.total_kernels, self.total_kernels-self.num_kernels, replace=False)] = True
        return arr

    def create_mask(self,):
        # Only if self.num_kernels < self.total_kernels, then for convolutional channels
        # (whether input_channel or output_channel) randomly select a subset of the kernels.
        # "mask" is a boolean tensor which specifies which kernels not to be used (or masked) in each channel
        mask = np.zeros((self.out_channels, self.in_channels // self.groups, self.total_kernels), dtype=bool)
        mask = np.apply_along_axis(self.random_selection, axis=2, arr=mask)
        mask = torch.as_tensor(mask,  dtype=torch.bool)
        # self.register_buffer("mask", mask)
        # print("size of mask tensor in bytes: ", self.mask.element_size() * self.mask.nelement())
        return mask

    # def hook_fn(self, grad):
    #     # You are not allowed to modify inplace what is given !
    #     # "grad" is a leaf node and cannot be modified in place, so you need to clone it.
    #     out = grad.clone()
    #     out[self.mask] = 0  # do not calculate gradients on self.mask values
    #     # grad.detach()
    #     # del grad
    #     return out

    def get_filterbank(self,):
        print(self.kernel_size)
        return Parseval(
            shape=self.kernel_size,
            low_pass_kernel='gauss',
            first_order=True,
            second_order=True,
            bank='nn_bank').fbank()

    def forward(self, x):
        # write the convolutional weight as a linear combination of a subset of the custom pre-designed filters
        # weight = sum(torch.mul(self.weight, self.kernels), dim=2, keepdim=False)
        weight = torch.einsum("ijk, klm -> ijlm", self.weight, self.kernels.detach())
        out = F.conv2d(
            input=x,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)
        return out


class Conv2dRF(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True,
                 ):
        super(Conv2dRF, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        kernels_filename_2d = str(kernel_size)+'x'+str(kernel_size)+'.npy'
        kernels = np.float32(np.load(rf_args.kernels_path_2d/(kernels_filename_2d)))
        assert kernels.ndim == 3
        total_kernels, kernel_size = kernels.shape[0], (kernels.shape[1], kernels.shape[2])
        assert 1 <= rf_args.num_kernels_2d <= total_kernels
        lin_comb_kernels = torch.Tensor(
            self.build_kernels(
            kernels,
            rf_args.num_kernels_2d,
            out_channels,
            in_channels))
        print(lin_comb_kernels.element_size() * lin_comb_kernels.nelement())
        self.register_buffer("lin_comb_kernels", lin_comb_kernels)
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, rf_args.num_kernels_2d))
        # print(self.weight.element_size() * self.weight.nelement())
        self.reset_parameters()

    def build_kernels(self, kernels, num_kernels, out_channels, in_channels):
        """
        :param kernels:  np.array, fixed kernels of shape [num_kernels, kernel_height, kernel_width]
        :param num_kernels: int, number of randomly selected subset of kernels for each channel (in_channel or
        out_channel)
        :param in_channels: int, number of input channels
        :param out_channels: int, number of output channels
        :return:
        """
        assert kernels.ndim == 3
        total_kernels, h, w = kernels.shape[0], kernels.shape[1], kernels.shape[2]
        lin_comb_kernels = np.zeros((out_channels, in_channels//self.groups, num_kernels, h, w), dtype=np.float32)
        for k in range(out_channels):
            for j in range(in_channels//self.groups):
                lin_comb_kernels[k, j] = kernels[sample(range(0, total_kernels), num_kernels), :, :]
        return lin_comb_kernels

    def forward(self, x):
        weight = torch.einsum("ijk, ijklm -> ijlm", self.weight, self.lin_comb_kernels),  # lin comb op
        out = F.conv2d(
            input=x,
            weight=weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)
        return out


class Conv2dRFv2(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_kernels,
                 kernel_path=None,
                 padding=0,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True,):
        super(Conv2dRFv2, self).__init__(
            in_channels,
            out_channels,
            num_kernels,
            kernel_path,
            padding,
            stride,
            dilation,
            groups,
            bias)
        self.kernel_path = kernel_path
        filters = np.load(self.kernel_path)
        print(filters.shape)
        self.w, self.h, self.num_filters = filters.shape
        assert filters.ndim == 3
        self.kernel_size = (filters.shape[0], filters.shape[1])

        # the kernels that are going to be used inside each convolution kernel
        # which will be written as a linear combinations of those kernels
        kernels = filters[:, :, sample(range(0, self.num_filters), self.num_kernels)]
        kernels = torch.Tensor(kernels.reshape((1, self.num_kernels, self.w, self.h)))
        self.register_buffer("kernels", kernels)

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)

        self.conv1x1 = nn.Conv2d(self.num_kernels, self.out_channels * self.in_channels, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv1x1.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        In the forward function we accept a torch.Tensor of input data and we must return
        a torch.Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on torch.Tensors.
        """
        conv2d_kernels = self.conv1x1(self.kernels)
        conv2d_kernels = conv2d_kernels.reshape(self.out_channels,
                                                self.in_channels // self.groups,
                                                self.w, self.h)

        return F.conv2d(
            input=x,
            weight=conv2d_kernels,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)


class Conv2dRFD(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            nrsffpc,
            dropout_probability,
            ffp,
            padding=0,
            stride=1,
            dilation=1,
            group=1):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        :param ffp: stands for Fixed filters Path, absolute path to filters numpy array.
        It must be a numpy array of shape = (width, height, num_filters)
        :param nrsffpc: stands for Number of Randomly Selected Fixed Kernels Per Channel
        (either out_channel or in_channel)
        """
        super(Conv2dRFD, self).__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.nrsffpc = nrsffpc
        self.dropout_probability = dropout_probability
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.group = group

        self.ffp = ffp
        self.filters = np.load(self.ffp)
        # print(self.filters.shape)
        assert self.filters.ndim == 3
        self.kernel_size = (self.filters.shape[0], self.filters.shape[1])

        rsffpc = torch.Tensor(self.build_fixed_kernels(
            self.filters,
            self.nrsffpc,
            self.out_channels,
            self.in_channels))
        self.register_buffer("rsffpc", rsffpc)

        mask_tensor = torch.Tensor(np.ones((self.out_channels, self.in_channels, self.nrsffpc)))
        self.register_buffer("mask_tensor", mask_tensor)
        # define dropout layer in __init__
        self.dropout_layer = nn.Dropout(p=self.dropout_probability)

        # self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.nrsffpc, 1, 1))
        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.nrsffpc))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        stdv = 1. / math.sqrt(in_channels)
        self.bias.data.uniform_(-stdv, stdv)

    def build_fixed_kernels(self, filters, nrsffpc, out_channels, in_channels):
        """
        :param filters:  np.array of fixed kernels;
        must be of shape [filters_height, filters_width, num_filters]
        :param nrsffpc: number of randomly selected fixed filters per channel
        :param in_channels: number of input channels in the tf.nn.conv2d
        :param out_channels: number of output channels in the tf.nn.conv2d
        :return:
        """
        assert filters.ndim == 3
        h = filters.shape[0]
        w = filters.shape[1]
        nff = filters.shape[2]  # number of fixed filters
        channels = np.zeros((out_channels, in_channels, h, w, nrsffpc), dtype=np.float32)
        for k in range(out_channels):
            for j in range(in_channels):
                channels[k, j] = filters[:, :, sample(range(0, nff), nrsffpc)]
        channels = np.transpose(channels, (0, 1, 4, 2, 3))
        return channels

    def forward(self, x):
        """
        In the forward function we accept a torch.Tensor of input data and we must return
        a torch.Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on torch.Tensors.
        """
        # apply model dropout, responsive to eval()
        self.masker = self.dropout_layer(self.mask_tensor)
        self.masked_weight = torch.mul(self.masker, self.weight)
        self.masked_filters = \
            torch.mul(self.masker.view((self.out_channels, self.in_channels, self.nrsffpc, 1, 1)), self.rsffpc)
        # print(self.name + "_self.masked_weight: ")
        # print(self.masked_weight)
        # print(self.name + "_self.masked_filters: ")
        # print(self.masked_filters)
        self.kernel = torch.einsum("ijk, ijklm -> ijlm", self.masked_weight, self.masked_filters)
        return F.conv2d(
            input=x,
            weight=self.kernel,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.group)


class Conv1dsRF(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            nrsffpc,
            ffp,
            padding,
            stride,
            dilation,
            group):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        :param ffp: stands for filters_path, absolute path to filters numpy array.
        It must be a numpy array of shape = (width, height, num_filters)
        :param nrsffpc: number of randomly selection fixed filters per channel (either out_channel or in_channel)
        """
        super(Conv1dsRF, self).__init__()

        self.out_channels = out_channels
        self.in_channels = in_channels
        self.nrsffpc = nrsffpc
        self.padding = padding
        self.dilation = dilation
        self.stride = stride
        self.group = group

        self.ffp = ffp
        self.filters = np.load(self.ffp)
        # print(self.filters.shape)
        assert self.filters.ndim == 4
        self.kernel_size = (self.filters.shape[0], self.filters.shape[1], self.filters.shape[2])

        rsffpc = torch.Tensor(self.build_fixed_kernels_1dspectral(
            self.filters,
            self.nrsffpc,
            self.out_channels,
            self.in_channels))
        self.register_buffer("rsffpc", rsffpc)

        # self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.nrsffpc, 1, 1))
        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.nrsffpc))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        stdv = 1. / math.sqrt(in_channels)
        self.bias.data.uniform_(-stdv, stdv)

    def build_fixed_kernels_1dspectral(self, filters, nrsffpc, out_channels, in_channels):
        """
        :param filters:  np.array of fixed kernels;
        must be of shape [filters_height, filters_width, num_filters]
        :param nrsffpc: number of randomly selected fixed filters per channel
        :param in_channels: number of input channels in the tf.nn.conv2d
        :param out_channels: number of output channels in the tf.nn.conv2d
        :return:
        """
        assert filters.ndim == 4
        d = filters.shape[0]  # which is equal to 3 or 7 or 9 or 11
        w = filters.shape[1]  # which is equal to 1
        h = filters.shape[2]  # which is equal to 1
        nff = filters.shape[3]  # number of fixed filters
        channels = np.zeros((out_channels, in_channels, d, h, w, nrsffpc), dtype=np.float32)
        for k in range(out_channels):
            for j in range(in_channels):
                channels[k, j] = filters[:, :, :, sample(range(0, nff), nrsffpc)]
        channels = np.transpose(channels, (0, 1, 5, 2, 3, 4))
        return channels

    def forward(self, x):
        """
        In the forward function we accept a torch.Tensor of input data and we must return
        a torch.Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on torch.Tensors.
        """
        # self.kernels = sum(torch.mul(self.weight, self.filters), dim=2, keepdim=False)
        self.kernel = torch.einsum("ijk, ijklmn -> ijlmn", self.weight, self.rsffpc)
        # print("self.kernel: ")
        # print(self.kernel.size())
        return F.conv3d(
            input=x,
            weight=self.kernel,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.group)