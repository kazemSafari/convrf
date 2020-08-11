from __future__ import print_function

import tensorflow as tf
import numpy as np
from random import sample


import keras
from tensorflow.keras import backend as K
from keras import regularizers
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Layer, Input, Conv2D, MaxPooling2D, BatchNormalization, Dense, Dropout, Flatten


class Conv2dRF(Layer):
    def __init__(self,
                 op_name,
                 fflp, # fixed filters load path
                 kernel_size,
                 nrsfkpc, # number of randomly selected filters per channel
                 c_in,
                 c_out,
                 kernel_initializer=keras.initializers.glorot_uniform(seed=None),
                 kernel_regularizer=regularizers.l2(1.e-4),
                 strides=(1, 1, 1, 1),
                 padding='SAME',
                 data_format='channels_last',
                 bias_initializer=keras.initializers.Zeros()):
        super(Conv2dRF, self).__init__(name=op_name)
        self.op_name = op_name
        self.fflp = fflp
        self.fixed_filters = np.load(self.fflp)
        self.kernel_size = kernel_size
        assert kernel_size[0] == np.shape(self.fixed_filters)[0] and kernel_size[1] == np.shape(self.fixed_filters)[1]
        self.nrsfkpc = nrsfkpc  # number of randomly selected fixed kernels per channel
        self.c_in = c_in
        self.c_out = c_out

        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.strides = strides
        self.padding = padding

        if data_format == 'channels_last':
            self.data_format = 'NHWC'
        else:
            self.data_format = 'NCHW'

        self.bias_initializer = bias_initializer

    def build_fixed_kernels(self, fixed_filters: np.ndarray, nrsfkpc: int, c_in: int, c_out: int) -> np.ndarray:
        """
        :param fixed_filters:  np.array of fixed kernels;
        must be of shape [filters_height, filters_width, num_filters]
        :param nrsfkpc: number of randomly selected fixed kernels per channel
        :param c_in: number of input channels in the tf.nn.conv2d
        :param c_out: number of output channels in the tf.nn.conv2d
        :return:
        """
        assert fixed_filters.ndim == 3  # dimension of fixed filters
        h = fixed_filters.shape[0]  # height of fixed filters
        w = fixed_filters.shape[1]  # width of fixed filters
        nff = fixed_filters.shape[2]  # number of fixed filters
        channels = np.zeros((c_out, c_in, h, w, nrsfkpc))
        for k in range(c_out):
            for j in range(c_in):
                channels[k, j] = fixed_filters[:, :, sample(range(0, nff), nrsfkpc)]
        channels = np.float32(np.transpose(channels, (2, 3, 4, 1, 0)))
        channels = tf.convert_to_tensor(channels)
        return channels

    def build(self, input_shape):
        self.fixed_kernels = self.build_fixed_kernels(
            fixed_filters=self.fixed_filters,
            nrsfkpc=self.nrsfkpc,
            c_in=self.c_in,
            c_out=self.c_out)
        self.coeff_matrix = self.add_weight(
            name='w',
            shape=[self.nrsfkpc, self.c_in, self.c_out],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        self.kernel = tf.einsum('ijklm,klm->ijlm', self.fixed_kernels, self.coeff_matrix)
        self.bias = self.add_weight(
            name='b',
            shape=[self.c_out],
            initializer=self.bias_initializer,
            trainable=True,)
        super(Conv2dRF, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        # print("self.data_format: ", self.data_format)
        output = tf.nn.conv2d(
            input=x,
            filter=self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            name='conv')  # + self.bias
        self.output_shape_ = K.int_shape(output)
        return output

    def compute_output_shape(self, input_shape):
        print(self.output_shape_)
        print(type(self.output_shape_))
        print(type(self.output_shape_[0]))
        return self.output_shape_

batch_size = 128
num_classes = 10
epochs = 2

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


print("K.image_data_format(): ", K.image_data_format())
# note: kernel_size must match the size of the kernel loaded from fflp
conv1 = Conv2dRF(op_name='RF1',
                 fflp='nikos_filters/3x3.npy',  # fixed_filters_load_ath
                 kernel_size=(3, 3),
                 nrsfkpc=2,  # number of randomly selected fixed_kernels_per_channel
                 c_in=1,
                 c_out=32,
                 data_format=K.image_data_format()
                 )
conv2 = Conv2dRF(op_name='RF2',
                 fflp='nikos_filters/3x3.npy',
                 kernel_size=(3, 3),
                 nrsfkpc=2,
                 c_in=32,
                 c_out=64,
                 data_format=K.image_data_format()
                 )

# Must define the input shape in the first layer of the neural network
x = Input(shape=x_train.shape[1:])
print(x)
out = conv1(x)
out = MaxPooling2D(pool_size=(2, 2), data_format=K.image_data_format())(out)
out = BatchNormalization()(out)
out = conv2(out)
out = MaxPooling2D(pool_size=(2, 2), data_format=K.image_data_format())(out)
out = BatchNormalization()(out)
out = Flatten()(out)
out = Dense(256, activation='relu')(out)
out = Dropout(0.5)(out)
print("out.shape: ", out.shape)
out = Dense(10, activation='softmax')(out)
print("out.shape: ", out.shape)

model = Model(inputs=x, outputs=out)
# Take a look at the model summary
# model.summary()

model.compile(optimizer=keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=2)
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

model.save_weights('w.h5')
print("model weights loaded!")