import numpy as np
from os.path import join
import tensorflow as tf
from random import sample
from tensorflow.keras import backend as K
print(tf.__version__)


class Conv2dRF(tf.keras.layers.Layer):
    def __init__(self,
                 op_name,
                 fflp,  # fixed filters load path
                 kernel_size,
                 nrsfkpc,  # number of randomly selected filters per channel
                 c_in,
                 c_out,
                 kernel_initializer=tf.keras.initializers.glorot_uniform(seed=None),
                 kernel_regularizer=tf.keras.regularizers.l2(1.e-4),
                 strides=(1, 1, 1, 1),
                 padding='SAME',
                 data_format='channels_last',
                 bias_initializer=tf.zeros_initializer()):
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
        self.coeff_matrix = self.add_variable(
            name='w',
            shape=[self.nrsfkpc, self.c_in, self.c_out],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            trainable=True,
        )
        self.kernel = tf.einsum('ijklm,klm->ijlm', self.fixed_kernels, self.coeff_matrix)

        if self.data_format == 'NHWC':
            bias_shape = [self.c_out]
        else:
            bias_shape = [self.c_out, 1, 1]
        self.bias = self.add_variable(
            name='b',
            shape=bias_shape,
            initializer=self.bias_initializer,
            trainable=True,)

    def call(self, input):
        return tf.nn.conv2d(
            input=input,
            filter=self.kernel,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            name='conv') + self.bias


def main(_):
    # hyper-parameters
    # C_IN = 1        # number of input channels
    # C_OUT = 32    # number of output channels
    # NRSFKPC = 1  # number of randomly selected filters per channel
    N1 = 64
    N2 = 32
    data_format = 'channels_first'

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    #                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print(train_images.shape)
    print(test_images.shape)

    if data_format == 'channels_last':
        train_images = np.expand_dims(train_images, axis=-1)
        test_images = np.expand_dims(test_images, axis=-1)
    else:
        train_images = np.expand_dims(train_images, axis=1)
        test_images = np.expand_dims(test_images, axis=1)

    print("train_images.shape: ", train_images.shape)
    print("test_images.shape: ", test_images.shape)
    # normalize
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # note: kernel_size must match the size of the kernel loaded from fflp
    conv1 = Conv2dRF(op_name='RF1',
                     fflp='nikos_filters/3x3.npy',  # fixed_filters_load_ath
                     kernel_size=(3, 3),
                     nrsfkpc=2,  # number of randomly selected fixed_kernels_per_channel
                     c_in=1,
                     c_out=N1,
                     data_format=data_format
                     )
    conv2 = Conv2dRF(op_name='RF2',
                     fflp='nikos_filters/3x3.npy',
                     kernel_size=(3, 3),
                     nrsfkpc=2,
                     c_in=N1,
                     c_out=N2,
                     data_format=data_format
                     )

    # Must define the input shape in the first layer of the neural network
    x = tf.keras.Input(shape=train_images.shape[1:])
    out = conv1(x)
    out = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format=data_format)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = conv2(out)
    out = tf.keras.layers.MaxPooling2D(pool_size=2, data_format=data_format)(out)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(256, activation='relu')(out)
    out = tf.keras.layers.Dropout(0.5)(out)
    out = tf.keras.layers.Dense(10, activation='softmax')(out)

    model = tf.keras.Model(inputs=x, outputs=out)
    # Take a look at the model summary
    model.summary()

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=2)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
    # predictions = model.predict(test_images)
    # print(np.argmax(predictions[0]))

    model.save_weights('w.h5')
    model.load_weights('w.h5')
    print("model weights loaded!")

    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    print('model complied!')
    # print(model.summary())
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
    K.clear_session()
    import gc
    gc.collect()


if __name__ == '__main__':
    tf.app.run(main=main)
