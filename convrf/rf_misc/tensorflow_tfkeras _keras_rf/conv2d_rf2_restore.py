from __future__ import print_function
import keras
from keras import backend as K
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, MaxPooling2D, BatchNormalization, Dense, Dropout, Flatten
from tf_keras.conv2d_rf import Conv2dRF


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


# note: kernel_size must match the size of the kernel loaded from fflp
conv1 = Conv2dRF(op_name='RF1',
                 fflp='nikos_filters/3x3.npy',  # fixed_filters_load_ath
                 kernel_size=(3, 3),
                 nrsfkpc=2,  # number of randomly selected fixed_kernels_per_channel
                 c_in=1,
                 c_out=32,
                 )
conv2 = Conv2dRF(op_name='RF2',
                 fflp='nikos_filters/3x3.npy',
                 kernel_size=(3, 3),
                 nrsfkpc=2,
                 c_in=32,
                 c_out=64,
                 )

# Must define the input shape in the first layer of the neural network
x = Input(shape=(28, 28, 1))
out = conv1(x)
out = MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(out)
out = BatchNormalization()(out)
out = conv2(out)
out = MaxPooling2D(pool_size=2, data_format="channels_last")(out)
out = BatchNormalization()(out)
out = Flatten()(out)
out = Dense(256, activation='relu')(out)
out = Dropout(0.5)(out)
out = Dense(10, activation='softmax')(out)

model = Model(inputs=x, outputs=out)

model.load_weights('w.h5')
print("model weights loaded!")
model.compile(
    optimizer=keras.optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
# print('model complied!')
# print(model.summary())
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)