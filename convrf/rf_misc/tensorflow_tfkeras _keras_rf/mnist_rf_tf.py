from typing import Any, Tuple, Generator

import tensorflow as tf
import numpy as np
from random import sample


def build_fixed_kernels(fixed_filters: np.ndarray, nrsfkpc: int, c_in: int, c_out: int) -> np.ndarray:
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


def conv2d_rf(name, input, fixed_filters, nrsfkpc, c_in, c_out):
    kernel_initializer = tf.keras.initializers.glorot_uniform(seed=None)
    kernel_regularizer = tf.keras.regularizers.l2(1.e-4)
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    data_format = 'NHWC'
    bias_shape = [c_out]
    bias_initializer = tf.keras.initializers.glorot_uniform(seed=None)

    fixed_kernels = build_fixed_kernels(
        fixed_filters=fixed_filters,
        nrsfkpc=nrsfkpc,
        c_in=c_in,
        c_out=c_out)
    weights = tf.get_variable(
        name=name+'w',
        shape=[nrsfkpc, c_in, c_out],
        initializer=kernel_initializer,
        regularizer=None,
        trainable=True,
    )
    kernel = tf.einsum('ijklm,klm->ijlm', fixed_kernels, weights)
    bias = tf.get_variable(
            name=name+'b',
            shape=bias_shape,
            initializer=bias_initializer,
            trainable=True,)

    return kernel, tf.nn.conv2d(
        input=input,
        filter=kernel,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name) + bias


def data_iterator(images: np.array, labels: np.array, batch_size: int) -> (np.array, np.array):
    """ A simple data iterator """
    n = images.shape[0]
    # batch_idx = 0
    while True:
        shuf_idxs = np.random.permutation(n).reshape((1, n))[0]
        shuf_images = images[shuf_idxs]
        shuf_labels = labels[shuf_idxs]

        for batch_idx in range(0, n, batch_size):
            # print(shuf_idxs[batch_idx: batch_idx + batch_size])
            batch_images = shuf_images[batch_idx: batch_idx + batch_size]
            batch_labels = shuf_labels[batch_idx: batch_idx + batch_size]
            # print(batch_images.shape)
            # print(batch_labels.shape)
            yield batch_images, batch_labels


def get_batch_accuracy(
        x,
        y,
        batch_images,
        batch_labels,
        sess,
        running_vars_initializer,
        tf_metric,
        tf_metric_update,
        step):
    # Reset the running variables
    sess.run(running_vars_initializer)
    # Update the running variables on new batch of samples and calculate the score on this batch
    feed_dict = {x: batch_images, y: batch_labels}
    sess.run(tf_metric_update, feed_dict=feed_dict)
    # Calculate the score on this batch
    score = sess.run(tf_metric)
    print("step {} score: {}".format(step, score))


def get_overall_accuracy(x, y, gen, N, batch_size, sess, running_vars_initializer, tf_metric, tf_metric_update):
    # initialize/reset the running variables
    sess.run(running_vars_initializer)

    if N % batch_size != 0:
        n_batches = N // batch_size + 1
    else:
        n_batches = N // batch_size

    for _ in range(n_batches):
        image, label = next(gen)
        # Update the running variables on new batch of samples
        feed_dict = {x: image, y: label}
        sess.run(tf_metric_update, feed_dict=feed_dict)

    # Calculate the score
    score = sess.run(tf_metric)
    print("score: {}".format(score))


# load dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
# assuming channels_last
x_train = np.float32(np.expand_dims(x_train, axis=-1))
x_test = np.float32(np.expand_dims(x_test, axis=-1))
print(x_train.dtype)
y_train = np.int32(y_train)
y_test = np.int32(y_test)
print(y_train.shape)

# define an instance of generator object
max_steps = 2000
batch_size = 1024
jen = data_iterator(images=x_train, labels=y_train, batch_size=batch_size)
jen_test = data_iterator(images=x_test, labels=y_test, batch_size=batch_size)
N = len(x_test)


# load fixed filters
ff = np.load('/home/kazem/PycharmProjects/nikos_sparse_filters/5x5.npy')

# define graph
x = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1))
y = tf.placeholder(dtype=tf.int32, shape=(None,))

kernel_1, out = conv2d_rf(name='conv1', input=x, fixed_filters=ff, nrsfkpc=5, c_in=1, c_out=32)
print(out.shape)
out = tf.layers.batch_normalization(out, axis=-1)
out = tf.nn.relu(out)
out = tf.layers.max_pooling2d(out, pool_size=[2, 2], strides=[2, 2], data_format='channels_last')
print(out.shape)
kernel_2, out = conv2d_rf(name='conv2', input=out, fixed_filters=ff, nrsfkpc=5, c_in=32, c_out=64)
out = tf.layers.batch_normalization(out, axis=-1)
out = tf.nn.relu(out)
out = tf.layers.max_pooling2d(out, pool_size=[2, 2], strides=[2, 2], data_format='channels_last')
print(out.shape)
out = tf.contrib.layers.flatten(out)
logits = tf.layers.dense(out, units=10)
print(logits.shape)

# cost function, optimizer and train_op
normal_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits, name='loss')
# reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
# reg_loss = reg_losses[0] + reg_losses[1]
# loss = normal_loss + reg_loss
optimizer = tf.train.AdamOptimizer(learning_rate=.001)
extra_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_ops):
    train_op = optimizer.minimize(normal_loss)

# Define the metric and update operations
predictions = tf.argmax(logits, axis=1)
tf_metric, tf_metric_update = tf.metrics.accuracy(y, predictions, name="my_metric")
# Isolate the variables stored behind the scenes by the metric operation
running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")
# Define initializer to initialize/reset running variables
running_vars_initializer = tf.variables_initializer(var_list=running_vars)


# define session
sess = tf.Session()

# training
init = tf.global_variables_initializer()
sess.run(init)
for i in range(max_steps + 1):
    batch_images, batch_labels = next(jen)

    if i % 10 == 0:
        # print(sess.run(kernel_1, feed_dict={x: batch_images, y: batch_labels}))
        # print(2*'\n')
        get_batch_accuracy(
            x,
            y,
            batch_images,
            batch_labels,
            sess,
            running_vars_initializer,
            tf_metric,
            tf_metric_update,
            step=i)
    sess.run(train_op, feed_dict={x: batch_images, y: batch_labels})

get_overall_accuracy(x, y, jen_test, N, batch_size, sess, running_vars_initializer, tf_metric, tf_metric_update)
