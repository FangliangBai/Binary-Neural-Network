#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import time


# return the value after hard sigmoid
def hard_sigmoid(x):
    return tf.contrib.keras.backend.clip((x+1.0)/2.0, 0.0, 1.0)


    # binarize the weight
def binarization(W, H=1, binary=False, stochastic=False):
    if not binary:
        Wb = W
    else:
        Wb = hard_sigmoid(W / H)
        if stochastic:
            # use hard sigmoid weight for possibility
            Wb = tf.contrib.keras.backend.random_binomial(tf.shape(Wb), p=Wb)
        else:
            # round weight to 0 and 1
            Wb = tf.round(Wb)
        # change range from 0~1  to  -1~1
        Wb = Wb*2-1
    return Wb


# shuffle data after one epoch
class data_generator(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.num_samples = x.shape[0]

    def data_gen(self, batch_size):
        x = self.x
        y = self.y
        num_batch = self.num_samples//batch_size
        batch_count = 0
        while 1:
            if batch_count < num_batch:
                a = batch_count*batch_size
                b = (batch_count+1)*batch_size
                batch_count += 1
                yield x[a:b, :], y[a:b]
            else:
                batch_count = 0
                mask = np.arange(self.num_samples)
                np.random.shuffle(mask)
                x = x[mask]
                y = y[mask]


class conv_layer(object):
    def __init__(self, input_x, in_channel, out_channel, kernel_shape, padding="SAME", binary=False, stochastic=False, is_training=None, index=0):
        # binary: whether to implement the Binary Connect
        # stochastic: whether implement stochastic weight if do Binary Connect
        
        assert len(input_x.shape) == 4 and input_x.shape[1] == input_x.shape[2] and input_x.shape[3] == in_channel

        with tf.variable_scope('conv_layer_%d' % index):
            with tf.name_scope('conv_kernel'):
                w_shape = [kernel_shape, kernel_shape, in_channel, out_channel]
                # the real value of weight
                weight = tf.get_variable(name='conv_kernel_%d' % index, shape=w_shape, initializer=tf.glorot_uniform_initializer())
                self.weight = weight

            with tf.variable_scope('conv_bias'):
                b_shape = [out_channel]
                bias = tf.get_variable(name='conv_bias_%d' % index, shape=b_shape, initializer=tf.constant_initializer(0.))
                self.bias = bias

            # use real value weights to test in stochastic BinaryConnect
            if binary and stochastic:
                wb = tf.cond(is_training, lambda: binarization(weight, H=1, binary=binary, stochastic=stochastic), lambda:weight)
            # otherwise, return binarization directly
            else:
                wb = binarization(weight, H=1, binary=binary, stochastic=stochastic)
                
            self.wb = wb

            cell_out = tf.nn.conv2d(input_x, self.wb, strides=[1, 1, 1, 1], padding=padding)

            cell_out = tf.add(cell_out, bias)

            self.cell_out = cell_out

            tf.summary.histogram('conv_layer/{}/kernel'.format(index), self.weight)
            tf.summary.histogram('conv_layer/{}/bias'.format(index), self.bias)

            # to store the moments for adam
            with tf.name_scope('conv_moment'):
                self.m_w = tf.get_variable(name='conv_first_moment_w_%d' % index, shape=w_shape, initializer=tf.constant_initializer(0.))
                self.v_w = tf.get_variable(name='conv_second_moment_w_%d' % index, shape=w_shape, initializer=tf.constant_initializer(0.))
                self.m_b = tf.get_variable(name='conv_first_moment_b_%d' % index, shape=b_shape, initializer=tf.constant_initializer(0.))
                self.v_b = tf.get_variable(name='conv_second_moment_b_%d' % index, shape=b_shape, initializer=tf.constant_initializer(0.))

    def output(self):
        return self.cell_out


class max_pooling_layer(object):
    def __init__(self, input_x, k_size, padding="SAME"):
        with tf.variable_scope('max_pooling'):
            # strides [1, k_size, k_size, 1]
            pooling_shape = [1, k_size, k_size, 1]
            cell_out = tf.nn.max_pool(input_x, strides=pooling_shape, ksize=pooling_shape, padding=padding)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class norm_layer(object):
    def __init__(self, input_x, is_training, is_drop_out, activation_function, index=0):
        with tf.variable_scope('batch_norm_%d' % index):
            cell_out = tf.contrib.layers.batch_norm(input_x, decay=0.99, updates_collections=None, is_training=is_training, epsilon=1e-4)

            # the activation function is after the batch norm
            if activation_function is not None:
                cell_out = activation_function(cell_out)

            # no dropout for CIFAR
            if is_drop_out:
                cell_out = tf.layers.dropout(cell_out, rate=0.0, training=is_training)

            self.cell_out = cell_out

    def output(self):
        return self.cell_out


class fc_layer(object):
    def __init__(self, input_x, in_size, out_size, binary=False, stochastic=False, is_training=None, index=0):
        # binary: whether to implement the Binary Connect
        # stochastic: whether implement stochastic weight if do Binary Connect
        
        with tf.variable_scope('fc_layer_%d' % index):
            with tf.name_scope('fc_kernel'):
                w_shape = [in_size, out_size]
                # the real value of weight
                weight = tf.get_variable(name='fc_kernel_%d' % index, shape=w_shape, initializer=tf.glorot_uniform_initializer())
                self.weight = weight

            with tf.variable_scope('fc_kernel'):
                b_shape = [out_size]
                bias = tf.get_variable(name='fc_bias_%d' % index, shape=b_shape, initializer=tf.constant_initializer(0.))
                self.bias = bias

            # use real value weights to test in stochastic BinaryConnect
            if binary and stochastic:
                wb = tf.cond(is_training, lambda: binarization(weight, H=1, binary=binary, stochastic=stochastic), lambda: weight)
            # otherwise, return binarization directly
            else:
                wb = binarization(weight, H=1, binary=binary, stochastic=stochastic)
                
            self.wb = wb

            cell_out = tf.add(tf.matmul(input_x, self.wb), bias)

            self.cell_out = cell_out

            tf.summary.histogram('fc_layer/{}/kernel'.format(index), self.weight)
            tf.summary.histogram('fc_layer/{}/bias'.format(index), self.bias)

            # to store the moments for adam
            with tf.name_scope('fc_moment'):
                self.m_w = tf.get_variable(name='fc_first_moment_w_%d' % index, shape=w_shape, initializer=tf.constant_initializer(0.))
                self.v_w = tf.get_variable(name='fc_second_moment_w_%d' % index, shape=w_shape, initializer=tf.constant_initializer(0.))
                self.m_b = tf.get_variable(name='fc_first_moment_b_%d' % index, shape=b_shape, initializer=tf.constant_initializer(0.))
                self.v_b = tf.get_variable(name='fc_second_moment_b_%d' % index, shape=b_shape, initializer=tf.constant_initializer(0.))

    def output(self):
        return self.cell_out


def Network(input_x, input_y, is_training, is_drop_out, is_binary, is_stochastic, channel_num, output_size,
            conv_featmap, fc_units, conv_kernel_size, pooling_size, learning_rate):
    # is_training: whether train the network or validate it
    # is_drop_out: whether to dropout during training
    # is_binary: whether use the Binary Connect or not
    # is_stochastic: if use Binary Connect, whether to be stochastic
    # channel_num: input channel number, =3
    # output_size: 10 class
    # conv_featmap: number of features for convolution layer
    # fc_units: number of units for full connect layer
    # conv_kernel_size: kernel size for convolution layer
    # pooling_size: pooling size for max pooling layer
    # learning_rate: used for optimization

    # here is the architecture of the network
    # 128Conv3-BN-128Conv3-MaxPool2-BN-256Conv3-BN-256Conv3-MaxPool2-BN-512Conv3-BN-512Conv3-MaxPool2-BN-1024fc-1024fc-10fc
    
    conv_layer_0 = conv_layer(input_x=input_x,
                              in_channel=channel_num,
                              out_channel=conv_featmap[0],
                              kernel_shape=conv_kernel_size[0],
                              padding='VALID',
                              binary=is_binary,
                              stochastic=is_stochastic,
                              is_training=is_training,
                              index=0)

    norm_layer_0 = norm_layer(input_x=conv_layer_0.output(),
                              is_training=is_training,
                              is_drop_out=False,
                              activation_function=tf.nn.relu,
                              index=0)

    conv_layer_1 = conv_layer(input_x=norm_layer_0.output(),
                              in_channel=conv_featmap[0],
                              out_channel=conv_featmap[1],
                              kernel_shape=conv_kernel_size[1],
                              padding='VALID',
                              binary=is_binary,
                              stochastic=is_stochastic,
                              is_training=is_training,
                              index=1)

    pooling_layer_0 = max_pooling_layer(input_x=conv_layer_1.output(),
                                        k_size=pooling_size[0],
                                        padding="SAME")

    norm_layer_1 = norm_layer(input_x=pooling_layer_0.output(),
                              is_training=is_training,
                              is_drop_out=False,
                              activation_function=tf.nn.relu,
                              index=1)

    conv_layer_2 = conv_layer(input_x=norm_layer_1.output(),
                              in_channel=conv_featmap[1],
                              out_channel=conv_featmap[2],
                              kernel_shape=conv_kernel_size[2],
                              padding='VALID',
                              binary=is_binary,
                              stochastic=is_stochastic,
                              is_training=is_training,
                              index=2)

    norm_layer_2 = norm_layer(input_x=conv_layer_2.output(),
                              is_training=is_training,
                              is_drop_out=False,
                              activation_function=tf.nn.relu,
                              index=2)

    conv_layer_3 = conv_layer(input_x=norm_layer_2.output(),
                              in_channel=conv_featmap[2],
                              out_channel=conv_featmap[3],
                              kernel_shape=conv_kernel_size[3],
                              padding='VALID',
                              binary=is_binary,
                              stochastic=is_stochastic,
                              is_training=is_training,
                              index=3)

    pooling_layer_1 = max_pooling_layer(input_x=conv_layer_3.output(),
                                        k_size=pooling_size[1],
                                        padding="SAME")

    norm_layer_3 = norm_layer(input_x=pooling_layer_1.output(),
                              is_training=is_training,
                              is_drop_out=False,
                              activation_function=tf.nn.relu,
                              index=3)

    conv_layer_4 = conv_layer(input_x=norm_layer_3.output(),
                              in_channel=conv_featmap[3],
                              out_channel=conv_featmap[4],
                              kernel_shape=conv_kernel_size[4],
                              padding='VALID',
                              binary=is_binary,
                              stochastic=is_stochastic,
                              is_training=is_training,
                              index=4)

    norm_layer_4 = norm_layer(input_x=conv_layer_4.output(),
                              is_training=is_training,
                              is_drop_out=False,
                              activation_function=tf.nn.relu,
                              index=4)

    conv_layer_5 = conv_layer(input_x=norm_layer_4.output(),
                              in_channel=conv_featmap[4],
                              out_channel=conv_featmap[5],
                              kernel_shape=conv_kernel_size[5],
                              padding='VALID',
                              binary=is_binary,
                              stochastic=is_stochastic,
                              is_training=is_training,
                              index=5)

    pooling_layer_2 = max_pooling_layer(input_x=conv_layer_5.output(),
                                        k_size=pooling_size[2],
                                        padding="SAME")

    norm_layer_5 = norm_layer(input_x=pooling_layer_2.output(),
                              is_training=is_training,
                              is_drop_out=False,
                              activation_function=tf.nn.relu,
                              index=5)

    # flatten the output of convolution layer
    pool_shape = norm_layer_5.output().get_shape()
    img_vector_length = pool_shape[1].value * pool_shape[2].value * pool_shape[3].value
    flatten = tf.reshape(norm_layer_5.output(), shape=[-1, img_vector_length])

    fc_layer_0 = fc_layer(input_x=flatten,
                          in_size=img_vector_length,
                          out_size=fc_units[0],
                          binary=is_binary,
                          stochastic=is_stochastic,
                          is_training=is_training,
                          index=0)

    norm_layer_6 = norm_layer(input_x=fc_layer_0.output(),
                              is_training=is_training,
                              is_drop_out=is_drop_out,
                              activation_function=tf.nn.relu,
                              index=6)

    fc_layer_1 = fc_layer(input_x=norm_layer_6.output(),
                          in_size=fc_units[0],
                          out_size=fc_units[1],
                          binary=is_binary,
                          stochastic=is_stochastic,
                          is_training=is_training,
                          index=1)

    norm_layer_7 = norm_layer(input_x=fc_layer_1.output(),
                              is_training=is_training,
                              is_drop_out=is_drop_out,
                              activation_function=tf.nn.relu,
                              index=7)

    fc_layer_2 = fc_layer(input_x=norm_layer_7.output(),
                          in_size=fc_units[1],
                          out_size=output_size,
                          binary=is_binary,
                          stochastic=is_stochastic,
                          is_training=is_training,
                          index=2)

    norm_layer_8 = norm_layer(input_x=fc_layer_2.output(),
                              is_training=is_training,
                              is_drop_out=False,
                              activation_function=None,
                              index=8)

    # compute loss
    with tf.name_scope("loss"):
        net_output = norm_layer_8.output()
        label = tf.one_hot(input_y, output_size)
        # the hinge square loss
        loss = tf.reduce_mean(tf.square(tf.losses.hinge_loss(label, net_output)))
        tf.summary.scalar('loss', loss)

    # update parameters with adam
    with tf.name_scope("Adam_optimize"):
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-08

        # time step
        t = tf.get_variable(name='timestep', shape=[], initializer=tf.constant_initializer(0))

        # function that return all the updates
        def true_fn(loss=loss, conv_layer_0=conv_layer_0, conv_layer_1=conv_layer_1, conv_layer_2=conv_layer_2,
                    conv_layer_3=conv_layer_3, conv_layer_4=conv_layer_4, conv_layer_5=conv_layer_5, fc_layer_0=fc_layer_0,
                    fc_layer_2=fc_layer_2, t=t):

            new_t = t.assign(t + 1)

            # calculate gradients
            grad_conv_wb0, grad_conv_wb1, grad_conv_wb2, grad_conv_wb3, grad_conv_wb4, grad_conv_wb5 \
                = tf.gradients(ys=loss, xs=[conv_layer_0.wb, conv_layer_1.wb, conv_layer_2.wb, conv_layer_3.wb,
                                            conv_layer_4.wb, conv_layer_5.wb])
            grad_fc_wb0, grad_fc_wb1, grad_fc_wb2 \
                = tf.gradients(ys=loss, xs=[fc_layer_0.wb, fc_layer_1.wb, fc_layer_2.wb])
            grad_conv_b0, grad_conv_b1, grad_conv_b2, grad_conv_b3, grad_conv_b4, grad_conv_b5 \
                = tf.gradients(ys=loss, xs=[conv_layer_0.bias, conv_layer_1.bias, conv_layer_2.bias, conv_layer_3.bias,
                                            conv_layer_4.bias, conv_layer_5.bias])
            grad_fc_b0, grad_fc_b1, grad_fc_b2 \
                = tf.gradients(ys=loss, xs=[fc_layer_0.bias, fc_layer_1.bias, fc_layer_2.bias])

            # calculate updates for conv_layer_0
            new_conv_m_wb0 = conv_layer_0.m_w.assign(beta1 * conv_layer_0.m_w + (1. - beta1) * grad_conv_wb0)
            new_conv_v_wb0 = conv_layer_0.v_w.assign(beta2 * conv_layer_0.v_w + (1. - beta2) * grad_conv_wb0 ** 2)
            new_conv_m_b0 = conv_layer_0.m_b.assign(beta1 * conv_layer_0.m_b + (1. - beta1) * grad_conv_b0)
            new_conv_v_b0 = conv_layer_0.v_b.assign(beta2 * conv_layer_0.v_b + (1. - beta2) * grad_conv_b0 ** 2)
            update_conv_wb0 = new_conv_m_wb0 / (tf.sqrt(new_conv_v_wb0) + epsilon)
            update_conv_b0 = new_conv_m_b0 / (tf.sqrt(new_conv_v_b0) + epsilon)

            # calculate updates for conv_layer_1
            new_conv_m_wb1 = conv_layer_1.m_w.assign(beta1 * conv_layer_1.m_w + (1. - beta1) * grad_conv_wb1)
            new_conv_v_wb1 = conv_layer_1.v_w.assign(beta2 * conv_layer_1.v_w + (1. - beta2) * grad_conv_wb1 ** 2)
            new_conv_m_b1 = conv_layer_1.m_b.assign(beta1 * conv_layer_1.m_b + (1. - beta1) * grad_conv_b1)
            new_conv_v_b1 = conv_layer_1.v_b.assign(beta2 * conv_layer_1.v_b + (1. - beta2) * grad_conv_b1 ** 2)
            update_conv_wb1 = new_conv_m_wb1 / (tf.sqrt(new_conv_v_wb1) + epsilon)
            update_conv_b1 = new_conv_m_b1 / (tf.sqrt(new_conv_v_b1) + epsilon)

            # calculate updates for conv_layer_2
            new_conv_m_wb2 = conv_layer_2.m_w.assign(beta1 * conv_layer_2.m_w + (1. - beta1) * grad_conv_wb2)
            new_conv_v_wb2 = conv_layer_2.v_w.assign(beta2 * conv_layer_2.v_w + (1. - beta2) * grad_conv_wb2 ** 2)
            new_conv_m_b2 = conv_layer_2.m_b.assign(beta1 * conv_layer_2.m_b + (1. - beta1) * grad_conv_b2)
            new_conv_v_b2 = conv_layer_2.v_b.assign(beta2 * conv_layer_2.v_b + (1. - beta2) * grad_conv_b2 ** 2)
            update_conv_wb2 = new_conv_m_wb2 / (tf.sqrt(new_conv_v_wb2) + epsilon)
            update_conv_b2 = new_conv_m_b2 / (tf.sqrt(new_conv_v_b2) + epsilon)

            # calculate updates for conv_layer_3
            new_conv_m_wb3 = conv_layer_3.m_w.assign(beta1 * conv_layer_3.m_w + (1. - beta1) * grad_conv_wb3)
            new_conv_v_wb3 = conv_layer_3.v_w.assign(beta2 * conv_layer_3.v_w + (1. - beta2) * grad_conv_wb3 ** 2)
            new_conv_m_b3 = conv_layer_3.m_b.assign(beta1 * conv_layer_3.m_b + (1. - beta1) * grad_conv_b3)
            new_conv_v_b3 = conv_layer_3.v_b.assign(beta2 * conv_layer_3.v_b + (1. - beta2) * grad_conv_b3 ** 2)
            update_conv_wb3 = new_conv_m_wb3 / (tf.sqrt(new_conv_v_wb3) + epsilon)
            update_conv_b3 = new_conv_m_b3 / (tf.sqrt(new_conv_v_b3) + epsilon)

            # calculate updates for conv_layer_4
            new_conv_m_wb4 = conv_layer_4.m_w.assign(beta1 * conv_layer_4.m_w + (1. - beta1) * grad_conv_wb4)
            new_conv_v_wb4 = conv_layer_4.v_w.assign(beta2 * conv_layer_4.v_w + (1. - beta2) * grad_conv_wb4 ** 2)
            new_conv_m_b4 = conv_layer_4.m_b.assign(beta1 * conv_layer_4.m_b + (1. - beta1) * grad_conv_b4)
            new_conv_v_b4 = conv_layer_4.v_b.assign(beta2 * conv_layer_4.v_b + (1. - beta2) * grad_conv_b4 ** 2)
            update_conv_wb4 = new_conv_m_wb4 / (tf.sqrt(new_conv_v_wb4) + epsilon)
            update_conv_b4 = new_conv_m_b4 / (tf.sqrt(new_conv_v_b4) + epsilon)

            # calculate updates for conv_layer_5
            new_conv_m_wb5 = conv_layer_5.m_w.assign(beta1 * conv_layer_5.m_w + (1. - beta1) * grad_conv_wb5)
            new_conv_v_wb5 = conv_layer_5.v_w.assign(beta2 * conv_layer_5.v_w + (1. - beta2) * grad_conv_wb5 ** 2)
            new_conv_m_b5 = conv_layer_5.m_b.assign(beta1 * conv_layer_5.m_b + (1. - beta1) * grad_conv_b5)
            new_conv_v_b5 = conv_layer_5.v_b.assign(beta2 * conv_layer_5.v_b + (1. - beta2) * grad_conv_b5 ** 2)
            update_conv_wb5 = new_conv_m_wb5 / (tf.sqrt(new_conv_v_wb5) + epsilon)
            update_conv_b5 = new_conv_m_b5 / (tf.sqrt(new_conv_v_b5) + epsilon)

            # calculate updates for fc_layer_0
            new_fc_m_wb0 = fc_layer_0.m_w.assign(beta1 * fc_layer_0.m_w + (1. - beta1) * grad_fc_wb0)
            new_fc_v_wb0 = fc_layer_0.v_w.assign(beta2 * fc_layer_0.v_w + (1. - beta2) * grad_fc_wb0 ** 2)
            new_fc_m_b0 = fc_layer_0.m_b.assign(beta1 * fc_layer_0.m_b + (1. - beta1) * grad_fc_b0)
            new_fc_v_b0 = fc_layer_0.v_b.assign(beta2 * fc_layer_0.v_b + (1. - beta2) * grad_fc_b0 ** 2)
            update_fc_wb0 = new_fc_m_wb0 / (tf.sqrt(new_fc_v_wb0) + epsilon)
            update_fc_b0 = new_fc_m_b0 / (tf.sqrt(new_fc_v_b0) + epsilon)

            # calculate updates for fc_layer_1
            new_fc_m_wb1 = fc_layer_1.m_w.assign(beta1 * fc_layer_1.m_w + (1. - beta1) * grad_fc_wb1)
            new_fc_v_wb1 = fc_layer_1.v_w.assign(beta2 * fc_layer_1.v_w + (1. - beta2) * grad_fc_wb1 ** 2)
            new_fc_m_b1 = fc_layer_1.m_b.assign(beta1 * fc_layer_1.m_b + (1. - beta1) * grad_fc_b1)
            new_fc_v_b1 = fc_layer_1.v_b.assign(beta2 * fc_layer_1.v_b + (1. - beta2) * grad_fc_b1 ** 2)
            update_fc_wb1 = new_fc_m_wb1 / (tf.sqrt(new_fc_v_wb1) + epsilon)
            update_fc_b1 = new_fc_m_b1 / (tf.sqrt(new_fc_v_b1) + epsilon)

            # calculate updates for fc_layer_2
            new_fc_m_wb2 = fc_layer_2.m_w.assign(beta1 * fc_layer_2.m_w + (1. - beta1) * grad_fc_wb2)
            new_fc_v_wb2 = fc_layer_2.v_w.assign(beta2 * fc_layer_2.v_w + (1. - beta2) * grad_fc_wb2 ** 2)
            new_fc_m_b2 = fc_layer_2.m_b.assign(beta1 * fc_layer_2.m_b + (1. - beta1) * grad_fc_b2)
            new_fc_v_b2 = fc_layer_2.v_b.assign(beta2 * fc_layer_2.v_b + (1. - beta2) * grad_fc_b2 ** 2)
            update_fc_wb2 = new_fc_m_wb2 / (tf.sqrt(new_fc_v_wb2) + epsilon)
            update_fc_b2 = new_fc_m_b2 / (tf.sqrt(new_fc_v_b2) + epsilon)

            return (update_conv_wb0, update_conv_wb1, update_conv_wb2, update_conv_wb3, update_conv_wb4, update_conv_wb5,
                    update_fc_wb0, update_fc_wb1, update_fc_wb2, update_conv_b0, update_conv_b1, update_conv_b2,
                    update_conv_b3, update_conv_b4, update_conv_b5, update_fc_b0, update_fc_b1, update_fc_b2), new_t

        # update = 0 in validation/test phase.
        def false_fn(t=t):
            return (0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.), t

        # if is_training, do update
        adam_update, new_t = tf.cond(is_training, true_fn, false_fn)

        # adjust learning rate with beta
        lr = learning_rate * tf.sqrt(1 - beta2 ** new_t) / (1 - beta1 ** new_t)

        # if is_binary, clip the weights to [-1, +1] before assignment
        if is_binary:
            new_conv_w0 = conv_layer_0.weight.assign(tf.contrib.keras.backend.clip(conv_layer_0.weight - lr * adam_update[0], -1.0, 1.0))
            new_conv_w1 = conv_layer_1.weight.assign(tf.contrib.keras.backend.clip(conv_layer_1.weight - lr * adam_update[1], -1.0, 1.0))
            new_conv_w2 = conv_layer_2.weight.assign(tf.contrib.keras.backend.clip(conv_layer_2.weight - lr * adam_update[2], -1.0, 1.0))
            new_conv_w3 = conv_layer_3.weight.assign(tf.contrib.keras.backend.clip(conv_layer_3.weight - lr * adam_update[3], -1.0, 1.0))
            new_conv_w4 = conv_layer_4.weight.assign(tf.contrib.keras.backend.clip(conv_layer_4.weight - lr * adam_update[4], -1.0, 1.0))
            new_conv_w5 = conv_layer_5.weight.assign(tf.contrib.keras.backend.clip(conv_layer_5.weight - lr * adam_update[5], -1.0, 1.0))
            new_fc_w0 = fc_layer_0.weight.assign(tf.contrib.keras.backend.clip(fc_layer_0.weight - lr * adam_update[6], -1.0, 1.0))
            new_fc_w1 = fc_layer_1.weight.assign(tf.contrib.keras.backend.clip(fc_layer_1.weight - lr * adam_update[7], -1.0, 1.0))
            new_fc_w2 = fc_layer_2.weight.assign(tf.contrib.keras.backend.clip(fc_layer_2.weight - lr * adam_update[8], -1.0, 1.0))
        else:
            new_conv_w0 = conv_layer_0.weight.assign(conv_layer_0.weight - lr * adam_update[0])
            new_conv_w1 = conv_layer_1.weight.assign(conv_layer_1.weight - lr * adam_update[1])
            new_conv_w2 = conv_layer_2.weight.assign(conv_layer_2.weight - lr * adam_update[2])
            new_conv_w3 = conv_layer_3.weight.assign(conv_layer_3.weight - lr * adam_update[3])
            new_conv_w4 = conv_layer_4.weight.assign(conv_layer_4.weight - lr * adam_update[4])
            new_conv_w5 = conv_layer_5.weight.assign(conv_layer_5.weight - lr * adam_update[5])
            new_fc_w0 = fc_layer_0.weight.assign(fc_layer_0.weight - lr * adam_update[6])
            new_fc_w1 = fc_layer_1.weight.assign(fc_layer_1.weight - lr * adam_update[7])
            new_fc_w2 = fc_layer_2.weight.assign(fc_layer_2.weight - lr * adam_update[8])
            
        new_conv_b0 = conv_layer_0.bias.assign(conv_layer_0.bias - lr * adam_update[9])
        new_conv_b1 = conv_layer_1.bias.assign(conv_layer_1.bias - lr * adam_update[10])
        new_conv_b2 = conv_layer_2.bias.assign(conv_layer_2.bias - lr * adam_update[11])
        new_conv_b3 = conv_layer_3.bias.assign(conv_layer_3.bias - lr * adam_update[12])
        new_conv_b4 = conv_layer_4.bias.assign(conv_layer_4.bias - lr * adam_update[13])
        new_conv_b5 = conv_layer_5.bias.assign(conv_layer_5.bias - lr * adam_update[14])
        new_fc_b0 = fc_layer_0.bias.assign(fc_layer_0.bias - lr * adam_update[15])
        new_fc_b1 = fc_layer_1.bias.assign(fc_layer_1.bias - lr * adam_update[16])
        new_fc_b2 = fc_layer_2.bias.assign(fc_layer_2.bias - lr * adam_update[17])

    return net_output, loss, (new_conv_w0, new_conv_w1, new_conv_w2, new_conv_w3, new_conv_w4, new_conv_w5, new_fc_w0,
                              new_fc_w1, new_fc_w2, new_conv_b0, new_conv_b1, new_conv_b2, new_conv_b3, new_conv_b4,
                              new_conv_b5, new_fc_b0, new_fc_b1, new_fc_b2)


# evaluate the output
def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('error_num', error_num)
    return error_num


def training(X_train, y_train, X_val, y_val, X_test, y_test, is_binary, is_stochastic, conv_featmap, fc_units, conv_kernel_size, pooling_size, lr_start, lr_end, epoch, batch_size, is_drop_out, verbose=False, pre_trained_model=None):
    # X_train, y_train, X_val, y_val, X_test, y_test:
    # is_binary: whether use the Binary Connect or not
    # is_stochastic: if use Binary Connect, whether to be stochastic
    # conv_featmap: number of features for convolution layer
    # fc_units: number of units for full connect layer
    # conv_kernel_size: kernel size for convolution layer
    # pooling_size: pooling size for max pooling layer
    # lr_start: init learning rate
    # lr_end: final learning rate, used for calculate lr_decay
    # epoch: times of training
    # batch_size; training batch size
    # is_drop_out: whether to dropout during training

    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64)
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')
    
    # update learning rate
    num_training = y_train.shape[0]
    num_val = y_val.shape[0]
    learning_rate = tf.Variable(lr_start, name="learning_rate", dtype=tf.float32)
    lr_decay = (lr_end / lr_start) ** (1 / epoch)
    lr_update = learning_rate.assign(tf.multiply(learning_rate, lr_decay))

    # build network
    output, loss, _update = Network(xs, ys, is_training,
                                     is_drop_out=is_drop_out,
                                     is_binary=is_binary,
                                     is_stochastic=is_stochastic,
                                     channel_num=3,
                                     output_size=10,
                                     conv_featmap=conv_featmap,
                                     fc_units=fc_units,
                                     conv_kernel_size=conv_kernel_size,
                                     pooling_size=pooling_size,
                                     learning_rate=learning_rate)

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    eve = evaluate(output, ys)
    
    # batch size for validation, since validation set is too large
    val_batch_size = 100 
    best_acc = 0
    cur_model_name = 'cifar10_{}'.format(int(time.time()))
    total_time = 0

    with tf.Session() as sess:
        merge = tf.summary.merge_all()
        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer(), {is_training: False})

        # try to restore the pre_trained
        if pre_trained_model is None:
            for epc in range(epoch):
                print("epoch {} ".format(epc + 1))

                train_eve_sum = 0
                loss_sum = 0
                for _ in range(iters + 1):
                    # randomly choose train data
                    mask = np.random.choice(num_training, batch_size, replace=False)
                    np.random.shuffle(mask)
                    train_batch_x = X_train[mask]
                    train_batch_y = y_train[mask]

                    start = time.time()
                    _, cur_loss, train_eve = sess.run([_update, loss, eve], feed_dict={xs: train_batch_x, ys: train_batch_y, is_training: True})
                    total_time += time.time()-start
                    train_eve_sum += np.sum(train_eve)
                    loss_sum += np.sum(cur_loss)

                train_acc = 100 - train_eve_sum * 100 / y_train.shape[0]
                loss_sum /= iters
                print('average train loss: {} ,  average accuracy : {}%'.format(loss_sum, train_acc))

                valid_eve_sum = 0
                for i in range(y_val.shape[0]//val_batch_size):
                    val_batch_x = X_val[i*val_batch_size:(i+1)*val_batch_size]
                    val_batch_y = y_val[i*val_batch_size:(i+1)*val_batch_size]
                    valid_eve, merge_result= sess.run([eve,merge], feed_dict={xs: val_batch_x, ys: val_batch_y, is_training: False})

                    valid_eve_sum += np.sum(valid_eve)

                valid_acc = 100 - valid_eve_sum * 100 / y_val.shape[0]

                _lr = sess.run([lr_update])
                print("update learning rate: ", _lr)

                if verbose:
                    print('validation accuracy : {}%'.format(valid_acc))

                # save the merge result summary
                writer.add_summary(merge_result, epc)

                # when achieve the best validation accuracy, we store the model paramters
                if valid_acc > best_acc:
                    print('* Best accuracy: {}%'.format(valid_acc))
                    best_acc = valid_acc
                    saver.save(sess, 'model/{}'.format(cur_model_name))
           
            # test the network
            test_eve_sum = 0
            for i in range(y_test.shape[0] // 100 + 1):
                a = i * 100
                if a >= y_test.shape[0]:
                    continue
                b = (i + 1) * 100 if (i + 1) * 100 < y_test.shape[0] else y_test.shape[0]
                test_batch_x = X_test[a:b]
                test_batch_y = y_test[a:b]              
                test_eve = sess.run([eve], feed_dict={xs: test_batch_x, ys: test_batch_y, is_training: False})

                test_eve_sum += np.sum(test_eve)

            test_acc = 100 - test_eve_sum * 100 / y_test.shape[0]
            print('test accuracy: {}%'.format(test_acc))
            
            print("Traning ends. The best valid accuracy is {}%. Model named {}.".format(best_acc, cur_model_name))
        else:
            pass
           