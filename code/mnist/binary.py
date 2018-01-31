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


class norm_layer(object):
    def __init__(self, input_x, is_training, is_drop_out, activation_function, index=0):
        with tf.variable_scope('batch_norm_%d' % index):
            cell_out = tf.contrib.layers.batch_norm(input_x, decay=0.99, updates_collections=None, is_training=is_training, epsilon=1e-4)

            # the activation function is after the batch norm
            if activation_function is not None:
                cell_out = activation_function(cell_out)

            # dropout for three layers, 0.205^3 = 0.5
            if is_drop_out:
                cell_out = tf.layers.dropout(cell_out, rate=0.205, training=is_training)

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

            #use real value weights to test in stochastic BinaryConnect
            if binary and stochastic:
                wb = tf.cond(is_training, lambda: binarization(weight, H=1, binary=binary, stochastic=stochastic), lambda:weight)
            # otherwise, return binarization directly
            else:
                wb = binarization(weight, H=1, binary=binary, stochastic=stochastic)
                
            self.wb = wb

            cell_out = tf.add(tf.matmul(input_x, self.wb), bias)

            self.cell_out = cell_out

            tf.summary.histogram('fc_layer/{}/kernel'.format(index), self.weight)
            tf.summary.histogram('fc_layer/{}/bias'.format(index), self.bias)

    def output(self):
        return self.cell_out


def Network(input_x, input_y, is_training, is_drop_out, is_binary, is_stochastic, output_size, fc_units, lr):

    # is_training: whether train the network or validate it
    # is_drop_out: whether to dropout during training
    # is_binary: whether use the Binary Connect or not
    # is_stochastic: if use Binary Connect, whether to be stochastic
    # channel_num: input channel number, =3
    # output_size: 10 class
    # fc_units: number of units for full connect layer
    # lr: learning rate, used for optimization

    # here is the architecture of the network
    # 2048fc-BN-Dropout-2048fc-BN-Dropout-2048fc-BN-Dropout-10fc
    
    fc_layer_0 = fc_layer(input_x=input_x,
                          in_size=784,
                          out_size=fc_units[0],
                          binary=is_binary,
                          stochastic=is_stochastic,
                          is_training = is_training,
                          index=0)

    norm_layer_0 = norm_layer(input_x=fc_layer_0.output(),
                              is_training=is_training,
                              is_drop_out=is_drop_out,
                              activation_function=tf.nn.relu,
                              index=0)

    fc_layer_1 = fc_layer(input_x=norm_layer_0.output(),
                          in_size=fc_units[0],
                          out_size=fc_units[1],
                          binary=is_binary,
                          stochastic=is_stochastic,
                          is_training = is_training,
                          index=1)

    norm_layer_1 = norm_layer(input_x=fc_layer_1.output(),
                              is_training=is_training,
                              is_drop_out=is_drop_out,
                              activation_function=tf.nn.relu,
                              index=1)

    fc_layer_2 = fc_layer(input_x=norm_layer_1.output(),
                          in_size=fc_units[1],
                          out_size=fc_units[2],
                          binary=is_binary,
                          stochastic=is_stochastic,
                          is_training = is_training,
                          index=2)

    norm_layer_2 = norm_layer(input_x=fc_layer_2.output(),
                              is_training=is_training,
                              is_drop_out=is_drop_out,
                              activation_function=tf.nn.relu,
                              index=2)

    fc_layer_3 = fc_layer(input_x=norm_layer_2.output(),
                          in_size=fc_units[2],
                          out_size=output_size,
                          binary=is_binary,
                          stochastic=is_stochastic,
                          is_training = is_training,
                          index=3)

    norm_layer_3 = norm_layer(input_x=fc_layer_3.output(),
                              is_training=is_training,
                              is_drop_out=False,
                              activation_function=None,
                              index=3)
    
    
    # compute loss    
    with tf.name_scope("loss"):
        net_output = norm_layer_3.output()
        label = tf.one_hot(input_y, output_size)
        # the hinge square loss
        loss = tf.reduce_mean(tf.square(tf.losses.hinge_loss(label, net_output)))
        tf.summary.scalar('loss', loss)
        
    # weight prarameters with SGD. 
    def true_fn(loss=loss, fc_layer_0=fc_layer_0, fc_layer_1=fc_layer_1, fc_layer_2=fc_layer_2, fc_layer_3=fc_layer_3):
        grad_wb0, grad_wb1, grad_wb2, grad_wb3 = tf.gradients(ys=loss, xs=[fc_layer_0.wb, fc_layer_1.wb, fc_layer_2.wb, fc_layer_3.wb])
        grad_b0, grad_b1, grad_b2, grad_b3 = tf.gradients(ys=loss, xs=[fc_layer_0.bias, fc_layer_1.bias, fc_layer_2.bias, fc_layer_3.bias])
        return (grad_wb0, grad_wb1, grad_wb2, grad_wb3, grad_b0, grad_b1, grad_b2, grad_b3)
    def false_fn():
        # graidents = 0 in validation/test phase. 
        return (0., 0., 0., 0., 0., 0., 0., 0.)
        
    grad_update = tf.cond(is_training, true_fn, false_fn)    

    # if is_binary, clip the weights to [-1, +1] before assignment
    if is_binary:
        w0 = tf.contrib.keras.backend.clip(fc_layer_0.weight - lr * grad_update[0], -1.0, 1.0)
        new_w0 = fc_layer_0.weight.assign(w0) 
        w1 = tf.contrib.keras.backend.clip(fc_layer_1.weight - lr * grad_update[1], -1.0, 1.0)
        new_w1 = fc_layer_1.weight.assign(w1)
        w2 = tf.contrib.keras.backend.clip(fc_layer_2.weight - lr * grad_update[2], -1.0, 1.0)
        new_w2 = fc_layer_2.weight.assign(w2)
        w3 = tf.contrib.keras.backend.clip(fc_layer_3.weight - lr * grad_update[3], -1.0, 1.0)
        new_w3 = fc_layer_3.weight.assign(w3)
    else:
        new_w0 = fc_layer_0.weight.assign(fc_layer_0.weight - lr * grad_update[0])
        new_w1 = fc_layer_1.weight.assign(fc_layer_1.weight - lr * grad_update[1])
        new_w2 = fc_layer_2.weight.assign(fc_layer_2.weight - lr * grad_update[2])
        new_w3 = fc_layer_3.weight.assign(fc_layer_3.weight - lr * grad_update[3])
    
    new_b0 = fc_layer_0.bias.assign(fc_layer_0.bias - lr * grad_update[4])
    new_b1 = fc_layer_1.bias.assign(fc_layer_1.bias - lr * grad_update[5])
    new_b2 = fc_layer_2.bias.assign(fc_layer_2.bias - lr * grad_update[6])
    new_b3 = fc_layer_3.bias.assign(fc_layer_3.bias - lr * grad_update[7])
        
    return net_output, loss, (new_b0, new_b1, new_b2, new_b3, new_w0, new_w1, new_w2, new_w3)

    
# evaluate the output
def evaluate(output, input_y):
    with tf.name_scope('evaluate'):
        pred = tf.argmax(output, axis=1)
        error_num = tf.count_nonzero(pred - input_y, name='error_num')
        tf.summary.scalar('error_num', error_num)
    return error_num


def training(X_train, y_train, X_val, y_val, X_test, y_test, binary, stochastic, fc_units, lr_start, lr_end, epoch, batch_size,
             is_drop_out, verbose=False, pre_trained_model=None):
             
    # X_train, y_train, X_val, y_val, X_test, y_test:
    # is_binary: whether use the Binary Connect or not
    # is_stochastic: if use Binary Connect, whether to be stochastic
    # fc_units: number of units for full connect layer
    # lr_start: init learning rate
    # lr_end: final learning rate, used for calculate lr_decay
    # epoch: times of training
    # batch_size; training batch size
    # is_drop_out: whether to dropout during training
             
    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[None, 784], dtype=tf.float32, name='xs')
        ys = tf.placeholder(shape=[None, ], dtype=tf.int64, name='ys')
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    iters = int(X_train.shape[0] / batch_size)
    print('number of batches for training: {}'.format(iters))

    # update learning rate
    learning_rate = tf.Variable(lr_start, name="learning_rate")
    lr_decay = (lr_end / lr_start) ** (1 / epoch)
    lr_update = learning_rate.assign(learning_rate * lr_decay)

    # build network
    output, loss, _updates = Network(xs, ys, is_training,
                           is_drop_out=is_drop_out,
                           is_binary=binary,
                           is_stochastic=stochastic,
                           output_size=10,
                           fc_units=fc_units,
                           lr=learning_rate)

    eve = evaluate(output, ys)

    train_data_generator = data_generator(X_train, y_train)
    batch_gen_train = train_data_generator.data_gen(batch_size)

    val_batch_size = 100
    val_data_generator = data_generator(X_val, y_val)
    batch_gen_val = val_data_generator.data_gen(val_batch_size)
    print('Data Generator Init')

    best_acc = 0
    cur_model_name = 'mnist_{}'.format(int(time.time()))

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
                for _ in range(iters):

                    # generate data
                    train_batch_x, train_batch_y = next(batch_gen_train)
                    _, cur_loss, train_eve= sess.run([_updates, loss, eve], feed_dict={xs: train_batch_x, ys: train_batch_y, is_training: True})

                    train_eve_sum += np.sum(train_eve)
                    loss_sum += np.sum(cur_loss)

                train_acc = 100 - train_eve_sum * 100 / y_train.shape[0]
                loss_sum /= iters
                print('average train loss: {} ,  average accuracy : {}%'.format(loss_sum, train_acc))

                valid_eve_sum = 0
                for _ in range(y_val.shape[0]//val_batch_size):
                    val_batch_x, val_batch_y = next(batch_gen_val)
                    valid_eve, merge_result= sess.run([eve,merge], feed_dict={xs: val_batch_x, ys: val_batch_y, is_training: False})
                    valid_eve_sum += np.sum(valid_eve)

                valid_acc = 100 - valid_eve_sum * 100 / y_val.shape[0]

                _lr = sess.run([lr_update])

                if verbose:
                    print('validation accuracy : {}%'.format(valid_acc))

                # save the merge result summary
                writer.add_summary(merge_result, epc)

                    # when achieve the best validation accuracy, we store the model paramters
                if valid_acc > best_acc:
                    print('* Best accuracy: {}%'.format(valid_acc))
                    best_acc = valid_acc
                    saver.save(sess, 'model/{}'.format(cur_model_name))

            print("Traning ends. The best valid accuracy is {}%. Model named {}.".format(best_acc, cur_model_name))
        else:
            pass
            
        # test the model no matter restore or not
        saver.restore(sess, 'model/{}'.format(cur_model_name))
        test_eve_sum = 0
        for i in range(20):
            test_batch_x = X_test[i*500:(i+1)*500]
            test_batch_y = y_test[i*500:(i+1)*500]
            test_eve = sess.run([eve], feed_dict={xs: test_batch_x, ys: test_batch_y, is_training: False})
            test_eve_sum += np.sum(test_eve)
        test_acc = 100 - test_eve_sum * 100 / y_test.shape[0]
        print('test accuracy : {}%'.format(test_acc))