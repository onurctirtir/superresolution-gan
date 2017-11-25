import tensorflow as tf

from numpy.random import permutation

def conv2d(x, output_dim, kernel=3, stride=2, stddev=0.02, padding='SAME', name=None, reuse=False):
    with tf.variable_scope(name) as scope:
        if reuse:
            scope.reuse_variables()

        weights = tf.get_variable(name='weights', \
                shape=[kernel, kernel, x.get_shape()[-1], output_dim], dtype=tf.float32, \
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        
        conv    = tf.nn.conv2d(x, filter=weights, strides=[1, stride, stride, 1], padding=padding)

        biases  = tf.get_variable(name='biases', shape=[output_dim], \
                dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        out     = tf.nn.bias_add(conv, biases)
        
    return out

def batch_norm(x, epsilon=1e-5, momentum = 0.999, scale=False, is_training=True, \
        name=None, reuse=False):
    with tf.variable_scope(name):
        if reuse:
            scope.reuse_variables()

        return tf.contrib.layers.batch_norm(x, decay=momentum, scale=scale, epsilon=epsilon, \
            updates_collections=None, is_training=is_training, scope=name)

def lrelu(x, leak=0.01):
    '''Leaky relu linear activation function with 'leak'.'''
    return tf.maximum(x, leak*x)

def linear(x, output_size, stddev=0.02, biases_start=0.0, name=None, reuse=False):
    '''Fully connected layer.'''
    with tf.variable_scope(name):
        if reuse:
            scope.reuse_variables()
    
        weights = tf.get_variable(name='weights', \
                shape=[x.get_shape()[1], output_size], dtype=tf.float32, \
                initializer=tf.random_normal_initializer(stddev=stddev))

        biases  = tf.get_variable(name='biases', shape=[output_size], dtype=tf.float32, \
                initializer=tf.constant_initializer(biases_start))

    return tf.nn.xw_plus_b(x, weights, biases)

def generator(x, is_training=True, reuse=False):
    '''Map input images from 64 x 64 x 3 to 128 x 128 x 3.'''
    with tf.variable_scope('generator') as scope:
        if reuse:            
            scope.reuse_variables()
   
        conv1 = conv2d(x, output_dim=32, stride=1, name='g_conv1')
        conv1 = batch_norm(conv1, is_training=is_training, name='g_conv1_bn')
        conv1 = lrelu(conv1)
        #64 x 64 x 32
        
        conv2 = conv2d(conv1, output_dim=128, stride=1, name='g_conv2')
        conv2 = batch_norm(conv2, is_training=is_training, name='g_conv2_bn')
        conv2 = lrelu(conv2)
        #64 x 64 x 128

        conv3 = conv2d(conv2, output_dim=128, stride=1, name='g_conv3')
        conv3 = batch_norm(conv3, is_training=is_training, name='g_conv3_bn')
        conv3 = lrelu(conv3)
        #64 x 64 x 128

        upsampled = tf.image.resize_images(conv3, size=[128, 128])

        conv4 = conv2d(upsampled, output_dim=128, stride=1, name='g_conv4')
        conv4 = batch_norm(conv4, is_training=is_training, name='g_conv4_bn')
        conv4 = lrelu(conv4)
        #128 x 128 x 128

        conv5 = conv2d(conv4, output_dim=64, stride=1, name='g_conv5')
        conv5 = batch_norm(conv5, is_training=is_training, name='g_conv5_bn')
        conv5 = lrelu(conv5)
        #128 x 128 x 64

        conv6 = conv2d(conv5, output_dim=3, stride=1, name='g_conv6')
        conv6 = tf.nn.sigmoid(conv6)
        #128 x 128 x 3

    return conv6

def discriminator(images, is_training=True, reuse=False):
    '''Discriminate 128 x 128 x 3 images fake or real within the range [fake, real] = [0, 1].'''

    with tf.variable_scope('discriminator') as scope:
        
        if reuse:
            scope.reuse_variables()
 
        conv1 = conv2d(images, output_dim=64, kernel=7, stride=1, name='d_conv1')
        conv1 = batch_norm(conv1, is_training=is_training, name='d_conv1_bn')
        conv1 = lrelu(conv1)
        #128 x 128 x 64
        
        conv2 = conv2d(conv1, output_dim=64, kernel=7, stride=2, name='d_conv2')
        conv2 = batch_norm(conv2, is_training=is_training, name='d_conv2_bn')
        conv2 = lrelu(conv2)
        #64 x 64 x 64
            
        conv3 = conv2d(conv2, output_dim=32, kernel=3, stride=2, name='d_conv3')
        conv3 = batch_norm(conv3, is_training=is_training, name='d_conv3_bn')
        conv3 = lrelu(conv3)
        #32 x 32 x 32

        conv4 = conv2d(conv3, output_dim=1, kernel=3, stride=2, name='d_conv4')
        conv4 = batch_norm(conv4, is_training=is_training, name='d_conv4_bn')
        conv4 = lrelu(conv4)
        #32 x 32 x 1

        fc = tf.reshape(conv4, [-1, 32 * 32 * 1])
        fc = linear(fc, output_size=1, name='d_fc')
    
    return fc

def costs_and_vars(real, generated, real_disc, gener_disc, is_training=True):
    '''Return generative and discriminator networks\' costs,
    and variables to optimize them if is_training=True.'''
    d_real_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_disc, \
            labels=tf.ones_like(real_disc)))
    d_gen_cost  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gener_disc, \
            labels=tf.zeros_like(gener_disc)))
     
    g_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gener_disc, \
            labels=tf.ones_like(gener_disc))) * 0.1 + \
            tf.reduce_mean(tf.abs(tf.subtract(generated, real)))

    d_cost = d_real_cost + d_gen_cost
    
    if is_training:
        t_vars = tf.trainable_variables()
        
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]
    
        return g_cost, d_cost, g_vars, d_vars

    else:
        return g_cost, d_cost

class BatchGenerator:
    '''Generator class returning list of indexes at every iteration.'''
    def __init__(self, batch_size, dataset_size):
        self.batch_size   = batch_size
        self.dataset_size = dataset_size

        assert (self.dataset_size > 0)               , 'Dataset is empty.'
        assert (self.dataset_size => self.batch_size), 'Invalid bathc_size.'
        assert (self.batch_size > 0)                 , 'Invalid bathc_size.'

        self.last_idx = -1
        self.idxs     = permutation(dataset_size)

    def __iter__(self):
        return self

    def __next__(self):
        if self.last_idx + self.batch_size <= self.dataset_size - 1:
            start = self.last_idx + 1
            self.last_idx += self.batch_size

            return self.idxs[start: self.last_idx + 1]

        else:
            if self.last_idx == self.dataset_size - 1:
                raise StopIteration

            start = self.last_idx + 1
            self.last_idx = self.dataset_size - 1

            return self.idxs[start, self.last_idx + 1]

