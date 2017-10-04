# Import tensorflow
import tensorflow as tf

model_path = 'model/'
model_name = 'model'

# Common operations for neural net.

def conv2d(x, output_dim, kernel=3, stride=2, stddev=0.02, padding='SAME', \
        name=None, reuse=False):

    with tf.variable_scope(name) as scope:

        if reuse:
            scope.reuse_variables()

        weights = tf.get_variable(name='weights', \
                shape=[kernel, kernel, x.get_shape()[-1], output_dim], dtype=tf.float32, \
                initializer=tf.truncated_normal_initializer(stddev=stddev))
        
        conv    = tf.nn.conv2d(x, filter=weights, strides=[1, stride, stride, 1], \
                padding=padding)

        biases  = tf.get_variable(name='biases', shape=[output_dim], \
                dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        out     = tf.nn.bias_add(conv, biases)
        
    return out

# Turn scale off when next layer is relu
def batch_norm(x, epsilon=1e-5, momentum = 0.999, scale=False, is_training=True, 
        name=None):

    return tf.contrib.layers.batch_norm(x, decay=momentum,
            scale=scale, epsilon=epsilon, updates_collections=None,
            is_training=is_training, scope=name)

def lrelu(x, leak=0.01):
    
    return tf.maximum(x, leak*x)

def linear(x, output_size, stddev=0.02, biases_start=0.0, name=None):

    with tf.variable_scope(name):
    
        weights = tf.get_variable(name='weights', \
                shape=[x.get_shape()[1], output_size], dtype=tf.float32, \
                initializer=tf.random_normal_initializer(stddev=stddev))

        biases  = tf.get_variable(name='biases', shape=[output_size], dtype=tf.float32, \
                initializer=tf.constant_initializer(biases_start))

    return tf.nn.xw_plus_b(x, weights, biases)

def conv_net(x, is_training=True, reuse=False):

    with tf.variable_scope('conv_net') as scope:
 
        if reuse:            
            scope.reuse_variables()
   
        #Make image ready to conv. process
        x = tf.reshape(x, shape=[-1, 64, 64, 3])
        #64 x 64 x 3
     
        layer1 = conv2d(x, output_dim=32, stride=1, name='g_layer1')
        layer1 = batch_norm(layer1, is_training=is_training, name='g_layer1_bn')
        layer1 = lrelu(layer1)
        #64 x 64 x 32
        
        layer2 = conv2d(layer1, output_dim=128, stride=1, name='g_layer2')
        layer2 = batch_norm(layer2, is_training=is_training, name='g_layer2_bn')
        layer2 = lrelu(layer2)
        #64 x 64 x 128

        layer3 = conv2d(layer2, output_dim=128, stride=1, name='g_layer3')
        layer3 = batch_norm(layer3, is_training=is_training, name='g_layer3_bn')
        layer3 = lrelu(layer3)
        #64 x 64 x 128

        layer3 = tf.image.resize_images(layer3, size=[128, 128])

        layer4 = conv2d(layer3, output_dim=128, stride=1, name='g_layer4')
        layer4 = batch_norm(layer4, is_training=is_training, name='g_layer4_bn')
        layer4 = lrelu(layer4)
        #128 x 128 x 128

        layer5 = conv2d(layer4, output_dim=64, stride=1, name='g_layer5')
        layer5 = batch_norm(layer5, is_training=is_training, name='g_layer5_bn')
        layer5 = lrelu(layer5)
        #128 x 128 x 64

        out = conv2d(layer5, output_dim=3, stride=1, name='g_layer_out')
        out = tf.nn.sigmoid(out)
        #128 x 128 x 3

    return out

