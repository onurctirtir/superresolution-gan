# Eliminate warnings
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Load images
from numpy import load as np_load

# Generate an 'index list' to shuffle data
from numpy.random import permutation as np_shuffle

# Derive 64 x 64 x 3 images
from scipy.misc import imresize as sci_resize

# Convert lists to numpy arrays
from numpy import array as np_array

# Import common operations for neural net.
from commons import *

# Import tensorflow
import tensorflow as tf

alpha_gen       = 0.0001
alpha_dis       = 0.0001
training_epochs = 5
batch_size      = 16
display_step    = 10

train_rgb_path = 'data/train_rgb.npy'

training_set_size = 13728

rgb_x = tf.placeholder(tf.float32, [batch_size, 128, 128, 3])
sml_x = tf.placeholder(tf.float32, [batch_size,  64,  64, 3])

print('Importing training set ...')
train_rgb = np_load(file=train_rgb_path, allow_pickle=False)
print('Training set imported')

def next_batch(batch_size):

    if not hasattr(next_batch, 'last_image'):
        next_batch.last_image = 0

    if not hasattr(next_batch, 'idxs'):
        next_batch.idxs = np_shuffle(training_set_size)

    if next_batch.last_image + batch_size > training_set_size:
        next_batch.idxs = np_shuffle(training_set_size)
        next_batch.last_image = 0

    start = next_batch.last_image
    next_batch.last_image += batch_size

    return train_rgb[next_batch.idxs[start: next_batch.last_image]] / 255.0

def discriminator_dcgan(x, is_training=True, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
 
        layer1 = conv2d(x, output_dim=64, kernel=7, stride=1, name='d_layer1')
        layer1 = batch_norm(layer1, is_training=is_training, name='d_layer1_bn')
        layer1 = lrelu(layer1)
        #128 x 128 x 64
        
        layer2 = conv2d(layer1, output_dim=64, kernel=7, stride=2, name='d_layer2')
        layer2 = batch_norm(layer2, is_training=is_training, name='d_layer2_bn')
        layer2 = lrelu(layer2)
        #64 x 64 x 64
            
        layer3 = conv2d(layer2, output_dim=32, kernel=3, stride=2, name='d_layer3')
        layer3 = batch_norm(layer3, is_training=is_training, name='d_layer3_bn')
        layer3 = lrelu(layer3)
        #32 x 32 x 32

        layer4 = conv2d(layer3, output_dim=1, kernel=3, stride=2, name='d_layer4')
        layer4 = batch_norm(layer4, is_training=is_training, name='d_layer4_bn')
        layer4 = lrelu(layer4)
        #32 x 32 x 3

        layer5 = tf.reshape(layer4, [-1, 32 * 32 * 1])
        layer5 = linear(layer5, output_size=1, name='d_layer5_lin')
    
    return layer5

def loss_dcgan(x, gen):

    real_d = discriminator_dcgan(x, is_training=True, reuse=False)
    fake_d = discriminator_dcgan(gen, is_training=True, reuse=True)
    
    d_real_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_d, labels=tf.ones_like(real_d)))
    d_gen_cost  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_d, labels=tf.zeros_like(fake_d)))
     
    g_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_d, labels=tf.ones_like(fake_d))) * 0.1 + \
            tf.reduce_mean(tf.abs(tf.subtract(gen, x)))

    d_cost = d_real_cost + d_gen_cost
    
    t_vars = tf.trainable_variables()

    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]
    
    return g_cost, d_cost, g_vars, d_vars

generated = conv_net(sml_x, is_training=True, reuse=False)

g_cost, d_cost, g_vars, d_vars = loss_dcgan(rgb_x, generated)

optimizerg = tf.train.AdamOptimizer(learning_rate=alpha_gen).minimize(g_cost, var_list=g_vars)
optimizerd = tf.train.AdamOptimizer(learning_rate=alpha_dis).minimize(d_cost, var_list=d_vars)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    step = 1

    while step * batch_size < training_epochs * training_set_size:
        batch_rgb = next_batch(batch_size)
        batch_sml = np_array([sci_resize(img, size=(64, 64, 3)) \
                for img in batch_rgb])
        
        _, gc, dc  = sess.run([optimizerg, g_cost, d_cost], \
                feed_dict={rgb_x : batch_rgb, sml_x : batch_sml})
      
        sess.run([optimizerd], feed_dict={rgb_x : batch_rgb, sml_x : batch_sml})

        if step % display_step == 0:
            print('Iter', str(step*batch_size), 'gc: {:.09}'.format(gc), \
                    'dc: {:.09}'.format(dc))
        
        step += 1

    print('Optimization finished.')

    saver = tf.train.Saver()    
    saver.save(sess, model_path + model_name)
    
    print("Model saved into directory: %s" % model_path)
    
