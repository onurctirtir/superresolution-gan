# Eliminate warnings
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL']='2'

# File operations
from os import system as os_sys
from os import path as os_path

# Export images
from scipy.misc import imsave as sci_save

# Derive 64 x 64 x 3 images
from scipy.misc import imresize as sci_resize

# Load images
from numpy import load as np_load

# Concatenate images over their channel axes
from numpy import concatenate as np_concat

# Import common operations for neural net.
from commons import *

# Convert lists into numpy arrays
from numpy import array as np_array

import tensorflow as tf

test_out_path  = 'test_imgs/'

test_rgb_path  = 'data/test_rgb.npy'

test_set_size   = 500
test_batch_size = 50

rgb_x = tf.placeholder(tf.float32, [test_batch_size, 128, 128, 3])
sml_x = tf.placeholder(tf.float32, [test_batch_size,  64,  64, 3])

print('Importing test set ...')
test_rgb = np_load(file=test_rgb_path, allow_pickle=False)
print('Test set imported')

init = tf.global_variables_initializer()

generated = conv_net(sml_x, is_training=True, reuse=False)

with tf.Session() as sess:
    sess.run(init)

    saver = tf.train.Saver()

    saver.restore(sess, model_path + model_name)

    print('Model has been restored from the directory: %s' % model_path)

    if os_path.isdir(test_out_path):
        print('Clean old test outputs ...')
        os_sys('rm ' + test_out_path + '/*')

    else:
        os_sys('mkdir ' + test_out_path)
    
    print('Saving test results ...')

    for batch in range(int(test_set_size / test_batch_size)):
        start = batch * test_batch_size
        end   = start + test_batch_size

        _test_rgb = test_rgb[start:end] / 255.0
        _test_sml = np_array([sci_resize(img, size=(64, 64, 3)) for img in _test_rgb])

        regenerated = sess.run(generated, feed_dict={sml_x: _test_sml})

        images = np_concat(
            (np_array([sci_resize(img, size=(128, 128, 3)) / 255.0 for img in _test_sml]),
                regenerated, _test_rgb), 2)

        for idx, image in enumerate(images):
            sci_save(test_out_path + 'img' + str(start+idx) + '.png', image)

