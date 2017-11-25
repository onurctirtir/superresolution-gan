# Eliminate warnings
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL']='2'

from numpy import load, concatenate
from os import makedirs
from scipy.misc import imsave, imresize
from time import strftime

from commons import generator
from commons import costs_and_vars

from commons import BatchGenerator

import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
add_arg = parser.add_argument

add_arg('--model'     , default='default'      , type=str, \
        help='Name of the trained model to use.')
add_arg('--batch-size', default=50             , type=int, \
        help='Number of images provided at each test iteration.')
add_arg('--npy-path'  , default='data/test.npy', type=str, \
        help='Path to numpy array containing test set images.')

args = parser.parse_args()

class Tester:
    def __init__(self): 
        print('Importing test set ...')
        self.dataset = load(file=args.npy_path, allow_pickle=False)
        print('Done.')

        self.batch_size    = args.batch_size
        self.model         = args.model
        self.dataset_size  = self.dataset.shape[0]
        self.out_path      = '/'.join(['test_out_imgs', strftime('%Y%m%d-%H%M%S']))

    def test(self):
        rgb_x   = tf.placeholder(tf.float32, [None, 128, 128, 3])
        sml_x   = tf.placeholder(tf.float32, [None,  64,  64, 3])
        gener_x = generator(sml_x, is_training=False, reuse=False)

        real_d  = discriminator(big_x, is_training=False, reuse=False)
        gener_d = discriminator(gener_x, is_training=False, reuse=True)

        g_cost, d_cost = costs_and_vars(big_x, gener_x, real_d, gener_d, is_training=False)

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            saver = tf.train.Saver()

            try:
                saver.restore(sess, '/'.join(['models', self.model, self.model]))
            except:
                print('Model coult not be restored. Exiting.')
                exit()

            makedirs(self.out_path)

            print('Saving test results ...')

            start = 0

            for batch in BatchGenerator(self.batch_size, self.dataset_size):
                batch_big = self.dataset[batch] / 255.0
                batch_sml = array([imresize(img, size=(64, 64, 3)) \
                        for img in batch_big])

                superres_imgs = sess.run(gener_x, feed_dict={sml_x: batch_sml})

                gc, dc  = sess.run([g_cost, d_cost], \
                        feed_dict={big_x : batch_big, sml_x : batch_sml})

                images = concatenate( \
                    ( \
                        array([imresize(img, size=(128, 128, 3)) / 255.0 \
                                for img in batch_sml]), \
                        superres_imgs,
                        batch_big \
                    ), 2)

                for idx, image in enumerate(images):
                    imsave('%s/%d.png' % (self.out_path, start+idx), image)

                start += self.batch_size

                print('%d/%d saved successfully: Generative cost=%.9f, Discriminative cost=%.9f' % \
                        (min(self.dataset_size - start, self.batch_size), self.dataet_size, gc, dc))

if __name__ == '__main__':
    tester = Tester()
    tester.test()

