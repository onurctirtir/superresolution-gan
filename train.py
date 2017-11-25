# Eliminate warnings
from os import environ
environ['TF_CPP_MIN_LOG_LEVEL']='2'

from numpy import array, load
from os import makedirs
from os.path import exists
from scipy.misc import imresize

from commons import generator
from commons import costs_and_vars

from commons import BatchGenerator

import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
add_arg = parser.add_argument

add_arg('--model'     , default='default'       , type=str  , \
        help='Name of the model to be trained.')
add_arg('--batch-size', default=16              , type=int  , \
        help='Number of images provided at each training iteration.')
add_arg('--lr-gen'    , default=1e-4            , type=float, \
        help='Learning rate of generative network.')
add_arg('--lr-dis'    , default=1e-4            , type=float, \
        help='Learning rate of discriminative network.')
add_arg('--epochs'    , default=5               , type=int  , \
        help='Number of training epochs.')
add_arg('--disp-every', default=10              , type=int  , \
        help='Display costs per each disp_every iterations.')
add_arg('--save-every', default=None            , type=int  , \
        help='Save model per each save_every epochs.')
add_arg('--npy-path'  , default='data/train.npy', type=str  , \
        help='Path to numpy array containing training set images.')
add_arg('--continues' , default=0               , type=int  , \
        help='If set to 1, continue to train the specified model. \
        If set to 0, a new model with specified name will be generated.')

args = parser.parse_args()

class Trainer:
    def __init__(self): 
        print('Importing training set ...')
        self.dataset = load(file=args.npy_path, allow_pickle=False)
        print('Done.')

        self.batch_size      = args.batch_size
        self.training_epochs = args.epochs
        self.model           = args.model
        self.display_step    = args.disp_every
        self.save_step       = args.save_every
        self.lr_gen          = args.lr_gen
        self.lr_dis          = args.lr_dis
        self.continues       = args.continues
        self.dataset_size    = self.dataset.shape[0]

    def train(self):
        big_x   = tf.placeholder(tf.float32, [None, 128, 128, 3])
        sml_x   = tf.placeholder(tf.float32, [None,  64,  64, 3])
        gener_x = generator(sml_x, is_training=True, reuse=False)

        real_d  = discriminator(big_x, is_training=True, reuse=False)
        gener_d = discriminator(gener_x, is_training=True, reuse=True)

        g_cost, d_cost, g_vars, d_vars = \
                costs_and_vars(big_x, gener_x, real_d, gener_d, is_training=True)
        
        g_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_gen).\
                minimize(g_cost, var_list=g_vars)
        d_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_dis).\
                minimize(d_cost, var_list=d_vars)
        
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            saver = tf.train.Saver()

            if self.continues:
                try:
                    saver.restore(sess, '/'.join(['models', self.model, self.model]))
                except:
                    print('Model coult not be restored. Exiting.')
                    exit()

            if not exists('models'):
                makedirs('models')

            passed_iters = 0

            for epoch in range(1, self.training_epochs+1):
                print('Epoch:', str(epoch))
                
                for batch in BatchGenerator(self.batch_size, self.dataset_size):
                    batch_big = self.dataset[batch] / 255.0
                    batch_sml = array([imresize(img, size=(64, 64, 3)) \
                            for img in batch_big])                        
            
                    _, gc, dc  = sess.run([g_optimizer, g_cost, d_cost], \
                            feed_dict={big_x : batch_big, sml_x : batch_sml})
                    sess.run([d_optimizer], \
                            feed_dict={big_x : batch_big, sml_x : batch_sml})                        
                    
                    passed_iters += 1

                    if passed_iters % self.display_step == 0:
                        print('Passed iterations=%d, Generative cost=%.9f, Discriminative cost=%.9f' %\
                                (passed_iters, gc, dc))

                if self.save_step and epoch % self.save_step == 0:
                    saver.save(sess, '/'.join(['models', self.model, self.model]))

                    print('Model \'%s\' saved in: \'%s/\'' \
                            % (self.model, '/'.join(['models', self.model])))

            print('Optimization finished.')

            saver.save(sess, '/'.join(['models', self.model, self.model]))
            
            print('Model \'%s\' saved in: \'%s/\'' \
                    % (self.model, '/'.join(['models', self.model])))

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()

