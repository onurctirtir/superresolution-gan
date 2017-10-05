# Put this script in the directory containing 'imgs/'

# Import images
from glob import glob
from skimage.io import imread

# Shuffle the array of file names
from random import shuffle

# File operations
from os.path import exists
from os import makedirs

import numpy as np

path2npys = 'data/'

train_rgb_path = path2npys + 'train_rgb.npy'
test_rgb_path  = path2npys + 'test_rgb.npy'

imgs_path = 'imgs/*'

training_set_size = 13728
test_set_size     = 500

if not exists(path2npys):
    makedirs(path2npys)

files = glob(imgs_path)

shuffle(files)

train_imgs = np.empty(shape=(training_set_size, 128, 128, 3), \
        dtype=np.uint8)

print('Generate train_imgs.npy ...')
for idx, fname in enumerate(files[test_set_size:]):
    train_imgs[idx] = imread(fname)
print('Done.')

# RGB images, [0, 255], unsigned 8-bit integer format
print('Save "train_imgs" ...')
np.save(file=train_rgb_path , arr=train_imgs , allow_pickle=False)
print('Done.')

test_imgs = np.empty(shape=(test_set_size, 128, 128, 3), dtype=np.uint8)

print('Generate test_imgs.npy ...')
for idx, fname in enumerate(files[:test_set_size]):
    test_imgs[idx] = imread(fname)
print('Done.')

# RGB images, [0, 255], unsigned 8-bit integer format
print('Save "test_imgs" ...')
np.save(file=test_rgb_path , arr=test_imgs , allow_pickle=False)
print('Done.')

