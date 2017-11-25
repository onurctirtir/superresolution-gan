from scipy.io import loadmat
from scipy.misc import imresize
from skimage.io import imread, imsave
from os.path import exists
from os import makedirs
from shutil import rmtree

import argparse

parser = argparse.ArgumentParser()
add_arg = parser.add_argument

add_arg('--input', default='cars/', type=str, \
        help='\'OUTPUTDIR\' set in \'downloadAndExtract.sh\'.')
add_arg('--output', default='imgs/', type=str, \
        help='Cropped images will be stored here.')

args = parser.parse_args()

if __name__ == '__main__':
    print('Running  ...')

    if not exists(args.output):
        makedirs(args.output)

    path2train  = args.input + 'cars_train/'

    anns = loadmat(args.input + 'devkit/cars_train_annos.mat')['annotations']

    for x1, y1, x2, y2, _, fname in anns[0]:
        x1 = x1[0][0]
        x2 = x2[0][0]
        y1 = y1[0][0]
        y2 = y2[0][0]
        fname = fname[0]

        img = imread(path2train + fname)

        if len(img.shape) == 3 and x2 - x1 >= 128 and y2 - y1 >= 128:
            img = imresize(img[y1:y2, x1:x2, :], size=(128, 128, 3))
            imsave(args.output + fname, img)

    path2test   = args.input + 'cars_test/'

    anns = loadmat(args.input + 'devkit/cars_test_annos.mat')['annotations']

    for x1, y1, x2, y2, fname in anns[0]:
        x1 = x1[0][0]
        x2 = x2[0][0]
        y1 = y1[0][0]
        y2 = y2[0][0]
        fname = fname[0]

        img = imread(path2test + fname)

        if len(img.shape) == 3 and x2 - x1 >= 128 and y2 - y1 >= 128:
            img = imresize(img[y1:y2, x1:x2, :], size=(128, 128, 3))
            imsave(args.output + '_' + fname, img)

    print('Done.\nClearing downloaded archives and temporarily extracted files ...')

    rmtree(args.input)

    print('Done.')

