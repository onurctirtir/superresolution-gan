# superresolution-gan
A DCGAN implementation in Tensorflow for super-resolution of 64 x 64 RGB images to 128 x 128 ones.

# Notes on dataset

Downloaded the car dataset from [ai.stanford.edu](http://ai.stanford.edu/~jkrause/cars/car_dataset.html).

Reason why I preferred this dataset is to maintain the coherence between images.

After cropping images according to bounding boxes, eliminated the ones with height < 128 or width < 128 and 
resized them roughly to 128 x 128 x 3 images. Randomly splitted 500 of them for test,
4D '.npy's with shape [size, 128, 128, 3] were generated under 'data/'.

# Downloading

Dataset I've used for train and test can be found at: [dropbox](https://www.dropbox.com/sh/on18ekittp46n9f/AAAmezABdsGv7RphhHbK6ljHa?dl=0)

Download dataset and put 'data/' into the same directory with train and test scripts.

# Training and test

After setting parameters like 'epochs' in 'train.py',first train the model:

```
python train.py
```

This script will also save trained model under the 'model/'  directory.

Then test it:

```
python test.py
```

Test script will restore the model from 'model/' and run on the images residing in 'test_rgb.npy'.
Then it will create test outputs in 'test_imgs/'. 

# Example test outputs

Left to right, test output images' format is like this:
 - after rough super-resolution with scipy.misc.imresize(img, size=(128, 128, 3)),
 - after super-superresolution with superresolution-gan,
 - original 128 x 128 x 3 image
 
 ![image1](https://github.com/onurctirtir/superresolution-gan/tree/master/example_test_imgs/img0.png)
