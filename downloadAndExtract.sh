OUTPUTDIR=${1:-cars/}

echo Downloading dataset, this may take a lot of time.
echo Do your other jobs :\)

wget imagenet.stanford.edu/internal/car196/cars_train.tgz -P $OUTPUTDIR
wget imagenet.stanford.edu/internal/car196/cars_test.tgz -P $OUTPUTDIR
wget ai.stanford.edu/~jkrause/cars/car_devkit.tgz -P $OUTPUTDIR

echo Done.
echo Extracting .tgz archives.

tar -xvzf cars/car_devkit.tgz -C $OUTPUTDIR
tar -xvzf cars/cars_train.tgz -C $OUTPUTDIR
tar -xvzf cars/cars_test.tgz -C $OUTPUTDIR

