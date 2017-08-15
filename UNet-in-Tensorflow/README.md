# U-Net Implementation in TensorFlow

Modified from the implementation of U-Net [here](https://github.com/kkweon/UNet-in-Tensorflow). The orginal README file can be seen `README_original.md`

* `train.py`: train a U-Net for segmentation
* `test.py`: test a trained U-Net for segmentation
* `utils.py`: utils for data processing, borrowed from [`tensorflow-deeplab-resnet`](https://github.com/DrSleep/tensorflow-deeplab-resnet) project
* `experiments_train.py`: script for running experiments using `train.py`
* `experiments_test.py`: script for running experiments using `test.py`

It also needs modules from `dataReader`, for data input pipeline, and `metrics` for evaluating the segmentation performance.

# More details about experiments
Since the train.py 