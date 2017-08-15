# Duke_2017Summer
This project includes the code for semantic segmentation of satelite images implemented by Zhenlin Xu as an intern at Duke

# What are in this project
Three CNN based segmentation models and some utilities like data reader, segmentation metric, pre/post-processing, etc.
* `dataReader`: functions of reading image data for deep nets
  * `image_reader.py`
  * `mat_batch_reader`
  * `matFileDescription.txt`
  * `patchify.py`
* `metrics`: implementations of segmentation performance metrics
* `notebook`: notebooks for visualizing experiment results or test functionalities of other functions
* `utils`: script for image pre/post processing
* `FCN`: Fully convolutional network modified from the implementation [here](https://github.com/shekkizh/FCN.tensorflow).
* `tensorflow-deeplab-resnet`: DeepLab model modified from the implementation [here](https://github.com/DrSleep/tensorflow-deeplab-resnet)
* `UNet-in-Tensorflow`: UNet model modified from the implementation [here](https://github.com/kkweon/UNet-in-Tensorflow)

# Requirements
The project were tested under the
* Tensorflow 1.1.0: FCN and DeepLab model are under TF of python 3.5 and U-Net was tested under TF of python 2.7.
* Numpy 1.12.1 
* scipy 0.19.0
Other used modules can be found at the importing part of every script

# Example: semantic segmentation of buildings from satellite images using U-Net 
This example shows a procedure of training a U-Net for building segmentation 
1. Data preparation: 
    * The original satellite images are of large sizes and can not be feed into U-Net or other CNNs. They are of different pysical resolution (length/pixel) as well. The first step is to unify them into the same resolution using `./utils/unify_image_resolution.py`. 
    * Use `patchify_building_images.py` to generated small patches from a pair of RGB image and a binary label image (ground truth). The generated patch pairs will be named as `{LargeImageName}_{index}_image.png` and `{LargeImageName}_{index}_truth.png`. Also, a text file, of which each line is the path of color image and truth image pair, is generated.
2. Training:  Use `train.py` to train models. 
    * The path of folder containing the generated patches and the text file of the path list from step 1 should be provided to `train.py`. 
    * Hyperparameters for training a CNN is needed as well. More details about needed parameters are in `train.py`. 
    * During training, checkpoints (trained model and meta data for tensorboard) are saved (was set under `./snapshots_building/{experiments_name}`) so that it can be used for testing or continuing an interuptted training. You can also use Tensorboard to analysis the training process, like the cost function values. 
    * `experiments_train.py` recursively run `train.py` with different training parameters.
3. Testing: Use `test.py` to evaluate trained models. 
    * The path of trained model and the large image to be tested on are needed. 
    * The large image are divided small patches and their predictions are stitched into a large label image for evaluation. Those patches are overlapped. So each patch is weighted with a 2D Gaussion distribution (same size with the path, higher weight at the center of each patch). The stitched images of weighted patches are normalized with the stitched Gaussion masks.
    * Evaluate the performance using the metric class in `seg_metric.py` or functions in `eval_segm.py`.
    * Run `experiments_test.py` for testing different models on different test data.