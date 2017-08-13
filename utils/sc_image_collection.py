"""
Read image_collection.mat file of solar panel data to get the list of training/validation/test images
"""
import numpy as np
import scipy.misc as misc
import scipy.io
import os

matFilePath = os.path.expanduser("~/Documents/data/igarssTrainingAndTestingData/imageCollections.mat")
matData = scipy.io.loadmat(matFilePath)
trainingList = matData['trainingImages'][:, (0, 7)]
validationList = matData['validationImages'][:, (0, 7)]
testingList = matData['testingImages'][:, (0, 7)]
