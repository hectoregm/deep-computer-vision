from .config import dogs_vs_cats_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pyimagesearch.preprocessing import AspectAwarePreprocessor
from pyimagesearch.inout import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

trainPaths = list(paths.list_images(config.IMAGES_PATH))
trainLabels = [p.split(os.path.sep)[-1].split(".")[0] for p in trainPaths]

le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_TEST_IMAGES, stratify=trainLabels, random_state=42)
(trainPaths, testPaths, trainLabels, testLabels) = split

split = train_test_split(trainPaths, trainLabels, test_size=config.NUM_VAL_IMAGES, stratify=trainLabels, random_state=42)
(trainPaths, valPaths, trainLabels, valLabels) = split

datasets = [
    ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
    ("val", valPaths, valLabels, config.VAL_HDF5),
    ("test", testPaths, testLabels, config.TEST_HDF5)]

aap = AspectAwarePreprocessor(256, 256)
(R, G, B) = ([], [], [])