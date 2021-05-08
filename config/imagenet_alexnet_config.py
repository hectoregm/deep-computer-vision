from os import path

BASE_PATH = "/home/hectoregm/Projects/imagenet/ILSVRC2015"

IMAGES_PATH = path.sep.join([BASE_PATH, "Data/CLS-LOC"])
IMAGE_SETS_PATH = path.sep.join([BASE_PATH, "ImageSets/CLS-LOC"])
DEVKIT_PATH = path.sep.join([BASE_PATH, "devkit/data"])

WORD_IDS = path.sep.join([DEVKIT_PATH, "map_clsloc.txt"])

TRAIN_LIST = path.sep.join([IMAGE_SETS_PATH, "train_cls.txt"])
VAL_LIST = path.sep.join([IMAGE_SETS_PATH, "val.txt"])
VAL_LABELS = path.sep.join([DEVKIT_PATH, "ILSVRC2015_clsloc_validation_ground_truth.txt"])

VAL_BLACKLIST = path.sep.join([DEVKIT_PATH, "ILSVRC2015_clsloc_validation_blacklist.txt"])

NUM_CLASSES = 1000
NUM_TEST_IMAGES = 50 * NUM_CLASSES

MX_OUTPUT = "imagenet"
TRAIN_MX_LIST = path.sep.join([MX_OUTPUT, "lists/train.lst"])
VAL_MX_LIST = path.sep.join([MX_OUTPUT, "lists/val.lst"])
TEST_MX_LIST = path.sep.join([MX_OUTPUT, "lists/test.lst"])

TRAIN_MX_REC = "/home/hectoregm/Projects/deep-computer-vision/imagenet/rec/train.rec"#path.sep.join([MX_OUTPUT, "rec/train.rec"])
VAL_MX_REC = "/home/hectoregm/Projects/deep-computer-vision/imagenet/rec/val.rec"#path.sep.join([MX_OUTPUT, "rec/val.rec"])
TEST_MX_REC = path.sep.join([MX_OUTPUT, "rec/test.rec"])

DATASET_MEAN = "output/imagenet_mean.json"

BATCH_SIZE = 128
NUM_DEVICES = 1
