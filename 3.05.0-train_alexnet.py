from config import imagenet_alexnet_config as config
from pyimagesearch.nn.mxconv import MxAlexNet
import mxnet as mx
import argparse
import logging
import json
import os

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--checkpoints", required=True, help="path to output checkpoint directory")
ap.add_argument("-p", "--prefix", required=True, help="name of model prefix")
ap.add_argument("-s", "--start-epoch", type=int, default=0, help="epoch to restart training at")
args = vars(ap.parse_args())

logging.basicConfig(level=logging.DEBUG, filename="training_{}.log".format(args["start_epoch"]), filemode="w")

means = json.loads(open(config.DATABASE_MEAN).read())
batchSize = config.BATCH_SIZE * config.NUM_DEVICES
print(config.TRAIN_MX_REC)
trainIter = mx.io.ImageRecordIter(
    path_imgrec=config.TRAIN_MX_REC,
    data_shape=(3, 227, 227),
    batch_size=batchSize,
    rand_crop=True,
    rand_mirror=True,
    rotate=15,
    mx_shear_ratio=0.1,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"],
    preprocess_threads=config.NUM_DEVICES * 2
)

valIter = mx.io.ImageRecordIter(
    path_imgrec=config.VAL_MX_REC,
    data_shape=(3, 227, 227),
    batch_size=batchSize,
    mean_r=means["R"],
    mean_g=means["G"],
    mean_b=means["B"]
)

opt = mx.optimizer.SGD(learning_rate=1e-2, momentum=0.9, wd=0.0005, rescale_grad=1.0 / batchSize)

checkpointsPath = os.path.sep.join([args["checkpoints"], args["prefix"]])
argParams = None
auxParams = None

if args["start_epoch"] <= 0:
    print("[INFO] building network...")
    model = MxAlexNet.build(config.NUM_CLASSES)
else:
    print("[INFO] loading epoch {}...".format(args["start_epoch"]))
    model = mx.model.FeedForward.load(checkpointsPath, args["start_epoch"])

    argParams = model.arg_params
    auxParams = model.aux_params
    model = model.symbol

model = mx.model.FeedForward(
    ctx=[mx.gpu(0)],
    symbol=model,
    initializer=mx.initializer.Xavier(),
    arg_params=argParams,
    aux_params=auxParams,
    optimizer=opt,
    num_epoch=90,
    begin_epoch=args["start_epoch"]
)

batchEndCBs = [mx.callback.Speedometer(batchSize, 500)]
epochEndCBs = [mx.callback.do_checkpoint(checkpointsPath)]
metrics = [mx.metric.Accuracy(), mx.metric.TopKAccuracy(top_k=5), mx.metric.CrossEntropy()]

print("[INFO] training network...")
model.fit(X=trainIter,
          eval_data=valIter,
          eval_metric=metrics,
          batch_end_callback=batchEndCBs,
          epoch_end_callback=epochEndCBs)