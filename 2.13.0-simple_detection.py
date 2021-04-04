from pyimagesearch.utils.simple_obj_det import image_pyramid
from pyimagesearch.utils.simple_obj_det import sliding_window
from pyimagesearch.utils.simple_obj_det import classify_batch
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

INPUT_SIZE = (350, 350)
PYR_SCALE = 1.5
WIN_STEP = 16
ROI_SIZE = (224, 224)
BATCH_SIZE = 64

print("[INFO] loading network...")
model = ResNet50(weights="imagenet", include_top=True)

labels = {}
orig = cv2.imread(args["image"])
(h, w) = orig.shape[:2]

resized = cv2.resize(orig, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)

batchROIs = None
batchLocs = []

print("[INFO] detecting objects...")
start = time.time()

for image in image_pyramid(resized, scale=PYR_SCALE, minSize=ROI_SIZE):
    for (x, y, roi) in sliding_window(image, WIN_STEP, ROI_SIZE):
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        roi = imagenet_utils.preprocess_input(roi)

        if batchROIs is None:
            batchROIs = roi
        else:
            batchROIs = np.vstack([batchROIs, roi])

        batchLocs.append((x, y))

        if len(batchROIs) == BATCH_SIZE:
            labels = classify_batch(model, batchROIs, batchLocs, labels, minProb=args["confidence"])

            batchROIs = None
            batchLocs = []

if batchROIs is not None:
    labels = classify_batch(model, batchROIs, batchLocs, labels, minProb=args["confidence"])

end = time.time()
print("[INFO] detections took {:.4f} seconds".format(end - start))

for k in labels.keys():
    clone = resized.copy()

    for (box, prob) in labels[k]:
        (xA, yA, xB, yB) = box
        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 255, 0), 2)

    cv2.imshow("Without NMS", clone)
    clone = resized.copy()

    boxes = np.array([p[0] for p in labels[k]])
    proba = np.array([p[1] for p in labels[k]])
    boxes = non_max_suppression(boxes, proba)

    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(clone, (xA, yA), (xB, yB), (0, 0, 255), 2)

    print("[INFO] {}: {}".format(k, len(boxes)))
    cv2.imshow("With NMS", clone)
    cv2.waitKey(0)

