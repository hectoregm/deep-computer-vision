from tensorflow.keras.applications import imagenet_utils
import imutils


def sliding_window(image, step, ws):
    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            yield (x, y, image[y:y + ws[1], x:x + ws[0]])


def image_pyramid(image, scale=1.5, minSize=(224, 224)):
    yield image

    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        yield image


def classify_batch(model, batchROIs, batchLocs, labels, minProb=0.5, top=10, dims=(224, 224)):
    preds = model.predict(batchROIs)
    P = imagenet_utils.decode_predictions(preds, top=top)

    for i in range(0, len(P)):
        for (_, label, prob) in P[i]:
            if prob > minProb:
                (pX, pY) = batchLocs[i]
                box = (pX, pY, pX + dims[0], pY + dims[1])

                L = labels.get(label, [])
                L.append((box, prob))
                labels[label] = L

    return labels