import os, sys
import os.path as path
import tarfile as tr
from urllib.request import urlretrieve

import tensorflow as tf
from object_detection.utils import label_map_util, config_util
from object_detection.builders import model_builder

import cv2 as cv
import numpy as np
from time import time
from random import randint
from typing import Iterable


colors: dict[str, tuple[int, int, int]] = dict()


def visualize_boxes_and_labels(
    image: np.ndarray,
    boxes: Iterable[Iterable[float]],
    class_ids: Iterable[int],
    scores: Iterable[float],
    class_names: dict[int, str],
) -> np.ndarray:
    hg, wd = image.shape[:2]

    for (ymin, xmin, ymax, xmax), cls_id, score in zip(boxes, class_ids, scores):
        perc = int(score * 100)
        if perc < 60:
            continue

        xmin, ymin = int(xmin * wd), int(ymin * hg)
        xmax, ymax = int(xmax * wd), int(ymax * hg)
        name = class_names[cls_id]

        if name in colors:
            color = colors[name]
        else:
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            while (True not in [(pxl > 200) for pxl in color]) or (
                color in colors.values()
            ):
                color = (randint(0, 255), randint(0, 255), randint(0, 255))
            colors[name] = color

        name = f"{name} {perc}%"
        font = cv.FONT_HERSHEY_COMPLEX_SMALL

        gts = cv.getTextSize(name, font, 2.0, 2)
        gtx = gts[0][0] + xmin
        gty = gts[0][1] + ymin

        cv.rectangle(image, (xmin, ymin), (xmax, ymax), color, 4)
        cv.rectangle(
            image, (xmin, ymin), (min(gtx + 3, wd), min(gty + 4, hg)), color, -1
        )
        cv.putText(image, name.capitalize(), (xmin, gty), font, 2.0, (0, 0, 0), 2)

    return image


class DetectionModel(object):
    def __init__(self, dataDir: str = "data") -> None:
        self.DATA_DIR = dataDir
        self.MODELS_DIR = path.join(self.DATA_DIR, "models")

        self.MODEL_DATE = "20200711"
        self.MODEL_NAME = "ssd_mobilenet_v2_320x320_coco17_tpu-8"  # "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
        self.MODEL_TAR_FILENAME = self.MODEL_NAME + ".tar.gz"

        self.MODELS_DOWNLOAD_BASE = (
            "http://download.tensorflow.org/models/object_detection/tf2/"
        )
        self.MODEL_DOWNLOAD_LINK = (
            self.MODELS_DOWNLOAD_BASE + self.MODEL_DATE + "/" + self.MODEL_TAR_FILENAME
        )

        self.PATH_TO_MODEL_TAR = path.join(self.MODELS_DIR, self.MODEL_TAR_FILENAME)
        self.PATH_TO_CKPT = path.join(self.MODELS_DIR, self.MODEL_NAME, "checkpoint/")
        self.PATH_TO_CFG = path.join(
            self.MODELS_DIR, self.MODEL_NAME, "pipeline.config"
        )

        self.LABEL_FILENAME = "mscoco_label_map.pbtxt"
        self.LABELS_DOWNLOAD_BASE = "https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/"
        self.PATH_TO_LABELS = path.join(
            self.MODELS_DIR, self.MODEL_NAME, self.LABEL_FILENAME
        )

        self.model = None

    def createModelDir(self) -> None:
        for dir in [self.DATA_DIR, self.MODELS_DIR]:
            if not path.exists(dir):
                os.mkdir(dir)

    def downloadModel(self) -> None:
        if not path.exists(self.PATH_TO_CKPT):
            print("Downloading model. This may take a while... ", end="")
            urlretrieve(self.MODEL_DOWNLOAD_LINK, self.PATH_TO_MODEL_TAR)
            tar_file = tr.open(self.PATH_TO_MODEL_TAR)
            tar_file.extractall(self.MODELS_DIR)
            tar_file.close()
            os.remove(self.PATH_TO_MODEL_TAR)
            print("Done")

    def downloadLabels(self) -> None:
        if not path.exists(self.PATH_TO_LABELS):
            print("Downloading label file... ", end="")
            urlretrieve(
                self.LABELS_DOWNLOAD_BASE + self.LABEL_FILENAME, self.PATH_TO_LABELS
            )
            print("Done")

    def loadModel(self) -> None:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        tf.get_logger().setLevel("ERROR")

        gpus = tf.config.experimental.list_physical_devices("GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        configs = config_util.get_configs_from_pipeline_file(self.PATH_TO_CFG)
        model_config = configs["model"]
        self.model = model_builder.build(model_config=model_config, is_training=False)

        ckpt = tf.compat.v2.train.Checkpoint(model=self.model)
        ckpt.restore(path.join(self.PATH_TO_CKPT, "ckpt-0")).expect_partial()

    def prepareAll(self) -> None:
        self.createModelDir()
        self.downloadModel()
        self.downloadLabels()
        self.loadModel()


dm = DetectionModel()
dm.prepareAll()


@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = dm.model.preprocess(image)
    prediction_dict = dm.model.predict(image, shapes)
    detections = dm.model.postprocess(prediction_dict, shapes)
    return (detections, prediction_dict, tf.reshape(shapes, [-1]))


category_index = label_map_util.create_category_index_from_labelmap(
    dm.PATH_TO_LABELS, use_display_name=True
)
names = {vl["id"]: vl["name"] for vl in category_index.values()}

cap = cv.VideoCapture(0)
prevTime = 0

if not cap.isOpened():
    sys.exit("Could not open the camera")

while True:
    ret, image_np = cap.read()
    if not ret:
        print("Ignoring empty camera frame")
        continue

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections, predictions_dict, shapes = detect_fn(input_tensor)

    boxes = detections["detection_boxes"][0].numpy()
    classes = (detections["detection_classes"][0].numpy() + 1).astype(int)
    scores = detections["detection_scores"][0].numpy()

    visualize_boxes_and_labels(image_np, boxes, classes, scores, names)

    currTime = time()
    fps = f"FPS: {round(1 / (currTime - prevTime), 1)}"
    prevTime = currTime

    cv.putText(
        image_np, fps, (5, 35), cv.FONT_HERSHEY_COMPLEX_SMALL, 2.0, (255, 0, 0), 2
    )

    cv.imshow("Object Detection", image_np)
    if cv.waitKey(2) == 27:  # esc
        break

cap.release()
cv.destroyAllWindows()
