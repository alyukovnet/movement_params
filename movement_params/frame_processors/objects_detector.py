from pathlib import Path
from cv2 import cv2
import logging

from movement_params.frame import BoundingBox, ObjectType, FrameObject, Frame
from movement_params.frame_processors import FrameProcessor
import movement_params.model


class ObjectsDetector(FrameProcessor):
    """
    Objects detector
    Adds objects on Frame
    """
    def __init__(self):
        CFG_PATH = Path(movement_params.model.__file__).parent / 'yolov4.cfg'
        WEIGHTS_PATH = Path(movement_params.model.__file__).parent / 'yolov4.weights'

        net = cv2.dnn.readNetFromDarknet(str(CFG_PATH), str(WEIGHTS_PATH))

        self.__model = cv2.dnn_DetectionModel(net)
        self.__model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

    def process(self, frame: Frame) -> Frame:
        classIds, scores, boxes = self.__model.detect(frame.image, confThreshold=0.5, nmsThreshold=0.5)

        for classId, score, box in zip(classIds, scores, boxes):
            if classId != ObjectType.CAR:
                # logging.warning(f'Detected not a car ({classId})')
                continue

            box = BoundingBox(box[0], box[1], box[0] + box[2], box[1] + box[3])

            frame_object = FrameObject(box, ObjectType.CAR)

            frame.objects.append(frame_object)

        return frame
