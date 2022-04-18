from collections import defaultdict

from movement_params.frame_processors import FrameProcessor
from movement_params.frame import Frame, FrameObject

import cv2


class ObjectsTracker(FrameProcessor):
    IOU_THRESHOLD = 0.7
    obj_id_inc = 0

    def __init__(self):
        self.last_objects: list[FrameObject] = []

    def process(self, frame: Frame) -> Frame:
        for obj in frame.objects:
            max_iou = -1
            max_iou_obj2 = None

            for obj2 in self.last_objects:
                if (iou := obj.box.iou(obj2.box)) > max_iou and iou > self.IOU_THRESHOLD:
                    max_iou = iou
                    max_iou_obj2 = obj2

            if max_iou_obj2 is not None:
                obj.obj_id = max_iou_obj2.obj_id
            else:
                obj.obj_id = self.obj_id_inc
                self.obj_id_inc += 1

        self.last_objects = frame.objects

        return frame
