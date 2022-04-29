from __future__ import annotations

from movement_params.frame_processors import FrameProcessor
from movement_params.frame import Frame, BoundingBox, FrameObject, ObjectType

import cv2


class MovementDetector(FrameProcessor):
    def __init__(self):
        self.prev_frame: Frame | None = None

    def process(self, frame: Frame) -> Frame:
        # source: https://myrusakov.ru/python-opencv-move-detection.html

        if self.prev_frame is None:
            self.prev_frame = frame
            return frame

        diff = cv2.absdiff(self.prev_frame.image, frame.image)

        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

        dilated = cv2.dilate(thresh, None, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        boxes: set[BoundingBox] = set()

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            box = BoundingBox(x, y, x + w, y + h)

            if box.area < 10000:
                continue

            boxes.add(box)

        boxes_to_delete: set[BoundingBox] = set()

        # delete intersections
        for box1 in boxes:
            for box2 in boxes:
                if box1 == box2:
                    continue

                if box1.belongs_to(box2) == 1:
                    boxes_to_delete.add(box1)

        boxes -= boxes_to_delete

        for box in boxes:
            frame_object = FrameObject(box, ObjectType.CAR)

            frame.objects.append(frame_object)

        self.prev_frame = frame
        return frame
