from __future__ import annotations

from enum import EnumMeta
from typing import Generator, Optional
import cv2
import numpy as np
from datetime import datetime


class BoundingBox:
    """
    Position and dimensions of bounding box on __frame

    Point 1 - left top corner of box (x1 - left border, y1 - top border)

    Point 2 - right bottom corner of box (x2 - right border, y2 - left border)
    """

    x1: int  # Left
    y1: int  # Top
    x2: int  # Right
    y2: int  # Bottom

    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2

    @property
    def h(self) -> int:
        """
        Height of box
        """
        return self.y2-self.y1

    @property
    def w(self) -> int:
        """
        Width of box
        """
        return self.x2-self.x1

    @property
    def p1(self) -> tuple[int, int]:
        """
        Point 1 (left top)
        """
        return self.x1, self.y1

    @property
    def p2(self) -> tuple[int, int]:
        """
        Point 2 (right bottom)
        """
        return self.x2, self.y2

    @property
    def center(self) -> tuple[int, int]:
        """
        Center point
        """
        return (self.x2 - self.x1) // 2, (self.y2 - self.y1) // 2

    @property
    def area(self) -> int:
        """
        Area
        """
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def iou(self, box2: BoundingBox) -> float:
        """
        Get intersection-over-union value
        :param box2: Box2
        :return: Intersection-over-union value
        """
        xa = max(self.x1, box2.x1)
        ya = max(self.y1, box2.y1)
        xb = min(self.x2, box2.x2)
        yb = min(self.y2, box2.y2)

        intersection_area = max(0, xb - xa) * max(0, yb - ya)

        iou = intersection_area / (box2.area + self.area - intersection_area)
        return iou

    def belongs_to(self, box2: BoundingBox) -> float:
        """
        Part of object belongs to box2
        :param box2: Box2
        :return: Part
        """
        xa = max(self.x1, box2.x1)
        ya = max(self.y1, box2.y1)
        xb = min(self.x2, box2.x2)
        yb = min(self.y2, box2.y2)

        interArea = max(0, xb - xa) * max(0, yb - ya)

        return float(interArea) / float(self.area)

    def __iter__(self) -> Generator[int]:
        """
        Get box values (Left, Top, Right, Bottom)
        :return: Generator of 4 values (x1, y1, x2, y2)
        """
        yield self.x1
        yield self.y1
        yield self.x2
        yield self.y2


class ObjectType:
    __metaclass__ = EnumMeta
    PERSON = 0
    CAR = 2  # Num from coco.names


class FrameObject:
    """
    Object on Frame
    """
    prev_objects: list[FrameObject] = []
    obj_id: Optional[int] = None
    __box: BoundingBox
    __type: ObjectType

    def __init__(self, box: BoundingBox, object_type: ObjectType):
        self.__box: BoundingBox = box
        self.__type: ObjectType = object_type

    @property
    def box(self):
        return self.__box

    @property
    def type(self):
        return self.__type


class Frame:
    """
    Frame with informative payload
    """
    created: datetime  # Creation time
    objects: list[FrameObject]  # Objects on frame
    __info: str = ''
    __image: np.ndarray

    def __init__(self, frame: np.ndarray):
        """
        Creates Frame object from numpy image
        :param frame: Numpy image
        """
        # self.__image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.__image = frame
        self.created = datetime.now()
        self.objects = []

    @property
    def image(self) -> np.ndarray:
        """
        Get numpy image
        :return: Numpy image
        """
        return self.__image

    @property
    def info_image(self) -> np.ndarray:
        image = self.__image.copy()
        for o in self.objects:
            cv2.rectangle(image, o.box.p1, o.box.p2, (0, 255, 0), 2)
            cv2.putText(image, f'{o.obj_id}', (o.box.x1, o.box.y1 + 40), 0, 0.7, (0, 255, 0))

        cv2.putText(image, self.__info, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        return image

    def put_info(self, s: str):
        self.__info = s


__all__ = [BoundingBox, ObjectType, FrameObject, Frame]
