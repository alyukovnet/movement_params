from movement_params.frame_processors import FrameProcessor
from movement_params.frame import Frame, FrameObject
from movement_params.io.output import WindowOutput
from movement_params import CONFIG
from time import time
import cv2
import numpy as np


def middle(a: float, b: float, c: float):
    if a <= b and a <= c:
        if b <= c:
            mid = b
        else:
            mid = c
    else:
        if b <= a and b <= c:
            if a <= c:
                mid = a
            else:
                mid = c
        else:
            if a <= b:
                mid = a
            else:
                mid = b
    return mid


class ParamsCalculator(FrameProcessor):
    def __init__(self):
        # self.last_objects: list[FrameObject] = []
        self.__sec = time()

    def process(self, frame: Frame) -> Frame:
        timebetweenframes = (time() - self.__sec)
        self.__sec = time()
        for obj in frame.objects:
            # for obj2 in self.last_objects:
            #     if obj.obj_id == obj2.obj_id:
            if len(obj.movement_params) > 2:
                obj2_params = obj.movement_params[-3]

                prevx, prevy = obj2_params.coordinates
                curx, cury = obj.box.center
                curspeedx, curspeedy = curx - prevx, cury - prevy
                obj.speed = ((curspeedx ** 2 + curspeedy ** 2) ** 0.5) / timebetweenframes
                midspeed = middle(obj.movement_params[-3].speed, obj.movement_params[-2].speed,
                                  obj.movement_params[-1].speed)
                k = 0.2
                if abs(obj.speed - midspeed) > 40:
                    k = 0.5
                obj.speed = (obj.speed - midspeed) * k + midspeed
                k = 0.2
                # k - коэф, больше - точнее, меньше - плавнее
                if obj2_params.speed is not None:
                    obj.acceleration = (obj.speed - obj2_params.speed) / timebetweenframes
                    midacc = middle(obj.movement_params[-3].acceleration, obj.movement_params[-2].acceleration,
                                    obj.movement_params[-1].acceleration)
                    if abs(obj.acceleration - midacc) > 200:
                        k = 0.3
                    obj.acceleration = (obj.acceleration - midacc) * k + \
                                       midacc
                    k = 0.2
        # self.last_objects = frame.objects
        # prediction
        # sumx, sumy, sumx2, sumxy = 0, 0, 0, 0
        # for obj in frame.objects:
        #     n = 5
        #     if len(obj.movement_params) > 6:
        #         last = [obj.movement_params[-5].coordinates, obj.movement_params[-4].coordinates,
        #                 obj.movement_params[-3].coordinates, obj.movement_params[-2].coordinates,
        #                 obj.movement_params[-1].coordinates]
        #         for i in last:
        #             sumx += i[0]
        #             sumy += i[1]
        #             sumx2 += i[0]**2
        #             sumxy += i[0]*i[1]
        #         a = (n * sumxy - sumx * sumy)/(n * sumx2 - sumx**2)
        #         b = (sumy - a * sumx)/n
        #         x, y = obj.coord
        #         x += 100
        #         y = int(a*x + b)
        return frame
