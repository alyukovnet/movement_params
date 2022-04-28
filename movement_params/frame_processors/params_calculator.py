from movement_params.frame_processors import FrameProcessor
from movement_params.frame import Frame, FrameObject
from movement_params.io.output import WindowOutput
from movement_params import CONFIG
from time import time


class ParamsCalculator(FrameProcessor):
    def __init__(self):
        self.last_objects: list[FrameObject] = []
        self.__sec = time()

    def process(self, frame: Frame) -> Frame:
        timebetweenframes = (time() - self.__sec)
        self.__sec = time()
        for obj in frame.objects:
            for obj2 in self.last_objects:
                if obj.obj_id == obj2.obj_id:
                    prevx, prevy = obj2.box.center
                    curx, cury = obj.box.center
                    curspeedx, curspeedy = curx-prevx, cury-prevy
                    obj.speed = ((curspeedx**2 + curspeedy**2)**0.5)/timebetweenframes
                    obj.acceleration = obj.speed - obj2.speed
        self.last_objects = frame.objects
        return frame
