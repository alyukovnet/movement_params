from movement_params.frame_processors import FrameProcessor
from movement_params.frame import Frame, FrameObject
from movement_params.io.output import WindowOutput
from movement_params import CONFIG
from time import time


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
            if len(obj.movement_params) > 1:
                obj2_params = obj.movement_params[-2]

                prevx, prevy = obj2_params.coordinates
                curx, cury = obj.box.center
                curspeedx, curspeedy = curx-prevx, cury-prevy
                obj.speed = ((curspeedx**2 + curspeedy**2)**0.5)/timebetweenframes
                if obj2_params.speed is not None:
                    obj.acceleration = (obj.speed - obj2_params.speed)/timebetweenframes
        # self.last_objects = frame.objects
        return frame
