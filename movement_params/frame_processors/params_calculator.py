from movement_params.frame_processors import FrameProcessor
from movement_params.frame import Frame


class ParamsCalculator(FrameProcessor):
    def __init__(self):
        pass

    def process(self, frame: Frame) -> Frame:
        for o in frame.objects:
            pass

        return frame
