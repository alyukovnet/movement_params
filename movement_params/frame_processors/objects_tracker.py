from movement_params.frame_processors import FrameProcessor
from movement_params.frame import Frame


class ObjectsTracker(FrameProcessor):
    def __init__(self):
        pass

    def process(self, frame: Frame) -> Frame:
        return frame
