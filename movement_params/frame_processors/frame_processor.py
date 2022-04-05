from abc import ABCMeta, abstractmethod

from movement_params.frame import Frame


class FrameProcessor:
    """
    Frame processor
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def process(self, frame: Frame) -> Frame:
        pass
