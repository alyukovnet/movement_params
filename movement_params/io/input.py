from __future__ import annotations

from abc import ABCMeta, abstractmethod
from pathlib import Path
from cv2 import cv2
from threading import Thread
import numpy as np
from time import time

from movement_params.frame import Frame


class Input:
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_frame(self) -> Frame:
        pass


class PhotoInput(Input):
    """
    Get picture as frame
    """
    def __init__(self, photo_path: Path):
        self.__frame = cv2.imread(str(photo_path))

    def get_frame(self) -> Frame:
        return Frame(self.__frame)


class StreamInput(Input):
    """
    Get frames from stream (or emulated stream from file with frames skipping)
    """
    def __init__(self, stream: int | str = 0):
        self.__cap = cv2.VideoCapture(stream)

        if not self.__cap.isOpened():
            exit("VideoCapture is not opened")

        self.__cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.__width = self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.__height = self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.__fps = self.__cap.get(cv2.CAP_PROP_FPS)
        self.__delay = 1 / self.__fps

        self.__frame: np.ndarray = None
        self.__ret = None
        self.__thread_kill: bool = False

        self.__thread = Thread(target=self.__background, args=(), name="reading_thread")
        self.__thread.daemon = True
        self.__thread.start()

    def __background(self):
        sec = 0
        while not self.__thread_kill:
            if time() - sec < self.__delay:
                continue
            sec = time()

            ret, frame = self.__cap.read()

            if ret is not None and frame is not None:
                self.__ret, self.__frame = ret, frame

    def get_frame(self) -> Frame:
        if self.__ret is not None and self.__frame is not None:
            return Frame(self.__frame)


class VideoFileInput(Input):
    """
    Frame-by-frame video input
    """
    def __init__(self, video_path: str | Path):
        self.__cap = cv2.VideoCapture(str(video_path))

        if not self.__cap.isOpened():
            exit("VideoCapture is not opened")

        self.__width = self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.__height = self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.__fps = self.__cap.get(cv2.CAP_PROP_FPS)

    def get_frame(self) -> Frame:
        ret, frame = self.__cap.read()

        if ret is not None and frame is not None:
            return Frame(frame)
