import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from cv2 import cv2
from time import time

from movement_params.frame import Frame


class Output:
    """
    Output interface
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def window_title(self):
        pass

    @abstractmethod
    def push(self, frame: Frame):
        pass


class PhotoOutput(Output):
    """
    Output to photo file
    """
    def __init__(self, photo_path: Path, multiple: bool = True):
        """
        Create Photo Output object
        Uses postfix _{created} to
        :param photo_path: Path to image file
        :param multiple: If true, uses _{created} after stem of path
        """
        self.__path = photo_path
        if multiple:
            self.path = lambda timestamp: self.__path.with_stem(f'{self.__path.stem}_{timestamp}')
        else:
            self.path = lambda _: self.__path

    def push(self, frame: Frame):
        cv2.imwrite(self.path(int(frame.created.timestamp())), frame.image)


class VideoFileOutput(Output):
    def __init__(self):
        logging.warning('Video File Output now is not supported')

    def push(self, frame: Frame):
        logging.warning('Video File Output now is not supported')


class WindowOutput(Output):
    def __init__(self, window_name: str = 'Processed frame'):
        self.__window_name = window_name
        self.__sec = time()
        cv2.namedWindow(self.__window_name, cv2.WINDOW_AUTOSIZE)

    @property
    def window_title(self):
        return self.__window_name

    def push(self, frame: Frame):
        delta: float = time() - self.__sec
        fps = .0 if delta == .0 else 1 / delta
        self.__sec = time()
        frame.put_info(f'{fps:0.3}')
        cv2.imshow(self.__window_name, frame.info_image)
        button_code = cv2.pollKey()
        if button_code == 27:  # escape pressed
            cv2.destroyAllWindows()
