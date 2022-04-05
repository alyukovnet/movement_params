"""
Software for calculating movement parameters of objects on video stream
"""
from movement_params.CONFIG import Config, DefaultConfig, DebugConfig
from movement_params.frame import BoundingBox, FrameObject, Frame
from movement_params.io import *
from movement_params.frame_processors import *

import sys


CONFIG: Config = DefaultConfig() if '--debug' not in sys.argv else DebugConfig()
