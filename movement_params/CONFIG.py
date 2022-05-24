from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from numpy import ndarray, array

from movement_params.CONFIG_STATIC import StaticConfig
from movement_params.io import Input, Output, StreamInput, WindowOutput, VideoFileInput
from movement_params.frame_processors import FrameProcessor, ObjectsDetector, ObjectsTracker, ObjectsIdentifier, \
    PositionCalculator, ParamsCalculator, MovementDetector


@dataclass
class Config:
    input_type: Input
    output_type: Optional[Output] = field(default_factory=lambda: WindowOutput())
    static: StaticConfig = field(default_factory=lambda: StaticConfig)
    processors: list[FrameProcessor] = field(default_factory=lambda: processors)


processors = [
    # ObjectsDetector(),
    MovementDetector(),
    ObjectsTracker(),
    # ObjectsIdentifier(),
    PositionCalculator(),
    ParamsCalculator(1),
]


def DefaultConfig() -> Config:
    """
    Default config. Uses camera to apply algorithms in real-time.
    """
    return Config(
        input_type=StreamInput(),  # Camera 0
    )


def DebugConfig() -> Config:
    """
    Debug config. Uses a predetermined video file.
    """
    return Config(
        input_type=StreamInput('./cars1.mp4'),  # Video file with frames skipping
        # input_type=VideoFileInput('./cars1.mp4'),  # Frame-by-frame video input
    )
