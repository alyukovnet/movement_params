from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from movement_params.io import Input, Output, StreamInput, WindowOutput, VideoFileInput
from movement_params.frame_processors import FrameProcessor, ObjectsDetector, ObjectsTracker, ObjectsIdentifier, \
    PositionCalculator, ParamsCalculator, MovementDetector


@dataclass
class Config:
    input_type: Input
    output_type: Optional[Output]
    processors: list[FrameProcessor]


processors = [
    # ObjectsDetector(),
    MovementDetector(),
    ObjectsTracker(),
    # ObjectsIdentifier(),
    # PositionCalculator(),
    ParamsCalculator(0),
]


def DefaultConfig() -> Config:
    """
    Default config
    """
    return Config(
        input_type=StreamInput(),  # Camera 0
        output_type=WindowOutput(),
        processors=processors
    )


def DebugConfig() -> Config:
    """
    Debug config
    """
    return Config(
        input_type=StreamInput('./cars1.mp4'),  # Video file with frames skipping
        # input_type=VideoFileInput('./cars1.mp4'),  # Frame-by-frame video input
        output_type=WindowOutput(),
        processors=processors
    )
