from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from movement_params.io import Input, Output, StreamInput, WindowOutput
from movement_params.frame_processors import FrameProcessor, ObjectsDetector, ObjectsTracker, ObjectsIdentifier, \
    PositionCalculator, ParamsCalculator


@dataclass
class Config:
    input_type: Input
    output_type: Optional[Output]
    processors: list[FrameProcessor]


processors = [
    ObjectsDetector(),
    # ObjectsTracker(),
    # ObjectsIdentifier(),
    # PositionCalculator(),
    # ParamsCalculator(),
]


def DefaultConfig():
    return Config(
        input_type=StreamInput(),
        output_type=WindowOutput(),
        processors=processors
    )


def DebugConfig():
    return Config(
        input_type=StreamInput('./cars1.mp4'),
        output_type=WindowOutput(),
        processors=processors
    )
