from dataclasses import dataclass, field
from typing import Optional
from numpy import ndarray, array


@dataclass
class StaticConfig:
    world_matrix: Optional[ndarray] = field(default_factory=lambda: array([[10, 10], [20, 10], [20, 20], [10, 20]]))
    camera_matrix: Optional[ndarray] = field(default_factory=lambda: array([[1, 0], [0, 0], [1, 0], [1, 1]]))
    aruco_ids: Optional[ndarray] = field(default_factory=lambda: array([61, 62, 63, 60]))


default_static = StaticConfig()
