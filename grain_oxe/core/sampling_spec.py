from dataclasses import dataclass
from typing import Any


@dataclass
class TrajSampleSpec:
    traj_id: int
    traj_len: int
    traj_start: int
    traj_end: int
    base_index: int

    # Some PyTree of integers that will get replaced with the actual frames
    frames: Any
