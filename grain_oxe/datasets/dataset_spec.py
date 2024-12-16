from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence


class ImageType(Enum):
    BASE = "base"
    SECONDARY = "secondary"
    WRIST = "wrist"
    WRIST_LEFT = "wrist_left"
    WRIST_RIGHT = "wrist_right"


class ActionType(Enum):
    RELATIVE_CARTESIAN_QUAT = "relative_cartesian_quat"
    ABSOLUTE_CARTESIAN_QUAT = "absolute_cartesian_quat"
    CHUNK_RELATIVE_CARTESIAN_QUAT = "chunk_relative_cartesian_quat"

    RELATIVE_CARTESIAN_EULER = "relative_cartesian_euler"
    ABSOLUTE_CARTESIAN_EULER = "absolute_cartesian_euler"
    CHUNK_RELATIVE_CARTESIAN_EULER = "chunk_relative_cartesian_euler"

    RELATIVE_JOINT = "relative_joint"
    ABSOLUTE_JOINT = "absolute_joint"
    CHUNK_RELATIVE_JOINT = "chunk_relative_joint"

    GRIPPER = "gripper"


class EmbodimentType(Enum):
    SINGLE_ARM = "single_arm"
    BIMANUAL = "bimanual"


class ProprioType(Enum):
    CARTESIAN = "cartesian"
    JOINT = "joint"
    GRIPPER = "gripper"


@dataclass
class DatasetSpec:
    embodiment_name: str

    embodiment_type: EmbodimentType
    supported_image_types: Sequence[ImageType]
    supported_action_types: Sequence[ActionType]
    supported_proprio_types: Sequence[ProprioType]

    standardization_transform: Callable

    dt: float
