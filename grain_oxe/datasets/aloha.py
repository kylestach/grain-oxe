import numpy as np

from grain_oxe.datasets.dataset_spec import (
    ActionType,
    DatasetSpec,
    EmbodimentType,
    ImageType,
    ProprioType,
)


def _right_left_joints(data, prefix=""):
    return {
        f"{prefix}joints_right": data[..., :6],
        f"{prefix}joints_left": data[..., 7:13],
        f"{prefix}joints": np.concatenate([data[..., :6], data[..., 7:13]], axis=-1),
    }


def _gripper(data, prefix=""):
    return {
        f"{prefix}gripper_right": data[..., 6:7],
        f"{prefix}gripper_left": data[..., 13:14],
        f"{prefix}gripper": np.concatenate([data[..., 6:7], data[..., 13:14]], axis=-1),
    }


def aloha_transform(data):
    return {
        "action": {
            **_right_left_joints(data["action"], "absolute_"),
            **_right_left_joints(data["action"], "chunk_relative_"),
            **_gripper(data["action"]),
        },
        "observation": {
            "image": {
                "base": data["observation"]["cam_high"],
                "secondary": data["observation"]["cam_low"],
                "wrist_left": data["observation"]["cam_left_wrist"],
                "wrist_right": data["observation"]["cam_right_wrist"],
            },
            "proprio": {
                **_right_left_joints(data["observation"]["state"]),
                **_gripper(data["observation"]["state"]),
            },
        },
        "language": {
            "global_instruction": data["base_step"]["global_instruction"],
            "local_instruction": data["base_step"]["clip_instruction"],
        },
    }


ALOHA_SPEC = DatasetSpec(
    embodiment_name="aloha",
    embodiment_type=EmbodimentType.BIMANUAL,
    supported_image_types=[
        ImageType.BASE,
        ImageType.SECONDARY,
        ImageType.WRIST_LEFT,
        ImageType.WRIST_RIGHT,
    ],
    supported_action_types=[
        ActionType.ABSOLUTE_JOINT,
        ActionType.RELATIVE_JOINT,
        ActionType.CHUNK_RELATIVE_JOINT,
    ],
    supported_proprio_types=[ProprioType.JOINT, ProprioType.GRIPPER],
    standardization_transform=aloha_transform,
    dt=0.02,
)
