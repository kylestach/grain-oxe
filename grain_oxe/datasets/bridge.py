from grain_oxe.datasets.dataset_spec import (
    ActionType,
    DatasetSpec,
    EmbodimentType,
    ImageType,
    ProprioType,
)


def bridge_transform(data):
    return {
        "action": {
            "relative_cartesian_euler": data["action"][..., :-1],
            "gripper": data["action"][..., -1:],
        },
        "observation": {
            "image": {
                "base": data["observation"]["image_0"],
            },
            "proprio": {
                "cartesian": data["observation"]["state"][..., :6],
                "gripper": data["observation"]["state"][..., 6:],
            },
        },
        "language": {
            "global_instruction": data["base_step"]["language_instruction"],
        },
    }


BRIDGE_SPEC = DatasetSpec(
    embodiment_name="bridge",
    embodiment_type=EmbodimentType.SINGLE_ARM,
    supported_image_types=[ImageType.BASE, ImageType.WRIST],
    supported_action_types=[ActionType.RELATIVE_CARTESIAN_EULER],
    supported_proprio_types=[ProprioType.CARTESIAN, ProprioType.GRIPPER],
    standardization_transform=bridge_transform,
    dt=0.2,
)
