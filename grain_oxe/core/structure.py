import dataclasses
from abc import ABC
from typing import Any, Callable, Dict

import grain.python as grain
import jax
import numpy as np
import numpy.ma as npma

from grain_oxe.core.sampling_spec import TrajSampleSpec


class SampleFramesTransform(grain.RandomMapTransform):
    def __init__(
        self, sample_fn: Callable[[TrajSampleSpec, np.random.Generator], TrajSampleSpec]
    ):
        self.sample_fn = sample_fn

    def random_map(self, traj_sample_spec: TrajSampleSpec, rng: np.random.Generator):
        return self.sample_fn(traj_sample_spec, rng)


class RestructureTransform(grain.MapTransform):
    def __init__(self, restructure_fn: Callable[[Any], Dict[str, Any]]):
        self.restructure_fn = restructure_fn

    def map(self, data):
        return self.restructure_fn(data)


class DatasetStructure(ABC):
    def sample(
        self, traj_sample_spec: TrajSampleSpec, rng: np.random.Generator | None
    ) -> TrajSampleSpec: ...

    def restructure(self, data) -> Dict[str, Any]: ...

    def sample_transform(self) -> grain.RandomMapTransform:
        return SampleFramesTransform(self.sample)

    def restructure_transform(self) -> grain.MapTransform:
        return RestructureTransform(self.restructure)


class BCDatasetStructure(DatasetStructure):
    def __init__(self, num_obs_steps: int, num_action_steps: int):
        self.num_obs_steps = num_obs_steps
        self.num_action_steps = num_action_steps

    def sample(self, traj_sample_spec: TrajSampleSpec, rng: np.random.Generator = None):
        base_frame = traj_sample_spec.base_index
        obs_frames = (base_frame + np.arange(1 - self.num_obs_steps, 1)).tolist()
        action_frames = (base_frame + np.arange(0, self.num_action_steps)).tolist()
        return dataclasses.replace(
            traj_sample_spec,
            frames={
                "obs_frames": obs_frames,
                "action_frames": action_frames,
                "base_frame": base_frame,
            },
        )

    def restructure(self, data):
        return {
            "observation": jax.tree.map(
                lambda *xs: npma.stack(xs),
                *[d["observation"] for d in data["steps"]["obs_frames"]],
            ),
            "action": jax.tree.map(
                lambda *xs: npma.stack(xs),
                *[d["action"] for d in data["steps"]["action_frames"]],
            ),
            "base_step": data["steps"]["base_frame"],
            "episode_metadata": data["episode_metadata"],
        }


class GCBCDatasetStructure(DatasetStructure):
    def __init__(self, num_obs_steps: int, num_action_steps: int):
        self.num_obs_steps = num_obs_steps
        self.num_action_steps = num_action_steps

    def sample(self, traj_sample_spec: TrajSampleSpec, rng: np.random.Generator):
        base_frame = traj_sample_spec.base_index
        goal_frame = rng.integers(base_frame + 1, traj_sample_spec.traj_len)
        obs_frames = (base_frame + np.arange(-self.num_obs_steps, 1)).tolist()
        action_frames = (base_frame + np.arange(0, self.num_action_steps)).tolist()
        return dataclasses.replace(
            traj_sample_spec,
            frames={
                "obs_frames": obs_frames,
                "action_frames": action_frames,
                "goal_frame": goal_frame,
                "base_frame": base_frame,
            },
        )

    def restructure(self, data):
        return {
            "observation": jax.tree.map(
                lambda *xs: npma.stack(xs),
                *[d["observation"] for d in data["steps"]["obs_frames"]],
            ),
            "action": jax.tree.map(
                lambda *xs: npma.stack(xs),
                *[d["action"] for d in data["steps"]["action_frames"]],
            ),
            "task": data["task"] | {"goal": data["steps"]["goal_frame"]},
            "episode_metadata": data["episode_metadata"],
        }


class RLDatasetStructure(DatasetStructure):
    def __init__(self, num_obs_steps: int, rl_stride: int = 1):
        self.num_obs_steps = num_obs_steps
        self.rl_stride = rl_stride

    def sample(self, traj_sample_spec: TrajSampleSpec, rng: np.random.Generator):
        base_frame = traj_sample_spec.base_index
        obs_frames = base_frame + np.arange(1 - self.num_obs_steps, 1)
        next_obs_frames = obs_frames + self.rl_stride
        action_frames = base_frame + np.arange(0, self.rl_stride)
        next_action_frames = action_frames + self.rl_stride
        return dataclasses.replace(
            traj_sample_spec,
            frames={
                "obs_frames": obs_frames.tolist(),
                "next_obs_frames": next_obs_frames.tolist(),
                "action_frames": action_frames.tolist(),
                "next_action_frames": next_action_frames.tolist(),
            },
        )

    def restructure(self, data):
        return {
            "observation": jax.tree.map(
                lambda *xs: npma.stack(xs),
                *[d["observation"] for d in data["steps"]["obs_frames"]],
            ),
            "next_observation": jax.tree.map(
                lambda *xs: npma.stack(xs),
                *[d["observation"] for d in data["steps"]["next_obs_frames"]],
            ),
            "action": jax.tree.map(
                lambda *xs: npma.stack(xs),
                *[d["action"] for d in data["steps"]["action_frames"]],
            ),
            "next_action": jax.tree.map(
                lambda *xs: npma.stack(xs),
                *[d["action"] for d in data["steps"]["next_action_frames"]],
            ),
            "task": {
                "language_instruction": data["steps"]["action_frames"][0][
                    "language_instruction"
                ],
                "reward": npma.sum(
                    [d["reward"] for d in data["steps"]["action_frames"]]
                ),
                "is_terminal": npma.any(
                    [
                        npma.filled(d["is_terminal"], fill_value=True)
                        for d in data["steps"]["next_obs_frames"]
                    ]
                ),
            },
        }


DATASET_STRUCTURES = {
    "bc": BCDatasetStructure,
    "gcbc": GCBCDatasetStructure,
    "rl": RLDatasetStructure,
}
