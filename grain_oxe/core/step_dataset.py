from typing import Any, Mapping

import grain.python as grain
import jax
import msgpack_numpy as msgpack
import numpy as np
from grain._src.python.data_sources import ArrayRecordDataSource

from grain_oxe.core.sampling_spec import TrajSampleSpec
from grain_oxe.mask_utils import make_masked


class EpisodeIndexDataSource(grain.MapDataset):
    def __init__(self, episode_starts, episode_ends):
        super().__init__()
        total_num_steps = episode_ends[-1]
        step = np.arange(total_num_steps)
        is_episode_start = np.isin(step, episode_starts)
        self.traj_index = np.cumsum(is_episode_start) - 1
        self.traj_start = episode_starts[self.traj_index]
        self.traj_end = episode_ends[self.traj_index]

    def __getitem__(self, i) -> TrajSampleSpec:
        traj_id = self.traj_index[i]
        traj_start = self.traj_start[i]
        traj_end = self.traj_end[i]
        return TrajSampleSpec(
            traj_id=traj_id,
            traj_len=traj_end - traj_start,
            traj_start=traj_start,
            traj_end=traj_end,
            base_index=i - traj_start,
            frames=i - traj_start,
        )

    def __getitems__(self, idcs):
        return [self.__getitem__(i) for i in idcs]

    def __len__(self):
        return len(self.traj_index)


class EpisodeStepsDataSource(grain.MapDataset):
    def __init__(self, dataset_name: str, shard_paths: list[str]):
        super().__init__()
        self.dataset_name = dataset_name
        self.source = ArrayRecordDataSource(shard_paths)

    def __getitem__(self, i) -> TrajSampleSpec:
        data = msgpack.unpackb(self.source[i])
        data["episode_metadata"]["dataset_name"] = self.dataset_name
        return data

    def __getitems__(self, idcs):
        data = []
        for e in self.source.__getitems__(idcs):
            e = msgpack.unpackb(e)
            e["episode_metadata"]["dataset_name"] = self.dataset_name
            data.append(e)
        return data

    def __len__(self):
        return len(self.source)


def idcs_for_spec(traj_sample_spec: TrajSampleSpec):
    """
    Get the needed global frame indices for a trajectory sample spec.
    """
    frame_idcs, _ = jax.tree_util.tree_flatten(traj_sample_spec.frames)
    frame_idcs = np.array(frame_idcs) + traj_sample_spec.traj_start
    clipped_frame_idcs = np.clip(
        frame_idcs, traj_sample_spec.traj_start, traj_sample_spec.traj_end - 1
    )
    return np.unique(clipped_frame_idcs)


def data_for_spec(
    traj_sample_spec: TrajSampleSpec,
    retrieved_frames: Mapping[int, Any],
):
    frame_idcs, frame_pytree_def = jax.tree_util.tree_flatten(traj_sample_spec.frames)
    frame_idcs = np.array(frame_idcs) + traj_sample_spec.traj_start
    clipped_frame_idcs = np.clip(
        frame_idcs, traj_sample_spec.traj_start, traj_sample_spec.traj_end - 1
    )

    frame_valid_masks = [
        traj_sample_spec.traj_start <= idx and idx < traj_sample_spec.traj_end
        for idx in frame_idcs
    ]
    frames = [
        (
            jax.tree.map(
                lambda x: make_masked(x, not mask),
                retrieved_frames[idx]["step"],
            )
        )
        for mask, idx in zip(frame_valid_masks, clipped_frame_idcs)
    ]
    frame_pytree = jax.tree_util.tree_unflatten(frame_pytree_def, frames)
    non_step_data = jax.tree.map(
        lambda x: make_masked(x, False),
        {
            k: v
            for k, v in retrieved_frames[
                traj_sample_spec.base_index + traj_sample_spec.traj_start
            ].items()
            if k != "steps"
        },
    )
    return {
        **non_step_data,
        "steps": frame_pytree,
        "episode_metadata": {
            "base_index": make_masked(traj_sample_spec.base_index, False),
            "traj_id": make_masked(traj_sample_spec.traj_id, False),
            "traj_len": make_masked(traj_sample_spec.traj_len, False),
            **non_step_data.get("episode_metadata", {}),
        },
    }


class EpisodeLookup(grain.MapTransform):
    def __init__(self, source: ArrayRecordDataSource):
        super().__init__()
        self.source = source

    def map(self, traj_sample_spec: TrajSampleSpec):
        required_frame_idcs = idcs_for_spec(traj_sample_spec)
        retrieved_frames = {
            idx: v
            for idx, v in zip(
                required_frame_idcs,
                self.source.__getitems__(required_frame_idcs.tolist()),
            )
        }
        return data_for_spec(traj_sample_spec, retrieved_frames)


class EpisodeLookupDataSource(grain.MapDataset):
    def __init__(
        self, index_source: EpisodeIndexDataSource, source: ArrayRecordDataSource
    ):
        super().__init__(parents=[index_source, source])
        self.index_source = index_source
        self.source = source

    def __getitems__(self, idcs):
        specs = [self.index_source[i] for i in idcs]
        required_frame_idcs = [i for spec in specs for i in idcs_for_spec(spec)]
        retrieved_frames = {
            idx: v
            for idx, v in zip(
                required_frame_idcs,
                self.source.__getitems__(required_frame_idcs),
            )
        }

        return [data_for_spec(spec, retrieved_frames) for spec in specs]

    def __getitem__(self, i):
        return self.__getitems__([i])[0]

    def __len__(self):
        return len(self.index_source)
