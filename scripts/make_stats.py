import os
from collections import defaultdict
from dataclasses import replace
from typing import Any, Dict, List

import jax
import numpy as np
from tqdm import tqdm

from grain_oxe.core.sampling_spec import TrajSampleSpec
from grain_oxe.datasets import DATASETS, create_standardized_dataset
from grain_oxe.datasets.normalization import NormalizationType, Normalizer, Stats
from grain_oxe.structure import DatasetStructure

# For now, always normalize over a fixed chunk length
# (Only matters for chunk-relative actions)
FIXED_CHUNK_LENGTH = 10


class DummyStructure(DatasetStructure):
    def __init__(self, chunk_length: int):
        self.chunk_length = chunk_length

    def sample(
        self, traj_sample_spec: TrajSampleSpec, rng: np.random.Generator | None
    ) -> TrajSampleSpec:
        return replace(
            traj_sample_spec,
            frames={
                "base_step": traj_sample_spec.frames,
                "action_steps": (
                    traj_sample_spec.frames + np.ma.arange(self.chunk_length)
                ).tolist(),
            },
        )

    def restructure(self, data) -> Dict[str, Any]:
        return {
            "episode_metadata": data["episode_metadata"],
            "base_step": data["steps"]["base_step"],
            **data["steps"]["base_step"],
            "action": jax.tree.map(
                lambda *xs: np.ma.stack(xs, axis=0),
                *(d["action"] for d in data["steps"]["action_steps"]),
            ),
        }


def update_stats_for_dataset(
    embodiment_name: str,
    dataset_names: List[str],
    data_dir: str,
    stats_dir: str,
    chunk_length: int,
):
    def _get_lowdim_data(data) -> Dict[str, Any]:
        return (
            [data]
            if isinstance(data, (np.ndarray, np.ma.MaskedArray))
            and np.issubdtype(data.dtype, np.floating)
            else None
        )

    def _get_stat(data):
        data = np.ma.stack(data, axis=0)

        reduce_axes = tuple(range(0, data.ndim - 1))

        return Stats(
            mean=np.ma.mean(data, axis=reduce_axes),
            std=np.ma.std(data, axis=reduce_axes),
            min=np.ma.min(data, axis=reduce_axes),
            max=np.ma.max(data, axis=reduce_axes),
            p01=np.nanpercentile(data.filled(np.nan), 1, axis=reduce_axes),
            p05=np.nanpercentile(data.filled(np.nan), 5, axis=reduce_axes),
            p95=np.nanpercentile(data.filled(np.nan), 95, axis=reduce_axes),
            p99=np.nanpercentile(data.filled(np.nan), 99, axis=reduce_axes),
        )

    lowdim_data = None

    for dataset_name in dataset_names:
        dataset = create_standardized_dataset(
            dataset_name,
            data_dir,
            dataset_structure=DummyStructure(chunk_length=chunk_length),
            split="train",
            seed=0,
        )

        if lowdim_data is None:
            lowdim_data = jax.tree.map(_get_lowdim_data, dataset[0])

        for data in tqdm(dataset[1::100], desc=f"Processing {dataset_name}"):
            lowdim_data = jax.tree.map(
                lambda x, y: x + y,
                lowdim_data,
                jax.tree.map(_get_lowdim_data, data),
                is_leaf=lambda x: isinstance(x, list),
            )

    lowdim_data = jax.tree.map(
        _get_stat, lowdim_data, is_leaf=lambda x: isinstance(x, list)
    )

    Normalizer({embodiment_name: lowdim_data}, NormalizationType.MEAN_STD).save(
        stats_dir
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        action="append",
        required=True,
        help="Name of dataset to process",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Directory containing the dataset"
    )
    parser.add_argument(
        "--stats_dir", type=str, required=True, help="Directory to save stats to"
    )
    args = parser.parse_args()

    os.makedirs(args.stats_dir, exist_ok=True)
    embodiment_datasets_mapping = defaultdict(list)
    for dataset_name in args.datasets:
        embodiment_datasets_mapping[DATASETS[dataset_name].embodiment_name].append(
            dataset_name
        )

    for embodiment_name, dataset_names in embodiment_datasets_mapping.items():
        update_stats_for_dataset(
            embodiment_name,
            dataset_names,
            args.data_dir,
            args.stats_dir,
            FIXED_CHUNK_LENGTH,
        )
