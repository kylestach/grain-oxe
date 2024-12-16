import json
import os

import numpy as np
from grain.python import MapDataset

from grain_oxe.core.step_dataset import (
    EpisodeIndexDataSource,
    EpisodeLookupDataSource,
    EpisodeStepsDataSource,
)
from grain_oxe.core.structure import DatasetStructure


def create_dataset(
    dataset_name: str,
    data_dir: str,
    dataset_structure: DatasetStructure,
    split: str = "train",
    seed: int | None = None,
) -> MapDataset:
    # Load metadata
    with open(
        os.path.join(data_dir, dataset_name, f"{dataset_name}_metadata.json"), "r"
    ) as f:
        metadata = json.load(f)

    # Index dataset
    ep_idx = np.load(
        os.path.join(
            data_dir, dataset_name, metadata["splits"][split]["episode_index_path"]
        )
    )

    episode_index_source = EpisodeIndexDataSource(
        ep_idx["episode_starts"], ep_idx["episode_ends"]
    )
    episode_index_source = episode_index_source.random_map(
        dataset_structure.sample_transform(),
        seed=seed,
    )

    # Look up indices in the real dataset
    episode_steps_source = EpisodeStepsDataSource(
        dataset_name,
        [
            os.path.join(data_dir, dataset_name, shard_path)
            for shard_path in metadata["splits"][split]["shard_paths"]
        ],
    )
    dataset = EpisodeLookupDataSource(episode_index_source, episode_steps_source)

    return dataset.map(dataset_structure.restructure_transform())
