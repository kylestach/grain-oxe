from typing import Any, Dict, List

from grain.python import MapDataset

from grain_oxe.datasets.create import create_standardized_dataset
from grain_oxe.datasets.normalization import Normalizer
from grain_oxe.structure import DATASET_STRUCTURES


def make_mixture_dataset(
    *,
    data_dir: str,
    normalizer: Normalizer,
    dataset_names: List[str],
    dataset_structure: str,
    default_dataset_structure_kwargs: Dict[str, Any],
    dataset_structure_kwargs: Dict[str, Any],
    dataset_weights: Dict[str, float],
    split: str = "train",
    seed: int = 0,
) -> MapDataset:
    datasets = []
    weights = []

    dataset_structure_cls = DATASET_STRUCTURES[dataset_structure]

    for dataset_name in dataset_names:
        datasets.append(
            create_standardized_dataset(
                dataset_name,
                data_dir,
                dataset_structure=dataset_structure_cls(
                    **(default_dataset_structure_kwargs | dataset_structure_kwargs)
                ),
                split=split,
                seed=seed,
                normalizer=normalizer,
            ).repeat(None)
        )
        weights.append(dataset_weights[dataset_name] * len(datasets[-1]))

    return MapDataset.mix(
        datasets,
        weights,
    )
