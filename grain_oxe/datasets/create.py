from typing import Any, Dict

from grain.python import MapTransform

from grain_oxe.core.create import create_dataset
from grain_oxe.datasets.dataset_spec import DatasetSpec
from grain_oxe.datasets.datasets import DATASETS
from grain_oxe.datasets.normalization import Normalizer
from grain_oxe.structure import DatasetStructure


class StandardizationTransform(MapTransform):
    def __init__(self, dataset_spec: DatasetSpec):
        self.dataset_spec = dataset_spec

    def map(self, data: Dict[str, Any]) -> Dict[str, Any]:
        metadata = data.get(
            "metadata",
            {
                "embodiment_name": self.dataset_spec.embodiment_name,
                "dataset_name": data["episode_metadata"]["dataset_name"],
                "base_index": data["episode_metadata"]["base_index"],
                "traj_id": data["episode_metadata"]["traj_id"],
                "traj_len": data["episode_metadata"]["traj_len"],
            },
        )
        data = self.dataset_spec.standardization_transform(data)
        data["metadata"] = metadata
        return data


class NormalizationTransform(MapTransform):
    def __init__(self, normalizer: Normalizer):
        self.normalizer = normalizer

    def map(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return self.normalizer.normalize(data["metadata"]["embodiment_name"], data)


def create_standardized_dataset(
    dataset_name: str,
    data_dir: str,
    *,
    dataset_structure: DatasetStructure,
    split: str,
    seed: int,
    normalizer: Normalizer | None = None,
):
    # Create the base dataset
    dataset = create_dataset(dataset_name, data_dir, dataset_structure, split, seed)

    # Standardize the dataset
    standardization_transform = StandardizationTransform(DATASETS[dataset_name])
    dataset = dataset.map(standardization_transform)

    # Normalize the dataset
    if normalizer is not None:
        dataset = dataset.map(NormalizationTransform(normalizer))

    return dataset
