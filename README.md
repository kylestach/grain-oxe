# grain-oxe
A dataloader framework for loading robotics data.

> [!WARNING]
> This is a work in progress and the API is subject to change.

`grain-oxe` is designed with the following goals:
- **Flexibility**: Support loading data for behavior cloning, RL, etc.
- **Mixed-modal data**: Support loading data where different datasets use different formats (e.g. action chunk lengths, etc.)
- **Masked data**: The dataloader is implemented using `np.ma.MaskedArray`, so masks (e.g. when passing the end of an episode) are preserved.
- **Performance**: Optimize for performance when loading data from disk.

## Installation
`grain-oxe` can be installed using `pip`:

```bash
pip install git+https://github.com/kylestach/grain-oxe.git
```

For development, setup is managed using [uv](https://docs.astral.sh/uv/getting-started/installation/).

## Usage
First, convert your data into a format that `grain-oxe` can load. We use [array-record](https://github.com/google/array_record) as the storage backend, with messages encoded using [msgpack-numpy](https://github.com/lebedov/msgpack-numpy):

```bash
python scripts/convert_from_rlds.py --data_dir /path/to/tensorflow/datasets --output_dir /path/to/output --dataset_name dataset_name
```

You can then load the dataset using `grain-oxe`:

```python
from grain_oxe.core import create_dataset, BCDatasetStructure

dataset = create_dataset(
    dataset_name,
    data_dir,
    dataset_structure=BCDatasetStructure(
        num_obs_steps=1,
        num_action_steps=10,
    ),
    split="train",
    seed=42,
)
```

`dataset` can then be used as a regular `grain.MapDataset`. You can apply additional transformations using `.map()`.

We recommend using `grain.DataLoader` to iterate over the dataset.

## Roadmap
 - [ ] Support image augmentations
 - [ ] Support padding for datasets with different lengths
 - [ ] Support loading from RLDS (https://github.com/google-research/rlds)