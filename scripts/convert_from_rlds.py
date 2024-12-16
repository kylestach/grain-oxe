import argparse
import json
import os

import array_record.python.array_record_module as array_record
import jax
import msgpack_numpy as msgpack
import numpy as np
import tensorflow_datasets as tfds
import tqdm

MIN_SHARD_SIZE = 100


def get_steps(trajectory, start_idx: int, traj_id: int):
    """
    Get the trajectory as an iterator over steps.
    """
    traj_len = jax.tree.leaves(trajectory["steps"])[0].shape[0]
    global_data = {k: v for k, v in trajectory.items() if k != "steps"}
    traj_start_idx = start_idx
    traj_end_idx = traj_start_idx + traj_len

    for i in range(traj_len):
        yield (
            start_idx + i,
            {
                **global_data,
                "step": jax.tree.map(lambda x: x[i], trajectory["steps"]),
                "step_metadata": {
                    "frame_idx": i,
                    "traj_id": traj_id,
                    "traj_start_idx": traj_start_idx,
                    "traj_end_idx": traj_end_idx,
                    "traj_len": traj_len,
                },
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="bridge_dataset")
    parser.add_argument("--in_dir", type=str, default="/data/rlds")
    parser.add_argument("--out_dir", type=str, default="/data/rlds_ar")
    parser.add_argument("--num_shards", type=int, default=1024)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    in_dir = args.in_dir
    out_dir = args.out_dir
    os.makedirs(os.path.join(out_dir, dataset_name), exist_ok=True)

    builder = tfds.builder(dataset_name, data_dir=in_dir)

    key = 0
    num_steps = 0

    metadata = {
        "dataset_name": dataset_name,
        "splits": {},
    }

    for split in builder.info.splits.keys():
        dataset = builder.as_dataset(
            split=split, decoders={"steps": tfds.decode.SkipDecoding()}
        )

        num_shards = args.num_shards
        trajs_per_shard = (len(dataset) + num_shards - 1) // num_shards
        episode_starts = []
        episode_ends = []

        writers = [
            array_record.ArrayRecordWriter(
                os.path.join(
                    out_dir,
                    dataset_name,
                    f"{dataset_name}-{split}.array_record.{i:05d}-of-{num_shards:05d}",
                ),
                "group_size:1",
            )
            for i in range(num_shards)
        ]

        for traj_id, trajectory in tqdm.tqdm(
            enumerate(dataset.as_numpy_iterator()),
            total=len(dataset),
            desc=f"Processing {split} split",
        ):
            shard_idx = traj_id // trajs_per_shard

            start = num_steps
            for key, step in get_steps(trajectory, key, traj_id):
                serialized = msgpack.packb(step)
                writers[shard_idx].write(serialized)
                num_steps += 1
            end = num_steps
            episode_starts.append(start)
            episode_ends.append(end)

        for writer in writers:
            writer.close()

        np.savez(
            os.path.join(
                out_dir, dataset_name, f"{dataset_name}_episode_index_{split}.npz"
            ),
            episode_starts=np.array(episode_starts),
            episode_ends=np.array(episode_ends),
        )
        metadata["splits"][split] = {
            "num_episodes": len(episode_starts),
            "num_steps": num_steps,
            "num_shards": num_shards,
            "episode_index_path": f"{dataset_name}_episode_index_{split}.npz",
            "shard_paths": [
                f"{dataset_name}-{split}.array_record.{i:05d}-of-{num_shards:05d}"
                for i in range(num_shards)
            ],
        }

    with open(
        os.path.join(out_dir, dataset_name, f"{dataset_name}_metadata.json"), "w"
    ) as f:
        json.dump(metadata, f)
