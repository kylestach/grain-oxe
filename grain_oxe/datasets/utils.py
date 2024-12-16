import numpy as np
from scipy.spatial.transform import Rotation as R


def rpy_to_quat(rpy: np.ndarray) -> np.ndarray:
    return R.from_euler("xyz", rpy, degrees=False).as_quat()


def make_chunk_relative(
    chunk: np.ndarray, *, base_step_proprio: np.ndarray
) -> np.ndarray:
    return chunk - base_step_proprio
