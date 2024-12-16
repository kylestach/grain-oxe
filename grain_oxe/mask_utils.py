import jax
import numpy as np
from grain.python import MapTransform


def make_masked(x, mask):
    x = np.asarray(x)
    return np.ma.MaskedArray(x, mask=np.full(x.shape, mask))


class ExtractMaskTransform(MapTransform):
    def map(self, masked_data):
        data = jax.tree_util.tree_map(
            lambda x: x.data if isinstance(x, np.ma.MaskedArray) else x, masked_data
        )
        masks = jax.tree_util.tree_map(
            lambda x: ~x.mask if isinstance(x, np.ma.MaskedArray) else None, masked_data
        )
        return {"data": data, "mask": masks}
