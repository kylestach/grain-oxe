import fnmatch
import io
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, OrderedDict, Tuple

import jax
import numpy as np
import numpy.ma as npma
from grain.python import RandomMapTransform
from PIL import Image, ImageEnhance


def decode_image(image_bytes: np.ndarray):
    return Image.open(io.BytesIO(image_bytes))


def random_crop(images: List[Image.Image], scale: float, rng: np.random.Generator):
    scale = rng.uniform(scale, 1.0)

    width, height = images[0].size
    crop_width = int(width / scale)
    crop_height = int(height / scale)
    crop_left = rng.integers(0, max(1, width - crop_width))
    crop_top = rng.integers(0, max(1, height - crop_height))
    return [
        image.crop(
            (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)
        )
        for image in images
    ]


def random_hsv(
    images: List[Image.Image],
    hue_shift: Tuple[float, float] | float,
    saturation_scale: Tuple[float, float] | float,
    value_scale: Tuple[float, float] | float,
    rng: np.random.Generator,
):
    hue_shift = (-hue_shift, hue_shift) if isinstance(hue_shift, float) else hue_shift
    saturation_scale = (
        (1 - saturation_scale, 1 + saturation_scale)
        if isinstance(saturation_scale, float)
        else saturation_scale
    )
    value_scale = (
        (1 - value_scale, 1 + value_scale)
        if isinstance(value_scale, float)
        else value_scale
    )

    hue_shift = rng.uniform(*hue_shift)
    saturation_scale = rng.uniform(*saturation_scale)
    value_scale = rng.uniform(*value_scale)

    shift = (hue_shift, 0, 0)
    scale = (1, saturation_scale, value_scale)

    def transform_fn(image: Image.Image):
        image = image.convert("HSV")
        image = np.array(image).astype(np.float32)
        image *= scale
        image = np.clip(image, 0, 255)
        image += shift
        image %= 360
        image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        return image.convert("RGB")

    return [transform_fn(image) for image in images]


def random_contrast(
    images: List[Image.Image],
    contrast_scale: Tuple[float, float] | float,
    rng: np.random.Generator,
):
    contrast_scale = (
        (contrast_scale, contrast_scale)
        if isinstance(contrast_scale, float)
        else contrast_scale
    )
    contrast_scale = rng.uniform(*contrast_scale)

    def transform_fn(image: Image.Image):
        return ImageEnhance.Contrast(image).enhance(contrast_scale)

    return [transform_fn(image) for image in images]


AUGMENT_OPS = {
    "random_crop": random_crop,
    "random_hsv": random_hsv,
    "random_contrast": random_contrast,
}


@dataclass
class ImageConfig:
    resize_size: Tuple[int, int]
    augmentations: OrderedDict[str, Dict[str, Any]]
    resize_first: bool = False

    def apply(self, images: List[Image.Image], rng: np.random.Generator):
        if self.resize_first and self.resize_size is not None:
            images = [
                (
                    image.resize(self.resize_size)
                    if image.size != self.resize_size
                    else image
                )
                for image in images
            ]

        for key, kwargs in self.augmentations.items():
            images = AUGMENT_OPS[key](images, **kwargs, rng=rng)

        if self.resize_size is not None:
            images = [
                (
                    image.resize(self.resize_size)
                    if image.size != self.resize_size
                    else image
                )
                for image in images
            ]

        return images


class ImageAugmentationTransform(RandomMapTransform):
    def __init__(self, **kwargs):
        super().__init__()
        self.image_key_to_config = {k: ImageConfig(**v) for k, v in kwargs.items()}

    def random_map(
        self,
        data,
        rng: np.random.Generator,
    ):
        def _decode_and_transform(path, maybe_image_bytes, rng):
            for image_key, config in self.image_key_to_config.items():
                if fnmatch.fnmatch(path[-1].key, image_key):
                    original_mask = maybe_image_bytes.mask
                    images = [decode_image(x) for x in maybe_image_bytes.data]
                    images = config.apply(images, rng)
                    images = np.stack(images)
                    images_valid = np.any(images != 0, axis=(-3, -2, -1), keepdims=True)
                    image_mask = original_mask[..., None, None, None] | ~images_valid
                    image_mask = np.broadcast_to(image_mask, images.shape)
                    return npma.MaskedArray(images, mask=image_mask) / 127.5 - 1.0
            return maybe_image_bytes

        return jax.tree_util.tree_map_with_path(
            partial(_decode_and_transform, rng=rng), data
        )
