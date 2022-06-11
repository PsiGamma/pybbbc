"""
Functions for working with images
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def correct_illumination(
    images: np.ndarray, sigma=500, min_percentile=0.02
) -> np.ndarray:
    img_avg = images.mean(axis=0)
    img_mask = gaussian_filter(img_avg.astype(np.float32), sigma=sigma).astype(
        np.float16
    )
    robust_min = np.percentile(img_mask[img_mask > 0], min_percentile)
    img_mask[img_mask < robust_min] = robust_min
    img_mask = img_mask / robust_min
    return images / img_mask


def scale_pixel_intensity(images: np.ndarray) -> np.ndarray:
    low = np.percentile(images, 0.1)
    high = np.percentile(images, 99.9)
    images = (images - low) / (high - low)
    return np.clip(images, 0, 1)


def anscombe_transform(image: np.ndarray) -> np.ndarray:
    a = 2 * np.sqrt(image + (3.0/8.0))
    return a

def normalize_pixel_intensity_dmso(images: np.ndarray, mean: float, std: float) -> np.ndarray:
    images = (images - mean) / std
    return images

def get_dmso_statistics(images: np.ndarray) -> list:
    stats = [images.ravel().mean(), images.ravel().std()]
    return stats

def convert_to_8bit_range(images: np.ndarray):
    # We convert individual 2D images to an 8-bit range
    # Note: this does NOT convert values to integers
    # assuming we get an image stack
    if images.ndim == 2:
        # make into stack
        shape = images.shape
        images = images.reshape(1,shape[0],shape[1])
    n_images = len(images)
    for i in range(n_images):
        img = images[i,:,:]
        min=img.min()
        max=img.max()
        scale = 256.0/(max-min+1);
        img = img - min
        img[img<0] = 0
        img = (img*scale)+0.5
        img[img>255] = 255
        images[i,:,:] = img

    return images.squeeze()
