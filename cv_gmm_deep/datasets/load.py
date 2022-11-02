from glob import glob
from os import path

import numpy as np

import imageio
from skimage import img_as_float32
from natsort import natsorted
from tqdm.auto import tqdm

prefixes = ['images', 'masks']


def load_data(input_dir, height, width):
    data = tuple()

    for prefix in prefixes:
        data_dir = path.join(input_dir, f'{prefix}-{width}x{height}')
        data_paths = natsorted(glob(f'{data_dir}/*.png'))
        raw_data = np.array([
            img_as_float32(imageio.imread(path))
            for path in tqdm(data_paths, f'Reading in {prefix}')
        ])
        data += (raw_data)

    return data
