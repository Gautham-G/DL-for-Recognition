import glob
import os
from typing import Tuple

import numpy as np
from PIL import Image
from PIL import ImageOps
# from sklearn.preprocessing import StandardScaler


def compute_mean_and_std(dir_name: str) -> Tuple[float, float]:
    """Compute the mean and the standard deviation of all images present within the directory.

    Note: convert the image in grayscale and then in [0,1] before computing mean
    and standard deviation

    Mean = (1/n) * \sum_{i=1}^n x_i
    Variance = (1 / (n-1)) * \sum_{i=1}^n (x_i - mean)^2
    Standard Deviation = sqrt( Variance )

    Args:
        dir_name: the path of the root dir

    Returns:
        mean: mean value of the gray color channel for the dataset
        std: standard deviation of the gray color channel for the dataset
    """
    mean = None
    std = None


    main_dir = os.listdir(dir_name)
    xi = []
    n = 0
    # var_list = []
    # scaler = StandardScaler()

    for i in range(len(main_dir)):
        directory = dir_name + main_dir[i]
        sub_directory = os.listdir(directory)

        for j in range(len(sub_directory)):
            image_address = directory + '/' + sub_directory[j] + '/*'

            for f in glob.iglob(image_address):
                image_gray = np.asarray(ImageOps.grayscale(Image.open(f)))
                image_scaled = image_gray/255
                image_as_array = image_scaled.flatten()
                xi.append(np.sum(image_as_array))
                n += image_as_array.shape[0]

    xi = np.array(xi)
    mean = np.sum(xi)/n

    var_list = []

    for i in range(len(main_dir)):
        directory = dir_name + main_dir[i]
        sub_directory = os.listdir(directory)

        for j in range(len(sub_directory)):
            image_address = directory + '/' + sub_directory[j] + '/*'

            for f in glob.iglob(image_address):
                image_gray = np.asarray(ImageOps.grayscale(Image.open(f)))
                image_scaled = image_gray/255
                image_as_array = image_scaled.flatten()
                var_list.append(np.sum((image_as_array - mean)**2))

    var_list = np.array(var_list)
    var = np.sum(var_list)/n
    std = var**0.5

    return mean, std
