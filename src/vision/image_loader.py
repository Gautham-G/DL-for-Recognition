"""
Script with Pytorch's dataloader class
"""

import glob
import os
from tkinter import image_names
from typing import Dict, List, Tuple

import torch
import torch.utils.data as data
import torchvision
from PIL import Image
import csv
import pandas as pd


class ImageLoader(data.Dataset):
    """Class for data loading"""

    train_folder = "train"
    test_folder = "test"

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: torchvision.transforms.Compose = None,
    ):
        """Initialize the dataloader and set `curr_folder` for the corresponding data split.

        Args:
            root_dir: the dir path which contains the train and test folder
            split: 'test' or 'train' split
            transform: the composed transforms to be applied to the data
        """
        self.root = os.path.expanduser(root_dir)
        self.transform = transform
        self.split = split

        if split == "train":
            self.curr_folder = os.path.join(root_dir, self.train_folder)
        elif split == "test":
            self.curr_folder = os.path.join(root_dir, self.test_folder)

        self.class_dict = self.get_classes()
        self.dataset = self.load_imagepaths_with_labels(self.class_dict)

    def load_imagepaths_with_labels(
        self, class_labels: Dict[str, int]
    ) -> List[Tuple[str, int]]:
        """Fetches all (image path,label) pairs in the dataset.

        Args:
            class_labels: the class labels dictionary, with keys being the classes in this dataset
        Returns:
            list[(filepath, int)]: a list of filepaths and their class indices
        """

        img_paths = []  # a list of (filename, class index)


        for class_key, class_index in class_labels.items():
            # print(class_key, class_index)
            img_dir = os.path.join(self.curr_folder, class_key, '*.jpg')
            # print(img_dir, 'dir')
            files = glob.glob(img_dir)
            # print()
            # print(files, 'glob')
            img_paths += [(f, class_index) for f in files]



        return img_paths

    def get_classes(self) -> Dict[str, int]:
        """Get the classes (which are folder names in self.curr_folder)

        NOTE: Please make sure that your classes are sorted in alphabetical order
        i.e. if your classes are ['apple', 'giraffe', 'elephant', 'cat'], the
        class labels dictionary should be:
        {"apple": 0, "cat": 1, "elephant": 2, "giraffe":3}

        If you fail to do so, you will most likely fail the accuracy
        tests on Gradescope

        Returns:
            Dict of class names (string) to integer labels
        """

        classes = dict()

        classes_list = [folder.name for folder in os.scandir(self.curr_folder) if folder.is_dir()]
        classes_list.sort()
        classes = {classes_list[idx]: idx for idx in range(len(classes_list))}
        classes = dict(sorted(classes.items()))

        return classes

    def load_img_from_path(self, path: str) -> Image:
        """Loads an image as grayscale (using Pillow).

        Note: do not normalize the image to [0,1]

        Args:
            path: the file path to where the image is located on disk
        Returns:
            image: grayscale image with values in [0,255] loaded using pillow
                Note: Use 'L' flag while converting using Pillow's function.
        """

        img = None

        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('L')

        return img

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Fetches the item (image, label) at a given index.

        Note: Do not forget to apply the transforms, if they exist

        Hint:
        1) get info from self.dataset
        2) use load_img_from_path
        3) apply transforms if valid

        Args:
            index: Index
        Returns:
            img: image of shape (H,W)
            class_idx: index of the ground truth class for this image
        """
        img = None
        class_idx = None


        file_name, class_idx = self.dataset[index]
        img_temp = self.load_img_from_path(file_name)
        # if self.transform is not None:
        img = self.transform(img_temp)

        return img, class_idx

    def __len__(self) -> int:
        """Returns the number of items in the dataset.

        Returns:
            l: length of the dataset
        """


        l = len(self.dataset)

        return l


class MultiLabelImageLoader(data.Dataset):
    """Class for data loading"""

    train_folder = "train"
    test_folder = "test"

    def __init__(
        self,
        root_dir: str,
        labels_csv: str,
        split: str = "train",
        transform: torchvision.transforms.Compose = None,
    ):
        """Initialize the dataloader and set `curr_folder` for the corresponding data split.

        Args:
            root_dir: the dir path which contains the train and test folder
            labels_csv: the path to the csv file containing ground truth labels
            split: 'test' or 'train' split
            transform: the composed transforms to be applied to the data
        """
        self.root = os.path.expanduser(root_dir)
        self.labels_csv = labels_csv
        self.transform = transform
        self.split = split

        if split == "train":
            self.curr_folder = os.path.join(root_dir, self.train_folder)
        elif split == "test":
            self.curr_folder = os.path.join(root_dir, self.test_folder)

        self.dataset = self.load_imagepaths_with_labels()

    def load_imagepaths_with_labels(self) -> List[Tuple[str, torch.Tensor]]:
        """Fetches all (image path,labels) pairs in the dataset from csv file. Ensure that only
        the images from the classes in ['coast', 'highway', 'mountain', 'opencountry', 'street']
        are included. 

        NOTE: Be mindful of the returned labels type

        Returns:
            list[(filepath, list(int))]: a list of filepaths and their labels
        """

        img_paths = []  # a list of (filename, class index)


        columns_csv = ['Place', 'Name_img', 'label0', 'label1', 'label2', 'label3', 'label4', 'label5', 'label6']
        data_from_csv = pd.read_csv(self.labels_csv, names = columns_csv)
        
        for index, row in data_from_csv.iterrows():
            img_paths.append( [self.curr_folder + '/' + str(row['Place']) + '/' + str(row['Name_img']), torch.Tensor( [int(row['label0']), int(row['label1']), int(row['label2']), int(row['label3']), int(row['label4']), int(row['label5']), int(row['label6']) ] ) ])


        return img_paths


    def load_img_from_path(self, path: str) -> Image:
        """Loads an image as grayscale (using Pillow).

        Note: do not normalize the image to [0,1]

        Args:
            path: the file path to where the image is located on disk
        Returns:
            image: grayscale image with values in [0,255] loaded using pillow
                Note: Use 'L' flag while converting using Pillow's function.
        """

        img = None

        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('L')



        return img

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Fetches the item (image, label) at a given index.

        
        1) get info from self.dataset
        2) use load_img_from_path
        3) apply transforms if valid

        Args:
            index: Index
        Returns:
            img: image of shape (H,W)
            class_idx: index of the ground truth class for this image
        """
        img = None
        class_idx = None


        file_name, class_idx = self.dataset[index]
        img_temp = self.load_img_from_path(file_name)
        # if self.transform is not None:
        img = self.transform(img_temp)

        return img, class_idx

    def __len__(self) -> int:
        """Returns the number of items in the dataset.

        Returns:
            l: length of the dataset
        """


        l = len(self.dataset)

        return l
