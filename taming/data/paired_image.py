import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import PairedImagePaths


class PairedImageBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example

class PairedImageTrain(PairedImageBase):
    def __init__(self, size, training_images_list_file=None, folder1=None, folder2=None):
        super().__init__()
        if training_images_list_file is not None:
            with open(training_images_list_file, "r") as f:
                paths = f.read().splitlines()
        else:
            paths = None
        self.data = PairedImagePaths(paths=paths, folder1=folder1, folder2=folder2, size=size, random_crop=False)


class PairedImageTest(PairedImageBase):
    def __init__(self, size, test_images_list_file=None, folder1=None, folder2=None):
        super().__init__()
        if test_images_list_file is not None:
            with open(test_images_list_file, "r") as f:
                paths = f.read().splitlines()
        else:
            paths = None
        self.data = PairedImagePaths(paths=paths, folder1=folder1, folder2=folder2, size=size, random_crop=False)


