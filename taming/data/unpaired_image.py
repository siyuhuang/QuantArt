import os
import numpy as np
import albumentations
from torch.utils.data import Dataset

from taming.data.base import UnpairedImagePaths


class UnpairedImageBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example

class UnpairedImageTrain(UnpairedImageBase):
    def __init__(self, size=None, random_crop=False, folder1=None, folder2=None, numpy_folder1=None, numpy_folder2=None, 
                 wikiart_info1=None, wikiart_key1=None, wikiart_info2=None, wikiart_key2=None,):
        super().__init__()
        self.data = UnpairedImagePaths(size=size, random_crop=random_crop, 
                   folder1=folder1, folder2=folder2, numpy_folder1=numpy_folder1, numpy_folder2=numpy_folder2, 
                 wikiart_info1=wikiart_info1, wikiart_key1=wikiart_key1, wikiart_info2=wikiart_info2, wikiart_key2=wikiart_key2)
        
class UnpairedImageVal(UnpairedImageBase):
    def __init__(self, size=None, random_crop=False, folder1=None, folder2=None, numpy_folder1=None, numpy_folder2=None, 
                 wikiart_info1=None, wikiart_key1=None, wikiart_info2=None, wikiart_key2=None):
        super().__init__()
        self.data = UnpairedImagePaths(size=size, random_crop=random_crop, 
                   folder1=folder1, folder2=folder2, numpy_folder1=numpy_folder1, numpy_folder2=numpy_folder2, 
                 wikiart_info1=wikiart_info1, wikiart_key1=wikiart_key1, wikiart_info2=wikiart_info2, wikiart_key2=wikiart_key2)
        self.data._length = min(self.data._length, 1000)
        
class UnpairedImageTest(UnpairedImageBase):
    def __init__(self, size=None, random_crop=False, folder1=None, folder2=None, numpy_folder1=None, numpy_folder2=None, 
                 wikiart_info1=None, wikiart_key1=None, wikiart_info2=None, wikiart_key2=None):
        super().__init__()
        self.data = UnpairedImagePaths(size=size, random_crop=random_crop, 
                   folder1=folder1, folder2=folder2, numpy_folder1=numpy_folder1, numpy_folder2=numpy_folder2, 
                 wikiart_info1=wikiart_info1, wikiart_key1=wikiart_key1, wikiart_info2=wikiart_info2, wikiart_key2=wikiart_key2)
        self.data._length = min(self.data._length, 1000)
        
        
        