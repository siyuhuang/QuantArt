import bisect
import os
import numpy as np
import albumentations
from PIL import Image,ImageFile
from torch.utils.data import Dataset, ConcatDataset
import csv
import random
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ConcatDatasetWithIndex(ConcatDataset):
    """Modified from original pytorch code to return dataset idx"""
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx

class UnpairedImagePaths(Dataset):
    def __init__(self, size=None, random_crop=False, folder1=None, folder2=None, numpy_folder1=None, numpy_folder2=None,
                 wikiart_info1=None, wikiart_key1=None, wikiart_info2=None, wikiart_key2=None):
        self.size = size
        self.random_crop = random_crop
        self.numpy_folder1 = numpy_folder1
        self.numpy_folder2 = numpy_folder2
        
        ### paths of folder 1 ###
        paths1 = []
        numpy_paths1 = []
        if wikiart_info1 is not None and wikiart_key1 is not None:
            with open(wikiart_info1, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['genre'] == wikiart_key1:
                        paths1.append(os.path.join(folder1, row['filename']))
        else:
            paths_ = os.listdir(folder1)
            for path in paths_:
                paths1.append(os.path.join(folder1, path))
        if numpy_folder1 is not None:
            image_paths1 = []
            for image_path in paths1:
                path = os.path.basename(image_path)
                numpy_path = os.path.join(numpy_folder1, path+'.npy')
                if os.path.exists(numpy_path):
                    image_paths1.append(image_path)
                    numpy_paths1.append(numpy_path)
            paths1 = image_paths1
                
        ### paths of folder 2 ###
        paths2 = []
        numpy_paths2 = []
        if wikiart_info2 is not None and wikiart_key2 is not None:
            with open(wikiart_info2, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    if row['genre'] == wikiart_key2:
                        paths2.append(os.path.join(folder2, row['filename']))
        else:
            paths_ = os.listdir(folder2)
            for path in paths_:
                paths2.append(os.path.join(folder2, path))
        if numpy_folder2 is not None:
            image_paths2 = []
            for image_path in paths2:
                path = os.path.basename(image_path)
                numpy_path = os.path.join(numpy_folder2, path+'.npy')
                if os.path.exists(numpy_path):
                    image_paths2.append(image_path)
                    numpy_paths2.append(numpy_path)
            paths2 = image_paths2
        
        if len(paths2) > 1: 
            self._length = min(len(paths1), len(paths2)) # len(paths1) 
        else: # single style in a folder
            paths2 = paths2 * len(paths1)
            self._length = len(paths1)
        self.paths1 = paths1
        self.paths2 = paths2
        self.numpy_paths1 = numpy_paths1
        self.numpy_paths2 = numpy_paths2
        
        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.Resize(height=self.size,width=self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs
            
    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image, image_path

    def __getitem__(self, i):
        example = dict()
        example["image1"], example["image1_path"] = self.preprocess_image(self.paths1[i])
        i2 = random.randrange(len(self.paths2))
        example["image2"], example["image2_path"] = self.preprocess_image(self.paths2[i2])
        if self.numpy_folder1 is not None:
            example["numpy1"] = np.load(self.numpy_paths1[i])
        if self.numpy_folder2 is not None:
            example["numpy2"] = np.load(self.numpy_paths2[i2])
        return example
    
class PairedImagePaths(Dataset):
    def __init__(self, paths=None, size=None, random_crop=False, labels=None, folder1=None, folder2=None):
        self.size = size
        self.random_crop = random_crop
        
        paths_ = []
        if paths is not None:
            for path in paths:
                if ',' in path:
                    p = path.split(',')
                p1 = p[1]
                p2 = p[0]
                if folder1 is not None:
                    p1 = os.path.join(folder1, p1)
                if folder2 is not None:
                    p2 = os.path.join(folder2, p2)
                paths_.append([p1, p2])
        else:
            paths1 = sorted(os.listdir(folder1))
            paths2 = sorted(os.listdir(folder2))
            l = min(len(paths1), len(paths2))
            for i in range(l):
                p1 = os.path.join(folder1, paths1[i])
                p2 = os.path.join(folder2, paths1[i])
                paths_.append([p1, p2])
                
        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths_
        self._length = len(paths_)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.Resize(height=self.size,width=self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image1"] = self.preprocess_image(self.labels["file_path_"][i][0])
        example["image2"] = self.preprocess_image(self.labels["file_path_"][i][1])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example

class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None):
        self.size = size
        self.random_crop = random_crop

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        if self.size is not None and self.size > 0:
            self.rescaler = albumentations.SmallestMaxSize(max_size = self.size)
            if not self.random_crop:
                self.cropper = albumentations.CenterCrop(height=self.size,width=self.size)
            else:
                self.cropper = albumentations.RandomCrop(height=self.size,width=self.size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
        else:
            self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


class NumpyPaths(ImagePaths):
    def preprocess_image(self, image_path):
        image = np.load(image_path).squeeze(0)  # 3 x 1024 x 1024
        image = np.transpose(image, (1,2,0))
        image = Image.fromarray(image, mode="RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        return image
