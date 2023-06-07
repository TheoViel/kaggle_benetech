import re
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

from data.centernet import make_hm_regr


class CenterNetDataset(Dataset):
    """
    Image torch Dataset.
    """

    def __init__(
        self,
        df,
        transforms=None,
    ):
        """
        Constructor

        Args:
            paths (list): Path to images.
            transforms (albumentation transforms, optional): Transforms to apply. Defaults to None.
        """
        self.df = df
        self.paths = df["path"].values
        self.transforms = transforms
        self.gt_paths = df["gt_path"].values
        
        self.coords = {}

    def __len__(self):
        return len(self.paths)
    
    
    def prepare_gt(self, gt_path, shape):
        if not len(gt_path):
            return None

        coords = open(gt_path, 'r').readlines()
        coords = np.array([c[2:-1].split(' ') for c in coords]).astype(float)
        
        hm, regr = make_hm_regr(coords, shape)
        return np.concatenate([hm[None], regr]).transpose(1, 2, 0)


    def __getitem__(self, idx):
        """
        Item accessor

        Args:
            idx (int): Index.

        Returns:
            np array [H x W x C]: Image.
            torch tensor [1]: Label.
            torch tensor [1]: Sample weight.
        """
        image = cv2.imread(self.paths[idx]).astype(np.float32) / 255.
        
        gt_path = self.gt_paths[idx]
        target = self.prepare_gt(gt_path, image.shape)
        
        if target is None:
            target = np.zeros((image.shape[0], image.shape[1], 3))

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=target)
            image = transformed["image"]
            target = transformed["mask"]

        return image, target, 0
    
    
class ClsDataset(Dataset):
    """
    Image torch Dataset.
    """

    def __init__(
        self,
        df,
        transforms=None,
    ):
        """
        Constructor

        Args:
            paths (list): Path to images.
            transforms (albumentation transforms, optional): Transforms to apply. Defaults to None.
        """
        self.df = df
        self.paths = df["path"].values
        self.transforms = transforms
        self.targets = df["target"].values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Item accessor

        Args:
            idx (int): Index.

        Returns:
            np array [H x W x C]: Image.
            torch tensor [1]: Label.
            torch tensor [1]: Sample weight.
        """
        image = cv2.imread(self.paths[idx]).astype(np.float32) / 255.

        if self.transforms:
            image = self.transforms(image=image)["image"]

        if len(image.size()) == 2:
            image = torch.stack([image, image, image], 0)
        elif image.size(0) == 1:
            image = torch.cat([image, image, image], 0)

        y = torch.tensor([self.targets[idx]], dtype=torch.float)
        y_aux = y

        return image.float(), y, y_aux