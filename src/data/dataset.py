import re
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


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