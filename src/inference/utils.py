import gc
import sys
import cv2
import torch
import numpy as np
import torch.nn as nn
import albumentations as albu

from tqdm import tqdm
from albumentations import pytorch as AT
from torch.utils.data import Dataset

from util.boxes import Boxes


class InferenceDataset(Dataset):
    """
    Detection dataset for inference.

    Attributes:
        df (DataFrame): The DataFrame containing the dataset information.
        paths (numpy.ndarray): The paths to the images in the dataset.
        transforms: Augmentations to apply to the images.
        gts (list): Ground truth boxes for each image.
        classes (list): Ground truth classes  for each image.

    Methods:
        __init__(self, df, transforms=None): Constructor
        __len__(self): Returns the length of the dataset.
        __getitem__(self, idx): Returns the item at the specified index.
    """

    def __init__(self, df, transforms=None, pad=False):
        """
        Constructor

        Args:
            df (DataFrame): The DataFrame containing the dataset information.
            transforms (albu transforms, optional): Augmentations. Defaults to None.
        """
        self.df = df
        self.paths = df['path'].values
        self.transforms = transforms
        self.pad = pad
        
        self.gts, self.classes = [], []
        for i in range(len(df)):
            try:
                with open(df['gt_path'][i], 'r') as f:
                    bboxes = np.array([l[:-1].split() for l in f.readlines()]).astype(float)
                    labels, bboxes = bboxes[:, 0], bboxes[:, 1:]
                    self.gts.append(bboxes)
                    self.classes.append(labels)
            except Exception:
                self.gts.append([])
                self.classes.append([])

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns the item at the specified index.

        Args:
            idx (int): Index.

        Returns:
            tuple: A tuple containing the image, ground truth, and image shape.
        """
        image = cv2.imread(self.paths[idx])
        
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         print(image.shape)
        if self.pad:
            if image.shape[1] > image.shape[0] * 1.5:
                padding = 255 * np.ones((image.shape[0] // 2, image.shape[1], image.shape[2]), dtype=image.dtype)
                image = np.concatenate([image, padding], 0)
#         print(image.shape)

        shape = image.shape

        if self.transforms is not None:
            try:
                image = self.transforms(image=image, bboxes=[], class_labels=[])["image"]
            except ValueError:
                image = self.transforms(image=image)["image"]

        return image, self.gts[idx], shape


def get_transfos(size):
    """
    Returns a composition of image transformations for preprocessing.

    Args:
        size (tuple): The desired size of the transformed image (height, width).

    Returns:
        albumentations.Compose: The composition of image transformations.
    """
    normalizer = albu.Compose(
        [
            albu.Normalize(mean=0, std=1),
            AT.transforms.ToTensorV2(),
        ],
        p=1,
    )

    return albu.Compose(
        [
            albu.Resize(size[0], size[1]),
            normalizer,
        ],
        bbox_params=albu.BboxParams(format="yolo", label_fields=["class_labels"]),
    )


class DetectionMeter:
    """
    Detection meter for evaluating object detection performance.

    Methods:
        __init__(pred_format, truth_format): Constructor
        update(y_batch, preds, shape): Update ground truths and predictions
        reset(): Resets all values

    Attributes:
        truth_format (str): Format of ground truth bounding box coordinates
        pred_format (str): Format of predicted bounding box coordinates
        preds (list): List of predicted bounding boxes (Boxes instances)
        labels (list): List of labels corresponding to predicted bounding boxes
        confidences (list): List of confidence scores for predicted bounding boxes
        truths (list): List of ground truth bounding boxes (Boxes instances)
        metrics (dict): Dictionary storing evaluation metrics (tp, fp, fn, precision, recall, f1_score)
    """
    def __init__(self, pred_format="coco", truth_format="yolo"):
        """
        Constructor

        Args:
            pred_format (str, optional): Format of predicted bounding box coordinates. Defaults to "coco".
            truth_format (str, optional): Format of ground truth bounding box coordinates. Defaults to "yolo".
        """
        self.truth_format = truth_format
        self.pred_format = pred_format
        self.reset()

    def update(self, y_batch, preds, shape):
        """
        Update ground truths and predictions.

        Args:
            y_batch (list of np arrays): Truths.
            preds (list of torch tensors): Predictions.
            shape (list or tuple): Image shape.

        Raises:
            NotImplementedError: Mode not implemented.
        """
        n, c, h, w = shape  # TODO : verif h & w

        self.truths += [Boxes(box, (h, w), bbox_format=self.truth_format) for box in y_batch]

        for pred in preds:
            pred = pred.cpu().numpy()
            
            if pred.shape[1] >= 5:
                label = pred[:, 5].astype(int)
                self.labels.append(label)

            pred, confidences = pred[:, :4], pred[:, 4]

            self.preds.append(Boxes(pred, (h, w), bbox_format=self.pred_format))
            self.confidences.append(confidences)

    def reset(self):
        """
        Resets everything.
        """
        self.preds = []
        self.labels = []
        self.confidences = []
        self.truths = []

        self.metrics = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "precision": 0,
            "recall": 0,
            "f1_score": 0,
        }


def collate_fn_val_yolo(batch):
    """
    Validation batch collating function for yolo-v5.

    Args:
        batch (tuple): Input batch.

    Returns:
        torch tensor: Images.
        list: Boxes.
        list: Image shapes.
    """
    img, boxes, shapes = zip(*batch)
    return torch.stack(list(img), 0), boxes, shapes
