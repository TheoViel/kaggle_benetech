import sys
import cv2
import torch
import numpy as np
import torch.nn as nn
import albumentations as albu
from albumentations import pytorch as AT
from torch.utils.data import Dataset, DataLoader

# from tracking.meter import DetectionMeter
from util.boxes import Boxes


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


def predict(model, dataset, config):
    """
    Predict function.

    Args:
        model (torch model): Model wrapper.
        dataset (InferenceDataset): Dataset.
        config (Config): Config.

    Returns:
        DetectionMeter: Meter containing predictions.
    """
    loader = DataLoader(
        dataset,
        batch_size=config.val_bs,
        shuffle=False,
        collate_fn=collate_fn_val_yolo,
        num_workers=2,
        pin_memory=True,
    )

    meter = DetectionMeter(pred_format=config.pred_format, truth_format=config.bbox_format)

    meter.reset()
    model.eval()

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(config.device)
            
#             print(x.mean())
            pred_boxes = model(x)
            meter.update(batch[1], pred_boxes, x.size())

    return meter


class YoloWrapper(nn.Module):
    """
    Wrapper for Yolo models.

    Methods:
        __init__(model, config): Constructor
        forward(x): Forward function

    Attributes:
        model (torch model): Yolo-v5 model.
        config (Config): Config.
        conf_thresh (float): Confidence threshold.
        iou_thresh (float): IoU threshold.
        max_per_img (int): Maximum number of detections per image.
        min_per_img (int): Minimum number of detections per image.
    """

    def __init__(self, model, config):
        """
        Constructor

        Args:
            model (torch model): Yolo model.
            config (Config): Config.
        """
        super().__init__()
        self.model = model
        self.config = config

        self.conf_thresh = config.conf_thresh
        self.iou_thresh = config.iou_thresh
        self.max_per_img = config.max_per_img
        self.min_per_img = config.min_per_img

    def forward(self, x):
        """
        Forward function.

        Args:
            x (torch tensor [BS x C x H x W]): Input images.

        Returns:
            torch tensor: Predictions.
        """
        pred_boxes, _ = self.model(x)
#         print(pred_boxes.size())

        pred_boxes = self.non_max_suppression(
            pred_boxes,
            multi_label=False,
            conf_thres=self.conf_thresh,
            iou_thres=self.iou_thresh,
            max_det=self.max_per_img,
            min_det=self.min_per_img,
        )

#         for p in pred_boxes:
#             print(p.mean())

        return pred_boxes


def retrieve_model(config, model=None):
    """
    Retrieves the detection model based on the provided configuration.

    Args:
        config (config): The configuration object containing model information.

    Returns:
        model: The detection model.

    Raises:
        AssertionError: If the specified weights are not supported.
    """
    weights = config.weights

    if "yolov5" in weights:
        assert "../yolov7/" not in sys.path
        sys.path.append("../yolov5/")
        from models.common import DetectMultiBackend
        from utils.general import non_max_suppression
        
        model = DetectMultiBackend(
            weights=weights,
            device=torch.device('cuda')
        ).model
    elif "yolov7" in weights:
        assert "../yolov5/" not in sys.path
        sys.path.append("../yolov7/")
        from models.experimental import attempt_load
        from utils.general import non_max_suppression

        if model is None:
            model = attempt_load(weights).cuda()
        else:
            model = model.cuda()
        
    model = YoloWrapper(model, config)
    
    model.non_max_suppression = non_max_suppression

    return model.eval()


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

    def __init__(self, df, transforms=None):
        """
        Constructor

        Args:
            df (DataFrame): The DataFrame containing the dataset information.
            transforms (albu transforms, optional): Augmentations. Defaults to None.
        """
        self.df = df
        self.paths = df['path'].values
        self.transforms = transforms
        
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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