import gc
import sys
import cv2
import torch
import numpy as np
import torch.nn as nn
import albumentations as albu

from tqdm import tqdm
from albumentations import pytorch as AT
from torch.utils.data import Dataset, DataLoader

from util.boxes import Boxes
from inference.utils import DetectionMeter, collate_fn_val_yolo




def predict(model, dataset, config, disable_tqdm=True, extract_fts=False):
    """
    Predict function.

    Args:
        model (torch model): Model wrapper.
        dataset (InferenceDataset): Dataset.
        config (Config): Config.

    Returns:
        DetectionMeter: Meter containing predictions.
        TODO
    """
    model.eval()
    
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

    fts_list, fts = [], []
    with torch.no_grad():
        for batch in tqdm(loader, disable=disable_tqdm):
            x = batch[0].to(config.device)

            try:
                pred_boxes, fts = model(x)
            except:
                pred_boxes = model(x)
#                 print(pred_boxes)
#             print(pred_boxes)
                
            meter.update(batch[1], pred_boxes, x.size())

            if extract_fts:
                if len(fts) > 1: # several fts
                    if not len(fts_list):
                        fts_list = [[ft] for ft in fts]
                    else:
                        for i in range(len(fts)):
                            fts_list[i].append(fts[i])
                else:
                    fts_list += fts
    
    if len(fts) > 1 and extract_fts:
        fts_list = [torch.cat(fts) for fts in fts_list]

    gc.collect()
    torch.cuda.empty_cache()
    return meter, fts_list


def custom_cat(fts):
    assert all([len(ft.size()) == 3 for ft in fts])

    dim = np.max([ft.size(-1) for ft in fts])
    n = np.sum([ft.size(1) for ft in fts])

    cat = torch.zeros((fts[0].size(0), n, dim))
    where = - torch.ones((fts[0].size(0), n, dim))

    current = 0
    for i, ft in enumerate(fts):
        cat[:, current: current + ft.size(1), :ft.size(2)] = ft
        where[:, current: current + ft.size(1), :ft.size(2)] = i
    return cat.cpu(), where.cpu()


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
        
        sys.path.append("../yolov7/")
        from utils.general import non_max_suppression
        self.non_max_suppression = non_max_suppression

    def forward(self, x):
        """
        Forward function.

        Args:
            x (torch tensor [BS x C x H x W]): Input images.

        Returns:
            torch tensor: Predictions.
        """
        try:
            pred_boxes, fts = self.model(x, return_fts=True)
        except:
            pred_boxes, _ = self.model(x)
            fts = []
            
        if isinstance(pred_boxes, torch.Tensor):
            if pred_boxes.size(1) == 5:  # YOLOv8
                pred_boxes = torch.cat([pred_boxes, torch.ones((pred_boxes.size(0), 1, pred_boxes.size(-1))).to(x.device)], 1)
                pred_boxes = pred_boxes.transpose(2, 1).contiguous()
        
#         pred_boxes[0] 
        
#         pred_boxes = [pred_boxes]
#         fts = fts[0]  # first layer

#         fts, where = custom_cat(fts)

#         try:
#             bs, n, _ = pred_boxes[0].size()
#         except:
#             return pred_boxes, []
#             pred_boxes = [pred_boxes]
#             bs, n, _ = pred_boxes[0].size()
    
#         print(pred_boxes)
#         print(len(fts), fts[0].size())
#         print(fts.size(), pred_boxes[0].size())
#         print(np.argsort(pred_boxes[0][0, :, 4].cpu().numpy())[::-1][:3])

        pred_boxes = self.non_max_suppression(
            pred_boxes,
            multi_label=False,
            conf_thres=self.conf_thresh,
            iou_thres=self.iou_thresh,
            max_det=self.max_per_img,
            min_det=self.min_per_img,
        )
        # Division by 3 accounts for ratios
#         pred_boxes, ids = [p[:, :-1].cpu() for p in pred_boxes], [p[:, -1].cpu().long() // 3 for p in pred_boxes]
        pred_boxes = [p[:, :-1].cpu() for p in pred_boxes]
#         filtered_fts = []
#         for ft, w, id_ in zip(fts, where, ids):
#             filtered_ft = ft[id_]
#             filtered_w = w[id_]
#             filtered_fts.append(
#                 [ff[: (fw >= 0).sum()].cpu().numpy() for ff, fw in zip(filtered_ft, filtered_w)]
#             )

        return pred_boxes, fts


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
    else:
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


def retrieve_model_robust(config, yolov7_path='../yolov7/'):
    """
    """
    sys.path.append(yolov7_path)
    from models.yolo import Model
    from utils.general import non_max_suppression

    model = Model(config.cfg)
    model.load_state_dict(torch.load(config.weights), strict=True)
    model = YoloWrapper(model, config).cuda()
    model.non_max_suppression = non_max_suppression

    return model.eval()
