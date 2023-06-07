import torch
import numpy as np
from util.metrics import compute_metrics
from util.boxes import Boxes


def pred2box(hm, regr, thresh=0.5, input_size=512, model_scale=1):
    # make binding box from heatmaps
    # thresh: threshold for logits.

    # get center
    pred = hm > thresh
    pred_center = np.where(hm > thresh)
    # get regressions
    pred_r = regr[:, pred].T

    # wrap as boxes
    # [xmin, ymin, width, height]
    # size as original image.
    boxes = []
    scores = hm[pred]
    for i, b in enumerate(pred_r):
        arr = np.array(
            [
                pred_center[1][i] * model_scale,
                pred_center[0][i] * model_scale,
                int(b[0] * input_size),
                int(b[1] * input_size),
            ]
        )
        arr = np.clip(arr, 0, input_size)
        boxes.append(arr)
    return np.asarray(boxes).astype(float), scores


def process_and_score(preds, df_val, th=0.5, shape=(128, 128), pool_size=5):
    pool = torch.nn.MaxPool2d(pool_size, stride=1, padding=pool_size // 2)

    f1s = []
    for i in range(len(df_val)):
        gt_path = df_val["gt_path"][i]
        coords = open(gt_path, "r").readlines()
        coords = np.array([c[2:-1].split(" ") for c in coords]).astype(float)

        heatmap = preds[i][0]
        sz = heatmap.shape[-1]

        if isinstance(heatmap, np.ndarray):
            heatmap = torch.from_numpy(heatmap)
        heatmap = heatmap.float()
    
        # Placeholder
        reg = torch.ones_like(heatmap) * 0.005
        reg = torch.stack([reg, reg])

        heatmap = heatmap.unsqueeze(0).unsqueeze(0)
        heatmap = torch.where(heatmap == pool(heatmap), heatmap, 0)
        heatmap = heatmap[0, 0]

        
        boxes, confs = pred2box(heatmap, reg, th)
#         while not len(boxes):
#             th = max(th - 0.1, th / 10)
#             boxes, confs = pred2box(heatmap, reg, th)

        if len(boxes):
            boxes[:, 2] = coords[:, 2].max() * sz
            boxes[:, 3] = coords[:, 3].max() * sz

        pred_boxes = Boxes(boxes / sz, shape, bbox_format="yolo")
        gt_boxes = Boxes(coords, shape, bbox_format="yolo")

        metrics = compute_metrics([pred_boxes], [gt_boxes])
        f1s.append(metrics['f1_score'])

    return f1s
