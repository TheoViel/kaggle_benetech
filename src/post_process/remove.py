import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from util.torch import seed_everything


def to_squares(boxes):
    w = (boxes[:, 2] - boxes[:, 0]).mean()
    h = (boxes[:, 3] - boxes[:, 1]).mean()
    
    xc = (boxes[:, 0] + boxes[:, 2]) / 2
    yc = (boxes[:, 1] + boxes[:, 3]) / 2

#     hw = min(h, w)
#     hw = hw // 2 * 2 - 1
#     hw = max(hw, 3)
    hw = 3

    boxes = np.array(
        [
            xc - hw / 2,
            yc - hw / 2,
            xc - hw / 2,
            yc - hw / 2,
        ]
    )  # .astype(int)
    boxes = np.ceil(boxes).astype(int)
    boxes[2] += int(hw)
    boxes[3] += int(hw)
    
    boxes = boxes.T

    return boxes, int(hw)


def remove_outlier_boxes(preds, img, verbose=0, coef=2):
    n_filters = 32

    pool_size = 5
    pool = torch.nn.MaxPool2d(pool_size, stride=1, padding=pool_size // 2).cuda()

    dots = preds[-1]
    x = torch.tensor(img / 255).cuda()
    x = x.transpose(1, 2).transpose(1, 0).float()
    
    dots, hw = to_squares(dots)

    seed_everything(0)
    conv = nn.Conv2d(
        3, n_filters, kernel_size=3, padding=0, bias=False
    ).cuda()

    fts = []
    with torch.no_grad():
        for i, box in enumerate(dots):
            crop = x[:, box[1]: box[3], box[0]: box[2]]
            ft = conv(crop).mean(-1).mean(-1)
            if verbose:
                print(crop.size(), ft.size())
            fts.append(ft)

            if verbose:
                plt.subplot(2, len(dots), i + 1)
                plt.imshow(crop.cpu().numpy().transpose(1, 2, 0))
                plt.axis(False)
     
    
    fts = torch.stack(fts)
    dists = ((fts.unsqueeze(1) - fts.unsqueeze(0)) ** 2).mean(-1).sqrt()
    
    if verbose:
        plt.matshow(dists.cpu().numpy())
        plt.colorbar()
        plt.show()

    dists = dists.mean(0).cpu().numpy()

    ref = np.percentile(dists, 75)
#     print(ref, np.max(dists))
    outliers = np.argwhere(dists > ref * coef)

    return outliers
