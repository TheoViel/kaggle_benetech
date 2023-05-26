import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


def to_square(box):
    xc = (box[0] + box[2]) / 2
    yc = (box[1] + box[3]) / 2

    w = box[2] - box[0]
    h = box[3] - box[1]

    hw = min(h, w)
    hw = hw // 2 * 2 - 1
    hw = max(hw, 7)

    box = np.array(
        [
            xc - hw / 2,
            yc - hw / 2,
            xc - hw / 2,
            yc - hw / 2,
        ]
    )  # .astype(int)
    box = np.ceil(box).astype(int)
    box[2] += hw
    box[3] += hw

    return box


def point_nms(coords, scores, dist_th=8):
    if len(coords) == 0:
        return []

    dists = np.sqrt(((coords[None] - coords[:, None]) ** 2).sum(-1))

    #     print(np.round(dists, 1))

    # Sort by x sth ??
    sorted_indices = np.argsort(scores)[::-1]
    # np.arange(len(coords))  # sorted(range(len(coords)), key=lambda i: coords[i][0])
    #     print(sorted_indices)

    # Initialize list to store selected indices
    selected_indices = [sorted_indices[0]]
    for i in range(1, len(sorted_indices)):
        is_selected = True
        for idx in selected_indices:
            if dists[sorted_indices[i], idx] < dist_th:
                is_selected = False
                break

        if is_selected:
            #             print(selected_indices, sorted_indices[i])
            selected_indices.append(sorted_indices[i])

    return np.array(selected_indices)


def retrieve_missing_boxes(preds, img, min_sim=0.85, verbose=0):
    n_filters = 32

    pool_size = 5
    pool = torch.nn.MaxPool2d(pool_size, stride=1, padding=pool_size // 2).cuda()

    dots = preds[-1]
    x = torch.tensor(img / 255).cuda()
    x = x.transpose(1, 2).transpose(1, 0).float()

    candidates = []
    for i, box in enumerate(dots[:5]):
        box = to_square(box)
        crop = x[:, box[1]: box[3], box[0]: box[2]]

        pad = crop.size(1) // 2
        conv = nn.Conv2d(
            3, n_filters, kernel_size=crop.size(1), padding=pad, bias=False
        ).cuda()

        if verbose:
            plt.subplot(2, 5, i + 1)
            plt.imshow(crop.cpu().numpy().transpose(1, 2, 0))
            plt.axis()
        #         plt.show()

        with torch.no_grad():
            crop_embed = conv(crop)[:, pad, pad].unsqueeze(-1).unsqueeze(-1)
            #         print(crop_embed.size(), pad)
            img_embed = conv(x)

            assert img_embed.size(1) == x.size(1) and img_embed.size(2) == x.size(2)

            sim = ((img_embed - crop_embed) ** 2).sum(0).unsqueeze(0)
            sim = 1 / (1 + sim)
            #         print(sim.size())
            sim = torch.where(sim > min_sim, sim, 0)
            sim = torch.where(sim == pool(sim), sim, 0)

            #             if verbose:
            #                 plt.imshow(sim.cpu().numpy()[0])
            #                 plt.show()

            yc, xc = torch.where(sim[0] > 0)
            coords = torch.cat([xc.unsqueeze(-1), yc.unsqueeze(-1)], -1)

            if len(coords) - len(dots) < 20:
                #                 print('!!')
                candidates.append(coords.cpu().numpy())

    if not len(candidates):
        return []
    candidates = np.concatenate(candidates)

    points = np.concatenate(
        [
            (dots[:, 0][:, None] + dots[:, 2][:, None]) / 2,
            (dots[:, 1][:, None] + dots[:, 3][:, None]) / 2,
        ],
        -1,
    )

    scores = np.ones(len(points) + len(candidates))
    scores[len(points):] = 0

    if verbose:
        plt.show()
        print(points)
        print(candidates)

    kept_ids = point_nms(np.concatenate([points, candidates], 0), scores)
    kept_ids = kept_ids[len(points):] - len(points)

    new_boxes = candidates[kept_ids]
    hw = 5
    new_boxes = np.concatenate(
        [
            new_boxes[:, :1] - hw,
            new_boxes[:, 1:] - hw,
            new_boxes[:, :1] + hw,
            new_boxes[:, 1:] + hw,
        ],
        -1,
    )

    if verbose:
        print("NMS", kept_ids)
        print(new_boxes)

    # Points are inside the graph
    margin = 0
    try:
        graph = preds[0][0]
    except Exception:
        return new_boxes
    new_boxes = new_boxes[new_boxes[:, 0] > graph[0] - margin]
    new_boxes = new_boxes[new_boxes[:, 1] > graph[1] - margin]
    new_boxes = new_boxes[new_boxes[:, 2] < graph[2] + margin]
    new_boxes = new_boxes[new_boxes[:, 3] < graph[3] + margin]

    return new_boxes


def retrieve_missing_boxes_2(preds, img, feats, min_sim=0.8, verbose=0):
#     pool_size = 5
#     pool = torch.nn.MaxPool2d(pool_size, stride=1, padding=pool_size // 2).cuda()

    dots = preds[-1]
    x = torch.tensor(img / 255).cuda()
    x = x.transpose(1, 2).transpose(1, 0).float()
    
    feats = feats / ((feats ** 2).sum(0, keepdims=True) + 1e-6).sqrt()

    candidates, scores_candids = [], []
    for i, box in enumerate(dots[:5]):
#         if verbose:
#             plt.subplot(2, 5, i + 1)
#             plt.imshow(crop.cpu().numpy().transpose(1, 2, 0))
#             plt.axis()
        #         plt.show()

        with torch.no_grad():
            y = (box[0] + box[2]) / 2
            y = int(y / img.shape[1] * feats.size(2))
            x = (box[1] + box[3]) / 2
            x = int(x / img.shape[0] * feats.size(1))

        #     print(x, y)

        #     plt.imshow(img[box[1]: box[3], box[0]: box[2]])
        #     plt.show()

            vec = feats[:, x, y][:, None, None]

        #     sim = ((feats - vec) ** 2).mean(0, keepdims=True)
        #     sim = 1 / (sim + 1)

            sim = (feats * vec).sum(0, keepdims=True)
            
            sim = torch.where(sim > min_sim, sim, 0)
#             sim = torch.where(sim == pool(sim), sim, 0)
            
            if verbose:
                plt.imshow(sim[0].cpu().numpy())
                plt.colorbar()
                plt.show()

            sim = sim[0]
            yc, xc = torch.where(sim > 0)
            scores_c = [sim[y, x].item() for y, x in zip(yc, xc)]
            scores_c = np.array(scores_c)

#             print(scores)
#             print(xc, yc, img.shape)

            yc = yc / feats.size(1) * img.shape[0]
            xc = xc / feats.size(2) * img.shape[1]
            
            coords = torch.cat([xc.unsqueeze(-1), yc.unsqueeze(-1)], -1)

            if len(coords) - len(dots) < 20:
                candidates.append(coords.cpu().numpy())
                scores_candids.append(scores_c)

    if not len(candidates):
        return []

    scores_candids = np.concatenate(scores_candids)
    candidates = np.concatenate(candidates)
    points = np.concatenate(
        [
            (dots[:, 0][:, None] + dots[:, 2][:, None]) / 2,
            (dots[:, 1][:, None] + dots[:, 3][:, None]) / 2,
        ],
        -1,
    )

    scores = np.ones(len(points) + len(candidates))
    scores[len(points):] = scores_candids * 0.9

    all_points = np.concatenate([points, candidates], 0)

    if verbose:
        plt.show()
        print(points)
        print(candidates)

    kept_ids = point_nms(all_points, scores, dist_th=8)
    kept_ids = kept_ids[len(points):] - len(points)

    new_boxes = candidates[kept_ids]
    hw = 5
    shift = 2
    new_boxes = np.concatenate(
        [
            new_boxes[:, :1] - hw + shift,
            new_boxes[:, 1:] - hw + shift,
            new_boxes[:, :1] + hw + shift,
            new_boxes[:, 1:] + hw + shift,
        ],
        -1,
    )

    if verbose:
        print("NMS", kept_ids)
        print(new_boxes)

    # Points are inside the graph
    margin = 0
    try:
        graph = preds[0][0]
    except Exception:
        return new_boxes
    new_boxes = new_boxes[new_boxes[:, 0] > graph[0] - margin]
    new_boxes = new_boxes[new_boxes[:, 1] > graph[1] - margin]
    new_boxes = new_boxes[new_boxes[:, 2] < graph[2] + margin]
    new_boxes = new_boxes[new_boxes[:, 3] < graph[3] + margin]

    return new_boxes
