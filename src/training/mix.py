import torch
import numpy as np
import torch.nn as nn


class Mixup(nn.Module):
    def __init__(self, alpha, additive=False, num_classes=1):
        super(Mixup, self).__init__()
        self.beta_distribution = torch.distributions.Beta(alpha, alpha)
        self.additive = additive
        self.num_classes = num_classes

    def forward(self, x, y, y_aux=None):
        bs = x.shape[0]
        n_dims = len(x.shape)
        perm = torch.randperm(bs)
        coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(x.device)
        
        if self.num_classes > y.size(-1):  # One-hot
            y = torch.zeros(
                y.size(0), self.num_classes
            ).to(y.device).scatter(1, y.view(-1, 1).long(), 1)

        if n_dims == 2:
            x = coeffs.view(-1, 1) * x + (1 - coeffs.view(-1, 1)) * x[perm]
        elif n_dims == 3:
            x = coeffs.view(-1, 1, 1) * x + (1 - coeffs.view(-1, 1, 1)) * x[perm]
        elif n_dims == 4:
            x = coeffs.view(-1, 1, 1, 1) * x + (1 - coeffs.view(-1, 1, 1, 1)) * x[perm]
        else:
            x = (
                coeffs.view(-1, 1, 1, 1, 1) * x
                + (1 - coeffs.view(-1, 1, 1, 1, 1)) * x[perm]
            )

        if self.additive:
            y = (y + y[perm]).clip(0, 1)
            if y_aux is not None:
                y_aux = (y_aux + y_aux[perm]).clip(0, 1)
        else:
            if len(y.shape) == 1:
                y = coeffs * y + (1 - coeffs) * y[perm]
                if y_aux is not None:
                    y_aux = coeffs * y_aux + (1 - coeffs) * y_aux[perm]
            else:
                y = coeffs.view(-1, 1) * y + (1 - coeffs.view(-1, 1)) * y[perm]
                if y_aux is not None:
                    y_aux = (
                        coeffs.view(-1, 1) * y_aux + (1 - coeffs.view(-1, 1)) * y_aux[perm]
                    )

        return x, y, y_aux

    
class Cutmix(nn.Module):
    def __init__(self, alpha, additive=False, num_classes=1):
        super(Cutmix, self).__init__()
        self.beta_distribution = torch.distributions.Beta(alpha, alpha)
        self.additive = additive
        self.num_classes = num_classes

    @staticmethod
    def rand_bbox(size, lam):
        """
        Retuns the coordinate of a random rectangle in the image for cutmix.

        Args:
            size (torch tensor [batch_size x c x w x h): Input size.
            lam (int): Lambda sampled by the beta distribution. Controls the size of the squares.

        Returns:
            int: 4 coordinates of the rectangle.
            int: Proportion of the unmasked image.
        """
        w = size[2]
        h = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(w * cut_rat)
        cut_h = np.int(h * cut_rat)

        # uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        return bbx1, bby1, bbx2, bby2, lam

    def forward(self, x, y, y_aux=None):
        n_dims = len(x.shape)
        perm = torch.randperm(x.shape[0])
        coeff = self.beta_distribution.rsample(torch.Size((1,))).to(x.device).view(-1).item()
        
        bbx1, bby1, bbx2, bby2, coeff = self.rand_bbox(x.size(), coeff)
        
        if self.num_classes > y.size(-1):  # One-hot
            y = torch.zeros(
                y.size(0), self.num_classes
            ).to(y.device).scatter(1, y.view(-1, 1).long(), 1)

        if n_dims == 3:
            x[:, bbx1:bbx2, bby1:bby2] = x[perm, bbx1:bbx2, bby1:bby2]
        elif n_dims == 4:
            x[:, :, bbx1:bbx2, bby1:bby2] = x[perm, :, bbx1:bbx2, bby1:bby2]
        else:
            raise NotImplementedError

        if self.additive:
            y = (y + y[perm]).clip(0, 1)
            if y_aux is not None:
                y_aux = (y_aux + y_aux[perm]).clip(0, 1)
        else:
            y = coeff * y + (1 - coeff) * y[perm]
            if y_aux is not None:
                y_aux = coeffs * y_aux + (1 - coeffs) * y_aux[perm]

        return x, y, y_aux
