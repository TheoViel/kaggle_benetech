import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss with label smoothing.

    Attributes:
        eps (float): Smoothing value.

    Methods:
        __init__(self, eps=0.0, device="cuda"): Constructor.
        forward(self, inputs, targets): Computes the loss.

    """
    def __init__(self, eps=0.0, device="cuda"):
        """
        Constructor.

        Args:
            eps (float, optional): Smoothing value. Defaults to 0.
            device (str, optional): Device to use for computations. Defaults to "cuda".
        """
        super(SmoothCrossEntropyLoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        """
        Computes the loss.

        Args:
            inputs (torch.Tensor): Predictions of shape [bs x n].
            targets (torch.Tensor): Targets of shape [bs x n] or [bs].

        Returns:
            torch.Tensor: Loss values, averaged.
        """
        if len(targets.size()) == 1:  # to one hot
            targets = torch.zeros_like(inputs).scatter(1, targets.view(-1, 1).long(), 1)

        if self.eps > 0:
            n_class = inputs.size(1)
            targets = targets * (1 - self.eps) + (1 - targets) * self.eps / (
                n_class - 1
            )

        loss = -targets * F.log_softmax(inputs, dim=1)
        loss = loss.sum(-1)
        return loss


class ClsLoss(nn.Module):
    """
    Loss wrapper for the problem.

    Attributes:
        config (dict): Configuration parameters.
        device (str): Device to use for computations.
        aux_loss_weight (float): Weight for the auxiliary loss.
        ousm_k (int): Number of samples to exclude in the OUSM variant. Defaults to 0.
        eps (float): Smoothing value. Defaults to 0.
        loss (nn.Module): Loss function.
        loss_aux (nn.Module): Auxiliary loss function.

    Methods:
        __init__(self, config, device="cuda"): Constructor.
        prepare(self, pred, y): Prepares the predictions and targets for loss computation.
        forward(self, pred, pred_aux, y, y_aux): Computes the loss.
    """
    def __init__(self, config, device="cuda"):
        """
        Constructor.

        Args:
            config (dict): Configuration parameters.
            device (str, optional): Device to use for computations. Defaults to "cuda".
        """
        super().__init__()
        self.config = config
        self.device = device

        self.aux_loss_weight = config["aux_loss_weight"]
        self.ousm_k = config.get("ousm_k", 0)
        self.eps = config.get("smoothing", 0)

        if config["name"] == "bce":
            self.loss = nn.BCEWithLogitsLoss(reduction="none")
        elif config["name"] == "ce":
            self.loss = SmoothCrossEntropyLoss(
                eps=self.eps, device=device
            )
        else:
            raise NotImplementedError

        self.loss_aux = nn.MSELoss(reduction="none")

    def prepare(self, pred, y):
        """
        Prepares the predictions and targets for loss computation.

        Args:
            pred (torch.Tensor): Predictions.
            y (torch.Tensor): Targets.

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Prepared predictions and targets.
        """
        if self.config["name"] in ["ce", "supcon"]:
            y = y.squeeze()
        else:
            y = y.float()
            pred = pred.float().view(y.size())

        if self.eps and self.config["name"] == "bce":
            y = torch.clamp(y, self.eps, 1 - self.eps)

        return pred, y

    def forward(self, pred, pred_aux, y, y_aux):
        """
        Computes the loss.

        Args:
            pred (torch.Tensor): Main predictions.
            pred_aux (list): Auxiliary predictions.
            y (torch.Tensor): Main targets.
            y_aux (list): Auxiliary targets.

        Returns:
            torch.Tensor: Loss value.
        """
        pred, y = self.prepare(pred, y)

        loss = self.loss(pred, y)

        if self.ousm_k:
            _, idxs = loss.topk(y.size(0) - self.ousm_k, largest=False)
            loss = loss.index_select(0, idxs)

        loss = loss.mean()

        if not self.aux_loss_weight > 0:
            return loss

        w_aux_tot = 0
        loss_aux_tot = 0
        if isinstance(pred_aux, list):
            for layer, p in enumerate(pred_aux):
                loss_aux = self.loss(p, y)

                if self.ousm_k:
                    loss_aux = loss_aux.index_select(0, idxs)

                loss_aux = loss_aux.mean()

                loss_aux_tot += (self.aux_loss_weight * (layer + 1)) * loss_aux
                w_aux_tot += (self.aux_loss_weight * (layer + 1))

        return (1 - w_aux_tot) * loss + w_aux_tot * loss_aux_tot
