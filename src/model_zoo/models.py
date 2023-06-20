import timm
import torch
import torch.nn as nn

from model_zoo.gem import GeM
from util.torch import load_model_weights


def define_model(
    name,
    num_classes=1,
    num_classes_aux=0,
    n_channels=3,
    pretrained_weights="",
    pretrained=True,
    reduce_stride=False,
    drop_rate=0,
    drop_path_rate=0,
    use_gem=False,
    verbose=1,
):
    """
    Define a classification model.

    Args:
        name (str): Name of the model architecture.
        num_classes (int): Number of main output classes. Defaults to 1.
        num_classes_aux (int): Number of auxiliary output classes. Defaults to 0.
        n_channels (int): Number of input channels. Defaults to 3.
        pretrained_weights (str): Path to the pretrained weights. Defaults to "".
        pretrained (bool): Whether to use pretrained weights. Defaults to True.
        reduce_stride (bool): Whether to reduce the stride of the model. Defaults to False.
        drop_rate (float): Dropout rate. Defaults to 0.
        drop_path_rate (float): Drop path rate for stochastic depth. Defaults to 0.
        use_gem (bool): Whether to use GeM pooling. Defaults to False.
        verbose (int): Verbosity level. Defaults to 1.

    Returns:
        ClsModel: Defined classification model.
    """
    # Load pretrained model
    if drop_path_rate > 0:
        encoder = timm.create_model(
            name,
            pretrained=pretrained,
            drop_path_rate=drop_path_rate,
            num_classes=0,
            global_pool='',
        )
    else:
        encoder = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='',
        )
    encoder.name = name

    model = ClsModel(
        encoder,
        num_classes=num_classes,
        num_classes_aux=num_classes_aux,
        n_channels=n_channels,
        drop_rate=drop_rate,
        use_gem=use_gem,
    )

    if pretrained_weights:
        model = load_model_weights(
            model, pretrained_weights, verbose=verbose, strict=False
        )

    if reduce_stride:
        model.reduce_stride()

    return model


class ClsModel(nn.Module):
    """
    Classification model based on a backbone encoder.

    Methods:
        __init__(encoder, num_classes, num_classes_aux, n_channels, drop_rate, use_gem): Constructor
        _update_num_channels(): Update the number of input channels for the encoder
        reduce_stride(): Reduce the stride of the stem convolutional layer
        extract_features(x): Extract features from the input batch
        get_logits(fts): Compute logits from the extracted features
        forward(x, return_fts): Forward pass of the model

    Attributes:
        encoder (nn.Module): Backbone encoder model
        nb_ft (int): Number of features in the encoder
        num_classes (int): Number of classes for the main classification task
        num_classes_aux (int): Number of classes for an auxiliary classification task
        n_channels (int): Number of input channels
        use_gem (bool): Whether to use GeM pooling
        global_pool (GeM): Global pooling layer
        dropout (nn.Dropout or nn.Identity): Dropout layer
        logits (nn.Linear): Linear layer for main classification task
        logits_aux (nn.Linear): Linear layer for auxiliary classification task
    """

    def __init__(
        self,
        encoder,
        num_classes=1,
        num_classes_aux=0,
        n_channels=3,
        drop_rate=0,
        use_gem=False,
    ):
        """
        Constructor.

        Args:
            encoder (nn.Module): Backbone encoder model.
            num_classes (int, optional): Number of classes for the main cls task. Defaults to 1.
            num_classes_aux (int, optional): Number of classes for an auxiliary cls task. Defaults to 0.
            n_channels (int, optional): Number of input channels. Defaults to 3.
            drop_rate (float, optional): Dropout rate. If 0, no dropout is applied. Defaults to 0.
            use_gem (bool, optional): Whether to use GeM pooling. Defaults to False.
        """
        super().__init__()

        self.encoder = encoder
        self.nb_ft = encoder.num_features

        self.num_classes = num_classes
        self.num_classes_aux = num_classes_aux
        self.n_channels = n_channels
        self.use_gem = use_gem

        self.global_pool = GeM(p_trainable=False)
        self.dropout = nn.Dropout(drop_rate) if drop_rate else nn.Identity()

        self.logits = nn.Linear(self.nb_ft, num_classes)
        if self.num_classes_aux:
            self.logits_aux = nn.Linear(self.nb_ft, num_classes_aux)

        self._update_num_channels()

    def _update_num_channels(self):
        """
        Update the number of input channels for the encoder if it differs from 3.
        """
        if self.n_channels != 3:
            for n, m in self.encoder.named_modules():
                if n:
                    # print("Replacing", n)
                    old_conv = getattr(self.encoder, n)
                    new_conv = nn.Conv2d(
                        self.n_channels,
                        old_conv.out_channels,
                        kernel_size=old_conv.kernel_size,
                        stride=old_conv.stride,
                        padding=old_conv.padding,
                        bias=old_conv.bias is not None,
                    )
                    setattr(self.encoder, n, new_conv)
                    break

    def reduce_stride(self):
        """
        Reduce the stride of the stem convolutional layer if the encoder is an EfficientNet or an NFNet.
        """
        if "efficient" in self.encoder.name:
            self.encoder.conv_stem.stride = (1, 1)
        elif "nfnet" in self.encoder.name:
            self.encoder.stem.conv1.stride = (1, 1)
        else:
            raise NotImplementedError

    def extract_features(self, x):
        """
        Compute logits from the extracted features.

        Args:
            fts (torch.Tensor): Features tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Main logits tensor of shape (batch_size, num_classes).
            torch.Tensor: Auxiliary logits tensor of shape (batch_size, num_classes_aux).
        """
        fts = self.encoder(x)

        if self.use_gem:
            fts = self.global_pool(fts)[:, :, 0, 0]
        else:
            while len(fts.size()) > 2:
                fts = fts.mean(-1)

        return fts

    def get_logits(self, fts):
        """
        Computes logits.

        Args:
            fts (torch tensor [batch_size x num_features]): Features.

        Returns:
            torch tensor [batch_size x num_classes]: logits.
            torch tensor [batch_size x num_classes_aux]: logits aux.
        """
        logits = self.logits(fts)

        if self.num_classes_aux:
            logits_aux = self.logits_aux(fts)
        else:
            logits_aux = torch.zeros((fts.size(0)))

        return logits, logits_aux

    def forward(self, x, return_fts=False):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input batch tensor of shape (batch_size, n_frames, h, w).
            return_fts (bool): Whether to return the extracted features.

        Returns:
            torch.Tensor: Main logits tensor of shape (batch_size, num_classes).
            torch.Tensor: Auxiliary logits tensor of shape (batch_size, num_classes_aux).
            torch.Tensor: Extracted features tensor of shape (batch_size, num_features) if return_fts is True.
        """
        fts = self.extract_features(x)

        fts = self.dropout(fts)

        logits, logits_aux = self.get_logits(fts)

        if return_fts:
            return logits, logits_aux, fts
        return logits, logits_aux
