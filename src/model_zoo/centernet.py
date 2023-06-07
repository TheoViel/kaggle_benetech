import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class CenterNet(nn.Module):
    def __init__(self, model_name="resnet18", n_channels=3, pretrained=True, drop_path_rate=0, num_classes=1, pretrained_weights=None, verbose=1):
        super().__init__()

        self.model = smp.Unet(
            encoder_name="tu-" + model_name,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=n_channels
        )
        
        self.model .segmentation_head = nn.Identity()
        self.model .decoder.blocks = self.model.decoder.blocks[:-2]
        
        n_fts = self.model.decoder.blocks[-1].conv2[0].out_channels
        self.cls = nn.Conv2d(n_fts, num_classes, kernel_size=1)

    def forward(self, x):
        y = self.model(x)
        return self.cls(y)
