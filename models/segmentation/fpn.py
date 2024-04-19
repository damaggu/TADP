import torch
from torch import nn
import torchvision.transforms as T

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.fpn.decoder import FPNDecoder
from segmentation_models_pytorch.base import SegmentationHead

from models.segmentation._abstract import SegmentationModel


class FPN(SegmentationModel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initialize_model(self):
        model = nn.Module()
        model.encoder = smp.encoders.get_encoder(
            name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
        )
        model.decoder = FPNDecoder(
            encoder_channels=model.encoder.out_channels,
            encoder_depth=5,
            pyramid_channels=256,
            segmentation_channels=128,
            dropout=0.2,
            merge_policy="add",
        )
        model.segmentation_head = SegmentationHead(
            in_channels=model.decoder.out_channels,
            out_channels=self.n_classes,
            activation=None,
            kernel_size=1,
            upsampling=4,
        )

        return model

    def forward(self, x):
        if self.normalize_images:
            x = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        x = self.model.encoder(x)
        x = self.model.decoder(*x)
        x = self.model.segmentation_head(x)
        return x

    def initialize_loss(self):
        loss = smp.losses.FocalLoss(mode="multiclass", ignore_index=self.ignore_index)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        return optimizer


if __name__ == "__main__":
    model = FPN(class_names=["a", "b", "c"])
    print(model.model)
    x = torch.rand(1, 3, 256, 256)
    y = model(x)
