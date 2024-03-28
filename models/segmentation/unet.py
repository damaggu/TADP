import torch
import segmentation_models_pytorch as smp

from models.segmentation._abstract import SegmentationModel


class UNetResnet18(SegmentationModel):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def initialize_model(self):
        model = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=3,
            classes=self.n_classes,
        )
        return model

    def initialize_loss(self):
        loss = smp.losses.FocalLoss(mode="multiclass", ignore_index=self.ignore_index)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        return optimizer