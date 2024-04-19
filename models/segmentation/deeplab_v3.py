import torch
import segmentation_models_pytorch as smp

from models.segmentation._abstract import SegmentationModel


class _DeeplabV3(SegmentationModel):

    encoder_name = None

    def __init__(self, pretrained="coco_voc", **kwargs):
        assert pretrained in ["coco_voc", "imagenet", None]
        self.pretrained = pretrained
        self.lr = kwargs.get("lr", 0.0001)
        self.weight_decay = kwargs.get("weight_decay", 0.0001)
        self.loss_mode = kwargs.get("loss_mode", "multiclass")

        super().__init__(**kwargs)

    def initialize_model(self):
        if self.pretrained == "coco_voc":
            model = torch.hub.load('pytorch/vision:v0.10.0', f'deeplabv3_{self.encoder_name}', pretrained=True)
        elif self.pretrained == "imagenet":
            model = smp.DeepLabV3(
                encoder_name="resnet50",
                encoder_weights="imagenet" if self.pretrained else None,
                in_channels=3,
                classes=self.n_classes
            )
        else:
            model = torch.hub.load('pytorch/vision:v0.10.0', f'deeplabv3_{self.encoder_name}', pretrained=False)

        return model

    def initialize_loss(self):
        loss = smp.losses.FocalLoss(mode=self.loss_mode, ignore_index=self.ignore_index)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 25, 50], gamma=0.5)
        # poly learning rate policy
        # max_iter = self.max_epochs * len(self.train_dataloader())
        # lambda1 = lambda step: pow((1 - step / max_iter), 0.9)
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

        # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: (1 - x / (self.cfg["dataset_len"] * self.cfg["max_epochs"])) ** 0.9)

        return [optimizer], [scheduler]


class DeeplabV3Resnet50(_DeeplabV3):
    encoder_name = "resnet50"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class DeeplabV3Resnet101(_DeeplabV3):
    encoder_name = "resnet101"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


if __name__ == '__main__':

    model = DeeplabV3Resnet101()
