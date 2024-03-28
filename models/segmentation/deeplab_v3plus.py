import torch
import segmentation_models_pytorch as smp
from models.segmentation import custom_focal_loss
from models.segmentation.deeplabv3plus_net.deeplabv3plus import deeplabv3plus as dlv3p
from models.segmentation.deeplabv3plus_net.model_configs.model_base_config import cfg
# TODO make this general so we can load different configs if needed

from models.segmentation._abstract import SegmentationModel


class DeeplabV3Plus(SegmentationModel):

    encoder_name = None

    def __init__(self, max_epochs, dataloader_length, optim_params, **kwargs):
        self.max_epochs = max_epochs
        self.dataloader_length = dataloader_length
        self.optim_params = optim_params
        self.freeze_backbone = kwargs.get("freeze_backbone")
        self.freeze_batchnorm = kwargs.get("freeze_batchnorm")
        # self.loss_mode = kwargs.get("loss_mode", "multiclass")

        super().__init__(**kwargs, normalize_images=False)

        if self.freeze_backbone:
            for param in self.model.backbone.parameters():
                param.requires_grad = False
        if self.freeze_batchnorm:
            for param_name, param in self.model.named_parameters():
                if "bn" in param_name or "BatchNorm" in param_name:
                    print(param_name)
                    param.requires_grad = False
        print('d')

    def initialize_model(self):
        model = dlv3p(cfg)
        return model

    def initialize_loss(self):
        # loss = smp.losses.FocalLoss(mode=self.loss_mode, ignore_index=self.ignore_index)
        loss = torch.nn.CrossEntropyLoss(ignore_index=255)
        return loss

    def configure_optimizers(self):
        train_lr = self.optim_params['lr']
        train_power = self.optim_params['power']
        train_momentum = self.optim_params['momentum']
        optimizer = torch.optim.SGD(
            params=[
                {'params': self.get_params(self.model, key='1x'), 'lr': train_lr},
                {'params': self.get_params(self.model, key='10x'), 'lr': 10 * train_lr}
            ],
            momentum=train_momentum
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[
            lambda x: (1 - x / (self.max_epochs * self.dataloader_length + 1)) ** train_power,
            lambda x: (1 - x / (self.max_epochs * self.dataloader_length + 1)) ** train_power
        ])
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def get_params(self, model, key):
        for m in model.named_modules():
            if key == '1x':
                if 'backbone' in m[0] and isinstance(m[1], torch.nn.Conv2d):
                    for p in m[1].parameters():
                        yield p
            elif key == '10x':
                if 'backbone' not in m[0] and isinstance(m[1], torch.nn.Conv2d):
                    for p in m[1].parameters():
                        yield p


if __name__ == '__main__':

    model = DeeplabV3Plus(10, 1000, class_names=[0, 1, 2, 3])
