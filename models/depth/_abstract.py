from abc import ABC, abstractmethod
from typing import List, Dict
import math

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import lightning.pytorch as pl
import segmentation_models_pytorch as smp
from models.depth.utils_depth import metrics
from argparse import Namespace
from misc.visualizer import make_axes_invisible, SegmentationMapVisualizer


class DepthModel(pl.LightningModule, ABC):

    def __init__(self, max_depth=10,
                 normalize_images: bool = True,
                 plot_preds_every_n_steps: int = 100,
                 visualizer_kwargs: Dict = None
                 ):
        super().__init__()

        self.max_depth = max_depth
        self.normalize_images = normalize_images
        self.plot_preds_every_n_steps = plot_preds_every_n_steps

        self.model = self.initialize_model()
        self.loss = self.initialize_loss()

        if visualizer_kwargs is None:
            visualizer_kwargs = {}
        self.visualizer = SegmentationMapVisualizer(**visualizer_kwargs)

    @abstractmethod
    def initialize_model(self):
        raise NotImplementedError

    @abstractmethod
    def initialize_loss(self):
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self):
        return None  # override this

    def forward(self, x):
        if self.normalize_images:
            x = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
        return self.model(x)

    def _step(self, images: torch.Tensor, masks: torch.LongTensor, phase: str):
        assert phase in ["train", "val"]

        pred_mask = self(images)
        loss = self.loss(pred_mask, masks)

        if self.global_step % self.plot_preds_every_n_steps == 0:
            self._plot_predictions(images.detach(), masks.detach(), pred_mask.detach(), phase)

        metrics_dict = self.compute_metrics(pred_mask, masks, phase)
        metrics_dict[f"{phase}_loss"] = loss.item()
        self.log_dict(metrics_dict, prog_bar=True, logger=True, on_step=True, batch_size=images.shape[0])
        # self.log(f"{phase}_loss", loss, prog_bar=True, logger=True, on_step=True)
        return loss

    def compute_metrics(self, pred_mask: torch.Tensor, masks: torch.Tensor, phase: str):
        # TODO switch over to these official metrics https://github.com/jspenmar/monodepth_benchmark/blob/main/src/core/metrics.py

        crop_args = Namespace(dataset='nyudepthv2', min_depth_eval=1e-4, max_depth_eval=10.0)
        valid_pred_mask, valid_gt_mask = metrics.cropping_img(crop_args, pred_mask, masks)
        metric_dict = metrics.eval_depth(valid_pred_mask, valid_gt_mask)
        del metric_dict['silog']
        # rmse = torch.sqrt(torch.mean((pred_mask - masks) ** 2))
        # rel = torch.mean(torch.abs(pred_mask - masks) / masks)
        # log10 = torch.mean(torch.abs(torch.log10(pred_mask) - torch.log10(masks)))
        #
        # ratio_mask1 = masks / pred_mask
        # ratio_mask2 = pred_mask / masks
        # max_ratio_mask = torch.max(torch.stack(ratio_mask1, ratio_mask2), dim=0)
        #
        # delta_1 = torch.mean(torch.le(max_ratio_mask, 1.25).float())
        # delta_2 = torch.mean(torch.le(max_ratio_mask, 1.25 ** 2).float())
        # delta_3 = torch.mean(torch.le(max_ratio_mask, 1.25 ** 3).float())
        #
        # metrics = dict(rmse=rmse, rel=rel, log10=log10, delta_1=delta_1, delta_2=delta_2, delta_3=delta_3)
        return metric_dict

    def training_step(self, batch, batch_idx):
        images = batch['image']
        masks = batch['depth']
        loss = self._step(images, masks, "train")
        return loss

    def validation_step(self, batch, batch_idx):
        images = batch['image']
        masks = batch['depth']
        self._step(images, masks, "val")

    # def on_validation_epoch_end(self) -> None:
    #     self._on_epoch_end("val")
    #
    # def on_train_epoch_end(self) -> None:
    #     self._on_epoch_end("train")

    def _plot_predictions(self, images: torch.Tensor, masks: torch.Tensor, pred_masks: torch.Tensor, phase: str):

        imgs_per_subplot = min(4, images.shape[0])

        fig, axes = plt.subplots(imgs_per_subplot, 4, figsize=(24, 24), squeeze=False)

        for i in range(imgs_per_subplot):
            img = images[i].cpu()
            mask = masks[i].cpu()
            pred = pred_masks[i].cpu()

            diff = torch.abs(mask - pred)
            axes[i, 0].imshow(img.permute(1, 2, 0), cmap='inferno')
            axes[i, 1].imshow(mask.permute(1, 2, 0), cmap='inferno')
            axes[i, 2].imshow(pred.permute(1, 2, 0), cmap='inferno')
            axes[i, 3].imshow(diff.permute(1, 2, 0), cmap='inferno')

        plt.tight_layout()
        [make_axes_invisible(ax) for ax in axes.flatten()]
        try:
            self.logger.experiment.log({"predictions": fig})
        except:
            print("Logger cannot save image. Displaying instead.")
            plt.show()
        plt.close()
