from abc import ABC, abstractmethod
from typing import List, Dict
import math

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import lightning.pytorch as pl
import segmentation_models_pytorch as smp

from misc.visualizer import make_axes_invisible, SegmentationMapVisualizer


class SegmentationModel(pl.LightningModule, ABC):

    def __init__(self, class_names: List[str],
                 ignore_index: int = 255,
                 normalize_images: bool = True,
                 plot_preds_every_n_steps: int = 200,
                 visualizer_kwargs: Dict = None,
                 num_val_dataloaders: int = 1,
                 compute_sce: bool = False,
                 sce_ratio: float = 0.5,
                 *args, **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.ignore_index = ignore_index
        self.normalize_images = normalize_images
        self.plot_preds_every_n_steps = plot_preds_every_n_steps
        self.num_val_dataloaders = num_val_dataloaders

        self.model = self.initialize_model()
        self.loss = self.initialize_loss()
        self.compute_sce = compute_sce
        self.sce_ratio = sce_ratio

        self.visualize_flag = {'train': False}
        self.step_outputs = {
            "train": {
                "tp": [],
                "fp": [],
                "fn": [],
                "tn": []
            },
        }

        for i in range(num_val_dataloaders):
            self.step_outputs[f"val_{i}"] = {
                "tp": [],
                "fp": [],
                "fn": [],
                "tn": []
            }
            self.visualize_flag[f"val_{i}"] = False

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
        out = self.model(x)
        if isinstance(out, dict):
            out = out["out"]
        return out

    @torch.no_grad()
    def evaluate_batch(self, batch: torch.Tensor, masks: torch.LongTensor):
        """Used for image-wise evaluation"""
        logits_mask = self(batch)
        prob_mask = logits_mask.sigmoid()
        pred_mask = torch.argmax(prob_mask, dim=1)

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.detach(),
                                               masks.detach(),
                                               mode="multiclass",
                                               num_classes=self.n_classes,
                                               ignore_index=self.ignore_index)

        return tp, fp, fn, tn

    def _step(self, images: torch.Tensor, masks: torch.LongTensor, img_metas: dict, phase: str):
        assert phase in self.visualize_flag.keys()

        logits_mask = self(images, img_metas)

        # Deals with the case where prob_masks are concatenated along the channel dimension
        target_prob_masks = None
        if masks.dtype == torch.float32 and not ((masks > 0) & (masks < 1)).any():
            masks = masks.long()
        else:
            target_prob_masks = masks[:, 1:]
            masks = masks[:, 0].long()

        loss = self.loss(logits_mask, masks)
        if self.compute_sce and phase == 'train':
            assert target_prob_masks is not None, 'probability masks must be concatenated along the channel dimension'
            loss_prob = self.loss(logits_mask, target_prob_masks.sigmoid())
            loss = (1 - self.sce_ratio) + self.sce_ratio * loss_prob

        prob_mask = logits_mask.sigmoid()
        pred_mask = torch.argmax(prob_mask, dim=1)

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.detach(),
                                               masks.detach(),
                                               mode="multiclass",
                                               num_classes=self.n_classes,
                                               ignore_index=self.ignore_index)

        self.step_outputs[phase]["tp"].append(tp.sum(dim=0))
        self.step_outputs[phase]["fp"].append(fp.sum(dim=0))
        self.step_outputs[phase]["fn"].append(fn.sum(dim=0))
        self.step_outputs[phase]["tn"].append(tn.sum(dim=0))

        if self.global_step % self.plot_preds_every_n_steps == 0:
            for k in self.visualize_flag.keys():
                self.visualize_flag[k] = True

        # TODO make sure this works as intended -- goal is to plot predictions for val even if it doesnt land
        #   on the global step count exactly
        if self.visualize_flag[phase]:
            self._plot_predictions(images.detach(), masks.detach(), pred_mask.detach(), prob_mask.detach(), phase)
            self.visualize_flag[phase] = False

        self.log(f"{phase}_loss", loss, prog_bar=True, logger=True, on_step=True)
        return loss

    def _on_epoch_end(self, phase: str) -> None:

        tp = torch.stack(self.step_outputs[phase]["tp"]).sum(dim=0)
        fp = torch.stack(self.step_outputs[phase]["fp"]).sum(dim=0)
        fn = torch.stack(self.step_outputs[phase]["fn"]).sum(dim=0)
        tn = torch.stack(self.step_outputs[phase]["tn"]).sum(dim=0)

        for buffer in self.step_outputs[phase].values():
            buffer.clear()

        iou_per_class = smp.metrics.iou_score(tp, fp, fn, tn, reduction=None)
        f1_per_class = smp.metrics.f1_score(tp, fp, fn, tn, reduction=None)

        for cls_name, _miou, _f1 in zip(self.class_names, iou_per_class, f1_per_class):
            self.log(f"{phase}_IoU_'{cls_name}'", _miou, sync_dist=True)
            self.log(f"{phase}_F1_'{cls_name}'", _f1, sync_dist=True)

        mean_iou = iou_per_class.mean()
        mean_f1 = f1_per_class.mean()
        self.log(f"{phase}_mIoU", mean_iou, sync_dist=True)
        self.log(f"{phase}_mF1", mean_f1, sync_dist=True)

    def training_step(self, batch, batch_idx):
        images, masks, img_metas = batch
        loss = self._step(images, masks, img_metas, "train")
        return loss

    # def validation_step(self, batch, batch_idx):#, dataloader_idx=0):
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images, masks, img_metas = batch
        self._step(images, masks, img_metas, f"val_{dataloader_idx}")

    def on_validation_epoch_end(self) -> None:
        for di in range(self.num_val_dataloaders):
            self._on_epoch_end(f"val_{di}")

    def on_train_epoch_end(self) -> None:
        self._on_epoch_end("train")

    def _plot_predictions(self, images: torch.Tensor, masks: torch.Tensor, pred_masks: torch.Tensor, prob_masks: torch.Tensor, phase: str, save_locally=None):

        plts = 5
        imgs_per_subplot = min(plts, images.shape[0])
        fig, axes = plt.subplots(imgs_per_subplot, plts, figsize=(24, 24), squeeze=False)

        for i in range(imgs_per_subplot):
            img = images[i].cpu()
            mask = masks[i].cpu()
            pred = pred_masks[i].cpu()
            prob = prob_masks[i].cpu()[1] # probability of class 1

            diff = mask == pred
            diff[mask == self.ignore_index] = True  # ignore classes should be displayed as correct

            mask = self.visualizer(mask.unsqueeze(0)).squeeze()
            pred = self.visualizer(pred.unsqueeze(0)).squeeze()

            axes[i, 0].imshow(img.permute(1, 2, 0))
            axes[i, 1].imshow(mask.permute(1, 2, 0))
            axes[i, 2].imshow(pred.permute(1, 2, 0))
            axes[i, 3].imshow(prob)
            axes[i, 4].imshow(diff.unsqueeze(-1).long())

        plt.tight_layout()
        [make_axes_invisible(ax) for ax in axes.flatten()]

        if save_locally is not None:
            plt.savefig(save_locally,)
            plt.close()
        else:
            try:
                self.logger.experiment.log({f"{phase}_predictions": fig})
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                print("Logger cannot save image. Displaying instead.")
                plt.show()
            plt.close()
