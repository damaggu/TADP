

class Object_Detection_lightning_Model(pl.LightningModule):
    def __init__(self, backbone, head, freeze_backbone=False):
        super().__init__()
        self.backbone = backbone.to(self.device)
        self.head = head.to(self.device)

        if freeze_backbone:
            # freeze the backbone
            for param in self.backbone.parameters():
                param.requires_grad = False

        # loss for object detection
        self.criterion = torch.nn.CrossEntropyLoss()

        self.all_predictions = []
        self.all_targets = []

        mAP = MeanAveragePrecision()

        self.metric = mAP

        self.dataset_name = "voc"

    def init_evaluator(self):
        self.evaluator = PascalVOCDetectionEvaluator(dataset_name='voc_2007_test')
        if self.dataset_name == "watercolor":
            self.evaluator._anno_file_template = './data/cross-domain-detection/datasets/watercolor/Annotations/{}.xml'
            self.evaluator._image_set_path = './data/cross-domain-detection/datasets/watercolor/ImageSets/Main/test.txt'
            self.evaluator._class_names = REDUCED_CLASS_NAMES[1:]
            self.evaluator._is_2007 = True
        elif self.dataset_name == "comic":
            self.evaluator._anno_file_template = './data/cross-domain-detection/datasets/comic/Annotations/{}.xml'
            self.evaluator._image_set_path = './data/cross-domain-detection/datasets/comic/ImageSets/Main/test.txt'
            self.evaluator._class_names = REDUCED_CLASS_NAMES[1:]
            self.evaluator._is_2007 = True
            print('d')
        self.evaluator.reset()

    def forward(self, x, y=None, train=True):
        x = [torchvision.transforms.ToTensor()(x) for x in x]
        if train:
            if y is not None:
                orig_images, y = self.head.transform(x, y)
            else:
                orig_images, _ = self.head.transform(x)
        else:
            # just imagenet normalization
            orig_images = [torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225])(x) for x in x]
            orig_images = [datapoints.Image(x) for x in orig_images]
            # resize to 800x800
            bboxes = [datapoints.BoundingBox(y['boxes'], format="XYXY", spatial_size=orig_images[i].shape[1:]) for i, y
                      in
                      enumerate(y)]
            from torchvision.transforms import v2 as T
            from torchvision.transforms.v2 import functional as F
            trans = T.Compose(
                [
                    T.Resize((800, 800)),
                ]
            )
            trans([orig_images[0]], [bboxes[0]])
            a = [trans(orig_images[i], bboxes[i]) for i in range(len(orig_images))]
            orig_images = [a[0] for a in a]
            bboxes = [a[1] for a in a]
            for i, _ in enumerate(y):
                y[i]['boxes'] = bboxes[i].clone().detach()

            print('d')
            # to detection list
            orig_images = torch.stack(orig_images)
            orig_images = torchvision.models.detection.image_list.ImageList(orig_images,
                                                                            image_sizes=[(800, 800)] * len(orig_images))

        orig_images_tensors = orig_images.tensors.to(self.device)
        features = self.backbone(orig_images_tensors)
        predi = self.head(images=orig_images, images_tensor=orig_images_tensors, features=features, targets=y)

        return predi, y, orig_images.tensors.to(self.device)

    def training_step(self, batch, batch_idx):
        x = [x[0] for x in batch]
        annotations = [x[1] for x in batch]
        y = annotations
        # calulate metrics for validation
        gt_boxes, gt_labels, gt_scores = annotations_to_boxes(y)

        targets = []
        for i in range(len(gt_boxes)):
            target = {
                "image_id": i,
                "boxes": gt_boxes[i].to(self.device),  # (n_objects, 4)
                "labels": gt_labels[i].to(self.device),  # (n_objects)
            }
            targets.append(target)
        y = targets
        losses, _, _ = self(x, y)
        # TODO: check in detectron engine how the losses are used
        total_loss = sum(losses.values())
        self.log("train_loss", total_loss)

        # print("----------------------")
        # print(self.head.rpn.head.conv[0][0].weight[0][0][0])

        # del suff
        del losses
        del x
        del y
        del gt_boxes
        del gt_labels
        del gt_scores
        del targets
        return total_loss

    def set_dataset_name(self, dataset_name):
        self.dataset_name = dataset_name

    def validation_step(self, batch, batch_idx):
        try:
            images, gt_boxes, gt_labels, annotation = zip(*batch)
            img_ids = [
                str(annot['image_id'])
                for
                annot in annotation]
            img_metas = {}
            img_metas['img_id'] = img_ids
            targets = []
            for i in range(len(gt_boxes)):
                target = {
                    "image_id": img_metas['img_id'][i],
                    "boxes": gt_boxes[i].to(self.device),  # (n_objects, 4)
                    "labels": gt_labels[i].to(self.device),  # (n_objects)
                }
                targets.append(target)
            y = targets
            images = [torchvision.transforms.ToPILImage()(x) for x in images]
        except:
            images, annotation = zip(*batch)

            img_ids = [
                annot['annotation']['filename'].split('.jpg')[0]
                for
                annot in annotation]
            img_metas = {}
            img_metas['img_id'] = img_ids
            # img_metas = [{'img_id': [annot['annotation']['filename'].split('.jpg')[0]]} for annot in annotation]
            # loss = self._step(images, masks, img_metas, "train")
            # sch = self.lr_schedulers()
            # sch.step()
            gt_boxes, gt_labels, gt_scores = annotations_to_boxes(annotation)

            targets = []
            for i in range(len(gt_boxes)):
                target = {
                    "image_id": img_metas['img_id'][i],
                    "boxes": gt_boxes[i].to(self.device),  # (n_objects, 4)
                    "labels": gt_labels[i].to(self.device),  # (n_objects)
                }
                targets.append(target)
            y = targets

        results, resized_targets, resized_imgs = self(images, y, train=False)

        # calulate metrics for validation
        # TODO: check in detectron engine how the losses are used
        pred_boxes = [x["boxes"] for x in results]
        pred_labels = [x["labels"] for x in results]
        pred_scores = [x["scores"] for x in results]

        original_shapes = [x.size for x in images]
        resized_shapes = [x.shape[1:] for x in resized_imgs]
        resizing_ratios = [(x[0] / y[0], x[1] / y[1]) for x, y in zip(original_shapes, resized_shapes)]

        # resize boxes to original size based on resizing ratios
        for i in range(len(pred_boxes)):
            pred_boxes[i][:, 0] *= resizing_ratios[i][0]
            pred_boxes[i][:, 1] *= resizing_ratios[i][1]
            pred_boxes[i][:, 2] *= resizing_ratios[i][0]
            pred_boxes[i][:, 3] *= resizing_ratios[i][1]

        use_wandb = True

        preds = []
        for i in range(len(pred_boxes)):
            boxes = pred_boxes[i]
            labels = pred_labels[i]
            scores = pred_scores[i]
            orig_shapes = original_shapes[i]

            # labels + 1 because 0 is background
            # labels = labels + 1

            # idx_scores = scores > 0.5
            # boxes = boxes[idx_scores]
            # labels = labels[idx_scores]
            # scores = scores[idx_scores]

            if self.dataset_name == "watercolor" or self.dataset_name == "comic":
                labels_as_words = [DETECTRON_VOC_CLASS_NAMES[l] for l in labels]

                #
                # remap labels from DETECTRON_VOC_CLASS_NAMES to REDUCED_CLASS_NAMES
                new_labels = []
                skipped_label_idx = []
                for idx, label in enumerate(labels_as_words):
                    try:
                        new_label = REDUCED_CLASS_NAMES.index(label)
                        new_labels.append(new_label)
                    except:
                        skipped_label_idx.append(idx)
                labels = torch.tensor(new_labels)
                # remove skipped boxes
                boxes = [boxes[i] for i in range(len(boxes)) if i not in skipped_label_idx]
                scores = [scores[i] for i in range(len(scores)) if i not in skipped_label_idx]
                if len(boxes) > 0:
                    scores = torch.tensor(scores)
                    boxes = torch.stack(boxes)
                else:
                    scores = torch.tensor([])
                    boxes = torch.tensor([])

            labels = labels.tolist()

            # pred = {"boxes": boxes, "labels": labels, "scores": scores, "image_id": torch.tensor([i])}
            pred = {"instances": Instances(image_size=orig_shapes, pred_boxes=boxes, pred_classes=labels,
                                           scores=scores)}
            preds.append(pred)

        gt = []
        for i in range(len(gt_boxes)):
            # gt.append({"boxes": gt_boxes[i], "labels": gt_labels[i]})
            # gt_labels_as_words = [DETECTRON_VOC_CLASS_NAMES[l] for l in resized_targets[i]["labels"]]
            gt_labels_as_words = resized_targets[i]["labels"].tolist()
            gt.append({
                # "boxes": resized_targets[i]["boxes"],
                "boxes": gt_boxes[i],
                "labels": gt_labels_as_words,
                "image_id": resized_targets[i]["image_id"]
            })

        # use detectron2  evaluator
        self.evaluator.process(gt, preds)

        del gt
        del preds

        # prep data for torchmetrics COCO API
        # remove 'image_id' from gt

        preds = []
        for i in range(len(pred_boxes)):
            boxes = pred_boxes[i]
            labels = pred_labels[i]
            scores = pred_scores[i]

            pred = {"boxes": boxes, "labels": labels, "scores": scores, "image_id": torch.tensor([i])}
            preds.append(pred)

        gt = []
        for i in range(len(gt_boxes)):
            gt.append({"boxes": gt_boxes[i], "labels": gt_labels[i]})
            # gt.append({"boxes": resized_targets[i]["boxes"], "labels": resized_targets[i]["labels"], })

        gt = [{k: v.to(self.device) for k, v in t.items()} for t in gt]
        preds = [{k: v.to(self.device) for k, v in t.items()} for t in preds]

        # self.all_predictions.extend(preds)
        # self.all_targets.extend(gt)

        res = self.metric(preds, gt)
        self.log("val_mAP_{}".format(self.dataset_name), res['map'], on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=len(batch))
        self.log("val_mAP_50_{}".format(self.dataset_name), res['map_50'], on_step=False, on_epoch=True, prog_bar=True,
                 logger=True, batch_size=len(batch))

        # for the first 5 images, save the images with bounding boxes using the torchvision function utils
        # in folder ./object_detection_results/
        res_path = "./object_detection_results/"
        if not os.path.exists(res_path):
            os.makedirs(res_path)

        if batch_idx < 10:
            for i, (img, result, gt) in enumerate(zip(images, results, gt)):
                if not type(images) == torch.Tensor:
                    img = torchvision.transforms.ToTensor()(img)
                img = img.permute(1, 2, 0).cpu().numpy()
                # denormalize from imagenet mean and std
                img = img * torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3).cpu().numpy()
                img = img + torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3).cpu().numpy()
                img = img * 255
                img = img.astype(np.uint8)
                img_pil = Image.fromarray(img)

                boxes = result["boxes"]
                labels = result["labels"]
                scores = result["scores"]
                filtered_boxes = boxes[scores > 0.5]
                filtered_labels = labels[scores > 0.5]

                img_tensor = torchvision.transforms.functional.to_tensor(img_pil)

                # Plot predicted bounding boxes
                fig, ax = plt.subplots()
                ax.imshow(torchvision.transforms.functional.to_pil_image(img_tensor))

                for box, label in zip(filtered_boxes, filtered_labels):
                    xmin, ymin, xmax, ymax = box.tolist()
                    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='red',
                                         linewidth=2)
                    ax.add_patch(rect)
                    ax.text(xmin, ymin, label.item(), bbox=dict(facecolor='red', alpha=0.5))

                plt.axis('off')

                # Save or log predicted image
                if use_wandb:
                    self.logger.experiment.log({"predicted_image": wandb.Image(fig)})
                else:
                    plt.savefig(f"./object_detection_results/epoch_{self.current_epoch}_{batch_idx}_{i}_pred.jpg",
                                bbox_inches='tight', pad_inches=0)

                boxes = gt["boxes"].to(torch.int64)
                labels = gt["labels"].to(torch.int64)

                # Plot ground truth bounding boxes
                fig, ax = plt.subplots()
                ax.imshow(torchvision.transforms.functional.to_pil_image(img_tensor))

                for box, label in zip(boxes, labels):
                    xmin, ymin, xmax, ymax = box.tolist()
                    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor='green',
                                         linewidth=2)
                    ax.add_patch(rect)
                    ax.text(xmin, ymin, label.item(), bbox=dict(facecolor='green', alpha=0.5))

                plt.axis('off')

                # Save or log ground truth image
                if use_wandb:
                    self.logger.experiment.log({"ground_truth_image": wandb.Image(fig)})
                else:
                    plt.savefig(f"./object_detection_results/epoch_{self.current_epoch}_{batch_idx}_{i}_gt.jpg",
                                bbox_inches='tight', pad_inches=0)

        # delete stuff
        del gt
        del preds
        del results
        del images
        del y
        del gt_boxes
        del gt_labels
        try:
            del gt_scores
        except:
            pass
        del pred_boxes
        del pred_labels
        del pred_scores

        return res

    def on_validation_epoch_end(self):
        # do full validation
        # self.validation_step(self.val_dataloader())
        print('validation epoch end')
        self.evaluator._is_2007 = True
        res = self.evaluator.evaluate()
        bbox_aps = res['bbox']
        self.log("pascal_eval_2007_val_mAP_{}".format(self.dataset_name), bbox_aps['AP'], on_step=False, on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log("pascal_eval_2007_val_mAP_50_{}".format(self.dataset_name), bbox_aps['AP50'], on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log("pascal_eval_2007_val_mAP_75_{}".format(self.dataset_name), bbox_aps['AP75'], on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        self.evaluator._is_2007 = False
        res = self.evaluator.evaluate()
        bbox_aps = res['bbox']
        self.log("pascal_eval_2012_val_mAP_{}".format(self.dataset_name), bbox_aps['AP'], on_step=False, on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log("pascal_eval_2012_val_mAP_50_{}".format(self.dataset_name), bbox_aps['AP50'], on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log("pascal_eval_2012_val_mAP_75_{}".format(self.dataset_name), bbox_aps['AP75'], on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)

        self.evaluator.reset()

        # also eval using chainercv
        # prepare nd arrays for eval with chainercv
        # all_pred_boxes = []
        # all_pred_labels = []
        # all_pred_scores = []
        # all_gt_boxes = []
        # all_gt_labels = []
        # for i in range(len(self.all_predictions)):
        #     pred_boxes = self.all_predictions[i]['boxes']
        #     pred_labels = self.all_predictions[i]["labels"]
        #     pred_scores = self.all_predictions[i]["scores"]
        #
        #     all_pred_boxes.append(pred_boxes)
        #     all_pred_labels.append(pred_labels)
        #     all_pred_scores.append(pred_scores)
        #
        #     gt_boxes = self.all_targets[i]["boxes"]
        #     gt_labels = self.all_targets[i]["labels"]
        #
        #     all_gt_boxes.append(gt_boxes)
        #     all_gt_labels.append(gt_labels)
        #
        # all_pred_boxes = torch.cat(all_pred_boxes)
        # all_pred_labels = torch.cat(all_pred_labels)
        # all_pred_scores = torch.cat(all_pred_scores)
        # all_gt_boxes = torch.cat(all_gt_boxes)
        # all_gt_labels = torch.cat(all_gt_labels)
        #
        # all_pred_boxes = all_pred_boxes.cpu().numpy()
        # all_pred_labels = all_pred_labels.cpu().numpy()
        # all_pred_scores = all_pred_scores.cpu().numpy()
        # all_gt_boxes = all_gt_boxes.cpu().numpy()
        # all_gt_labels = all_gt_labels.cpu().numpy()
        #
        # res = eval_detection_voc(all_pred_boxes, all_pred_labels, all_pred_scores, all_gt_boxes, all_gt_labels,
        #                          use_07_metric=True)
        #
        # res = eval_detection_voc(self.all_predictions, self.all_targets, use_07_metric=True)
        # self.log("chainercv_val_mAP_{}".format(self.dataset_name), res['map'], on_step=False, on_epoch=True,
        #          prog_bar=True,
        #          logger=True)
        # self.log("chainercv_val_AP_{}".format(self.dataset_name), res['ap'], on_step=False, on_epoch=True,
        #          prog_bar=True,
        #          logger=True)

        self.all_predictions = []
        self.all_targets = []

        pass

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.head.parameters(), lr=1e-3)
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005, weight_decay=0.0001, momentum=0.9)
        # learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.2, threshold=0.0001, patience=1)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_mAP_{}'.format(self.dataset_name)
        }

    # TODO...

    import random

    def subsample_dataset(dataset, target_class, oversample_ratio, subset_length):
        indices = []
        target_indices = []
        other_indices = []

        # Iterate over the dataset to separate indices by target class and other classes
        for idx, data in enumerate(dataset):
            if data[1]['labels'] == target_class:
                target_indices.append(idx)
            else:
                other_indices.append(idx)

        # Sample the target class indices based on the oversample ratio
        num_target_samples = int(min(len(target_indices) * oversample_ratio, subset_length))
        sampled_target_indices = random.choices(target_indices, k=num_target_samples)

        # Combine the sampled target class indices with other class indices
        indices.extend(sampled_target_indices)
        remaining_samples = max(0, subset_length - num_target_samples)
        indices.extend(random.sample(other_indices, k=remaining_samples))

        # Create a subset of the dataset using the sampled indices
        subsampled_dataset = Subset(dataset, indices)

        return subsampled_dataset

    ### insepect gradients if necessary
    # def on_before_optimizer_step(self, optimizer, watch_gradients=False):
    #     # Compute the 2-norm for each layer
    #     # If using mixed precision, the gradients are already unscaled here
    #     norms = [grad_norm(p, norm_type=2.0) for p in self.backbone.parameters()]
    #     self.log_dict({"backbone_grad_norm_{}".format(i): norm for i, norm in enumerate(norms)}, prog_bar=True)
    #     head_norms = [grad_norm(p, norm_type=2.0) for p in self.head.parameters()]
    #     self.log_dict({"head_grad_norm_{}".format(i): norm for i, norm in enumerate(head_norms)}, prog_bar=True)

