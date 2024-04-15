import argparse
import os

import torch
import lightning.pytorch as pl
import yaml
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from TADP.tadp_seg import TADPSeg
from TADP.tadp_objdet import TADPObj
from datasets.VOCDataset import VOCDataset
from datasets.datamodules import PascalVOCDataModule
import numpy as np
import datetime
from datasets.VOC_config import cfg as voc_cfg

from TADP.utils.detection_utils import voc_classes, empty_collate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="segmentation")  # options are segmentation, detection
    parser.add_argument("--val_dataset_name", default="pascal", type=str)
    parser.add_argument("--cross_domain_target", default="watercolor", type=str)
    parser.add_argument('--cross_blip_caption_path', type=str, default='blip_captions/watercolor_captions.json',
                        help='path to cross blip captions')
    parser.add_argument('--dreambooth_checkpoint', type=str, default=None, help='path to dreambooth checkpoint')
    parser.add_argument('--textual_inversion_token_path', type=str, default=None,
                        help='path to textual inversion token path')
    parser.add_argument("--train_dataset_name", default="pascal", type=str)

    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument("--log_model_every_n_epochs", type=int, default=-1)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--wandb_group", type=str, default="FT_baseline_runs")

    # debugging presets
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--val_debug", action='store_true', default=False)
    parser.add_argument("--wandb_debug", action='store_true', default=False)
    # test remote machine if it is working without wasting time downloading datasets
    parser.add_argument("--test_machine", action='store_true', default=False)

    # experiment parameters
    parser.add_argument("--model_name", type=str, default="DeeplabV3Plus")
    parser.add_argument("--from_scratch", action='store_true', default=False)
    parser.add_argument("--max_epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument('--eval_dataset', type=str, nargs='+', default=['pascal'])
    parser.add_argument('--optimizer_config_preset', type=int, default=0)
    parser.add_argument('--strategy', type=str, default='')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--freeze_backbone', type=int, default=0)
    parser.add_argument('--freeze_batchnorm', type=int, default=0)

    parser.add_argument("--accum_grad_batches", type=int, default=1)
    parser.add_argument("--freeze_text_adapter", type=int, default=1)

    parser.add_argument('--train_dataset', type=str, nargs='+', default=['VOC2012_ext'])
    parser.add_argument('--train_max_samples', type=int, default=None)

    # TADP specific parameters
    parser.add_argument('--text_conditioning', type=str, default='class_emb')
    parser.add_argument('--min_blip', type=int, default=0)
    parser.add_argument('--task_inversion_lr', type=float, default=0.002)
    parser.add_argument('--use_scaled_encode', action='store_true', default=False)
    parser.add_argument('--append_self_attention', action='store_true', default=False)
    parser.add_argument('--use_decoder_features', action='store_true', default=False)
    parser.add_argument('--use_text_adapter', action='store_true', default=False)
    parser.add_argument('--cond_stage_trainable', action='store_true', default=False)
    parser.add_argument('--blip_caption_path', type=str, default=None)
    parser.add_argument('--no_attn', action='store_true', default=False)
    parser.add_argument('--use_only_attn', action='store_true', default=False)
    parser.add_argument('--present_class_embeds_only', action='store_true', default=False)

    parser.add_argument('--trainer_ckpt_path', type=str, default=None)
    parser.add_argument('--save_checkpoint_path', type=str, default='')
    parser.add_argument('--train_debug', action='store_true', default=False)
    args = parser.parse_args()

    model_name = args.model_name
    pretrained = not args.from_scratch
    max_epochs = args.max_epochs
    batch_size = args.batch_size
    val_batch_size = args.val_batch_size
    num_workers = args.num_workers
    log_freq = args.log_freq
    log_every_n_steps = args.log_every_n_steps
    wandb_group = args.wandb_group
    wandb_name = args.exp_name
    checkpoint = args.checkpoint
    strategy = args.strategy
    accum_grad_batches = args.accum_grad_batches
    freeze_text_adapter = args.freeze_text_adapter
    log_model_every_n_epochs = args.log_model_every_n_epochs
    blip_caption_path = args.blip_caption_path  # depends on dataset
    use_decoder_features = args.use_decoder_features
    cond_stage_trainable = args.cond_stage_trainable
    save_checkpoint_path = args.save_checkpoint_path

    save_topk = 1
    save_last = True
    limit_train_batches = None
    limit_val_batches = None
    if args.debug:
        max_epochs = 4
        os.environ["WANDB_MODE"] = "dryrun"
        num_workers = 0
        batch_size = 16 if 'TADP' not in args.model else batch_size
        log_freq = 1
        save_last = False
        save_topk = 0
    if args.wandb_debug:
        num_workers = 0
        batch_size = 16 if 'TADP' not in args.model else batch_size
        limit_val_batches = 2
        limit_train_batches = 2
        wandb_group = "wandb_debugging"
        wandb_name = f"dummy_{datetime.datetime.now().__str__()}"
        save_last = False
        save_topk = 0
    if args.val_debug:
        limit_val_batches = 2
        limit_train_batches = 2
        os.environ["WANDB_MODE"] = "dryrun"
    if args.test_machine:
        args.train_dataset = ['dummy_data']
        args.eval_dataset = ['dummy_data']
        os.environ["WANDB_MODE"] = "dryrun"

    pl.seed_everything(args.seed)

    if args.task == 'segmentation':
        train_datasets = []
        if 'VOC2012_ext' in args.train_dataset:
            print('Using VOC2012_ext dataset')
            voc_train_dataset = VOCDataset('./', 'VOC2012', voc_cfg, 'train', True)
            train_datasets.append(voc_train_dataset)
            if blip_caption_path is None:
                blip_caption_path = f'blip_captions/pascal_captions_min={args.min_blip}_max=77.json'

        val_loaders = []
        for v_dset in args.eval_dataset:
            if v_dset == 'pascal':
                val_dataset = VOCDataset('./', 'VOC2012', voc_cfg, 'val', False)

            val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
                                    drop_last=True)
            val_loaders.append(val_loader)

        class_names = train_datasets[0].classes

        train_dataset = ConcatDataset(train_datasets)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=True)

    elif args.task == 'detection':
        from torchvision.datasets import VOCDetection
        base_path = './data/'
        dl = True
        pascal_2012_trainval = VOCDetection(os.path.join(base_path, "VOCdevkit/VOC2012"), year='2012',
                                            image_set='trainval',
                                            download=dl)
        pascal_2007_trainval = VOCDetection(os.path.join(base_path, "VOCdevkit/VOC2007"), year='2007',
                                            image_set='trainval',
                                            download=dl)

        # pascal_2012_val = VOCDetection(os.path.join(base_path, "VOCdevkit/VOC2012"), year='2012', image_set='val',
        #                                download=dl)
        pascal_2007_test = VOCDetection(os.path.join(base_path, "VOCdevkit/VOC2007"), year='2007', image_set='test',
                                        download=dl)

        train_datasets = ConcatDataset([pascal_2012_trainval, pascal_2007_trainval])
        pascal_val_dataset = pascal_2007_test

        val_loaders = []

        class_names = voc_classes

        train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=True, collate_fn=empty_collate)


    else:
        raise ValueError(f'Invalid task: {args.task}')

    assert len(train_datasets) > 0, 'No valid train dataset specified'

    # for tdi, td in enumerate(train_datasets):
    #     print(f'Train dataset {args.train_dataset[tdi]}: {len(td)} samples')

    class_embedding_path = './data/pascal_class_embeddings.pth'
    cfg = yaml.load(open("./sd_tune.yaml", "r"), Loader=yaml.FullLoader)
    cfg["annotator"]["type"] = "ground_truth"
    cfg["stable_diffusion"]["use_diffusion"] = True
    cfg["max_epochs"] = max_epochs
    cfg["dataset_len"] = len(train_loader)
    cfg["freeze_text_adapter"] = freeze_text_adapter

    if args.no_attn and args.use_only_attn:
        raise ValueError('Cannot use both no_attn and use_only_attn')

    model_kwargs = {}
    cfg['text_conditioning'] = args.text_conditioning
    cfg['blip_caption_path'] = blip_caption_path
    cfg['use_scaled_encode'] = args.use_scaled_encode
    cfg['class_names'] = class_names
    cfg['append_self_attention'] = args.append_self_attention
    cfg['use_text_adapter'] = args.use_text_adapter
    cfg['cond_stage_trainable'] = cond_stage_trainable
    if args.append_self_attention:
        model_kwargs['unet_config'] = {'attn_selector': 'up_cross+down_cross-up_self+down_self'}
    model_kwargs['unet_config'] = {'use_attn': not args.no_attn}  # default for use_attn ends up true
    cfg['use_attn'] = not args.no_attn
    cfg['use_only_attn'] = args.use_only_attn
    cfg['use_decoder_features'] = use_decoder_features
    cfg['use_token_embeds'] = False
    cfg['present_class_embeds_only'] = args.present_class_embeds_only

    cfg['dreambooth_checkpoint'] = args.dreambooth_checkpoint
    cfg['textual_inversion_token_path'] = args.textual_inversion_token_path
    cfg['dataset_len'] = len(train_datasets)
    cfg['val_dataset_name'] = args.val_dataset_name
    cfg['cross_blip_caption_path'] = args.cross_blip_caption_path

    if args.task == 'segmentation':
        model = TADPSeg(class_names=VOCDataset.classes,
                        ignore_index=VOCDataset.ignore_index,
                        visualizer_kwargs=VOCDataset.visualizer_kwargs,
                        num_val_dataloaders=len(val_loaders),
                        class_embedding_path=class_embedding_path,
                        cfg=cfg,
                        **model_kwargs
                        )
    elif args.task == 'detection':

        from datasets.VOCDataset import classes
        model = TADPObj(class_embedding_path="./data/pascal_class_embeddings.pth", cfg=cfg, class_names=classes,
                        freeze_backbone=args.freeze_backbone)

        if args.val_dataset_name == 'cross':
            model.dataset_name = args.cross_domain_target
        else:
            model.dataset_name = args.val_dataset_name
        model.init_evaluator()

        cross_domain_train = PascalVOCDataModule(
            os.path.join(base_path, "cross-domain-detection/datasets/" + args.cross_domain_target),
            "train", classes)
        cross_domain_val = PascalVOCDataModule(
            os.path.join(base_path, "cross-domain-detection/datasets/" + args.cross_domain_target),
            "test", classes)

        if args.train_dataset_name == 'pascal':
            pass  # already pascal
        elif args.train_dataset_name == 'cross':
            train_datasets = cross_domain_train
        else:
            raise ValueError('train dataset name not recognized')

        if args.train_debug:
            train_datasets = torch.utils.data.Subset(train_datasets, range(100))
            cross_domain_val = torch.utils.data.Subset(cross_domain_val, range(100))

        if args.val_dataset_name == 'pascal':
            val_loader = DataLoader(pascal_val_dataset, batch_size=val_batch_size, shuffle=False,
                                    num_workers=num_workers,
                                    drop_last=True, collate_fn=empty_collate)

        elif args.val_dataset_name == 'cross':
            val_loader = DataLoader(cross_domain_val, batch_size=val_batch_size, shuffle=False,
                                    num_workers=num_workers,
                                    drop_last=True, collate_fn=empty_collate)

        else:
            raise ValueError('val dataset name not recognized')

        val_loaders.append(val_loader)

        train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  drop_last=True, collate_fn=empty_collate)

    if checkpoint is not None and pretrained:
        try:
            state_dict = torch.load(checkpoint)["state_dict"]
            # making older state dicts compatible with current model
            if list(state_dict.keys())[0] != list(model.state_dict().keys())[0]:
                print('Loading pretrained model with different key names')
                # replace each key in state_dict with the corresponding key in model.state_dict()
                state_dict = {list(model.state_dict().keys())[i]: list(state_dict.values())[i] for i in
                              range(len(state_dict))}
            model.load_state_dict(state_dict, strict=True)
        except KeyError:
            model.load_state_dict(torch.load(checkpoint))

    checkpoint_callbacks = []
    for i in range(len(val_loaders)):
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor=f'val_{i}_loss_epoch/dataloader_idx_{i}' if model_name == 'DeeplabV3Plus' else f'val_{i}_loss',
            dirpath=f'./checkpoints/{args.exp_name}/',
            filename=f'model_checkpoint_{args.exp_name}',
            # save_top_k=save_topk,  # Save top1 Why?? this is 40GB of checkpoints -->> # Save all checkpoints.
            save_top_k=-1 if log_model_every_n_epochs > 0 else save_topk,
            mode='min',  # Mode for comparing the monitored metric
            save_last=save_last,
            every_n_epochs=log_model_every_n_epochs if log_model_every_n_epochs > 0 else None,
        )
        checkpoint_callbacks.append(checkpoint_callback)

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    callbacks = [lr_callback] + checkpoint_callbacks

    logger = pl.loggers.WandbLogger(
        name=wandb_name or "segmentation_test, model={}".format(model_name) + "usingDecoderFeatures={}".format(
            use_decoder_features),
        group=wandb_group or "markusShit",
        project="madman",
        log_model="all",
        entity="vision-lab",
    )

    # watch model
    logger.watch(model, log="all", log_freq=log_freq)

    print("batch_size: ", batch_size)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        strategy=strategy if strategy != '' else 'auto',  # check somehow ddp is using more gpu memory than auto
        logger=logger,
        max_epochs=max_epochs,
        log_every_n_steps=log_every_n_steps,
        callbacks=callbacks,
        limit_train_batches=limit_train_batches,  # None unless --wandb_debug or --val_debug flag is set
        limit_val_batches=limit_val_batches,  # None unless --wandb_debug or --val_debug flag is set
        check_val_every_n_epoch=args.check_val_every_n_epoch,  # None unless --wandb_debug flag is set
        sync_batchnorm=True if args.num_gpus > 1 else False,
        accumulate_grad_batches=accum_grad_batches,
    )
    if trainer.global_rank == 0:
        logger.experiment.config.update(args)

    if not args.debug or args.val_debug:
        trainer.validate(model, dataloaders=val_loaders)

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loaders,
        ckpt_path=args.trainer_ckpt_path,
    )
    # save the model
    if save_checkpoint_path != '':
        save_model_name = f'{args.exp_name}.ckpt'
        # results paths
        if not os.path.exists(save_checkpoint_path):
            os.makedirs(save_checkpoint_path)
        torch.save(model.state_dict(), save_checkpoint_path + save_model_name)


if __name__ == "__main__":
    main()
