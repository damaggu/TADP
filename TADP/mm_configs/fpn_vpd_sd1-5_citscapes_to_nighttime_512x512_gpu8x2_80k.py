_base_ = [
    '_base_/models/fpn_r50.py',
    '_base_/datasets/cityscapes_to_nighttime_512x512.py', ### for cross domain
    '_base_/default_runtime.py',
    '_base_/schedules/schedule_80k.py',
]
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='TADPSeg',
    sd_path='checkpoints/v1-5-pruned-emaonly.ckpt',
    neck=dict(
        type='FPN',
        # in_channels=[320, 790, 1430, 1280],
        in_channels=[320, 659, 1299, 1280],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        # num_classes=150,
        num_classes=19,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        # norm_cfg=norm_cfg,
        # align_corners=False,
    ),
)
lr_config = dict(policy='poly', power=1, min_lr=0.0, by_epoch=False,
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)
# optimizer_config = None
optimizer = dict(type='AdamW', lr=0.00008, weight_decay=0.001, # before 0.005
                 paramwise_cfg=dict(custom_keys={'unet': dict(lr_mult=0.1),
                                                 'encoder_vq': dict(lr_mult=0.0),
                                                 'text_encoder': dict(lr_mult=0.0),
                                                 'norm': dict(decay_mult=0.)}))
data = dict(samples_per_gpu=8, workers_per_gpu=8)
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
evaluation = dict(interval=4000, metric='mIoU')
# Meta Information for Result Analysis
name = 'citscapes_to_nighttime_512x512'
exp = 'basic'
name_dataset = 'cityscapes_to_nighttime_512x512'
name_architecture = 'fpn_vpd_sd1-5_512x512'
name_encoder = 'vpd_sd1-5'
name_decoder = 'fpn'
name_opt = 'adamw_vpd_sd1-5_ade20k'