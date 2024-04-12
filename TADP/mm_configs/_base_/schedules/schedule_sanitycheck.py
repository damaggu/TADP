# optimizer
optimizer = dict(type='SGD', lr=0.01, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False)
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=1)
checkpoint_config = dict(by_epoch=False, interval=10, max_keep_ckpts=0)
evaluation = dict(interval=10, metric='mIoU', save_best='mIoU', rule='greater')
