# yapf:disable
import os

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='MMSegWandbHook', by_epoch=False, init_kwargs=dict(project=os.environ.get("WANDB_PROJECT") or 'tadp_seg',
                                                                      entity=os.environ.get("WANDB_ENTITY") or 'default',
                                                                      group=os.environ.get("GROUP_NAME") or f'vpd_mmseg',
                                                                      name=os.environ.get("RUN_NAME") or f'vpd_ade20k'
                                                                      ),
             log_checkpoint=False,
             log_checkpoint_metadata=False,
             num_eval_images=0,
             )
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
find_unused_parameters = True
