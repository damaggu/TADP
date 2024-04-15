###---- TADP Depth ----###
#WANDB_MODE=dryrun python train_tadp_depth.py --epochs 1 --batch_size 2 --sanity_check

###---- TADP Sem. Segmentation ADE ----###
#WANDB_MODE=dryrun python train_tadp_mm.py TADP/mm_configs/seg_ade20k_sanitycheck.py

###---- TADP Sem. Segmentation Pascal ----###

###---- TADP Object Detection (cros domain) ----###

### inference test
#WANDB_MODE=dryrun python train_tadp.py --task detection --val_batch_size 1 --num_workers 0 --batch_size 1 --freeze_backbone 0 --max_epochs 1 --check_val_every_n_epoch 1 --model_name VPDObj --text_conditioning blip+dashcam --cross_domain_target watercolor --val_dataset_name cross --blip_caption_path /mnt/m2_data/PycharmProjects/neurips2023_v2/blip_captions/pascal_object_captions_min40_max77.json --cross_blip_caption_path /mnt/m2_data/PycharmProjects/neurips2023_v2/blip_captions/watercolor_captions.json --checkpoint /mnt/m2_data/PycharmProjects/neurips2023_v2/madman_pascal_watercolor_checkpoints/TextCond_BlipANDDashcam_WatercolorFT_model.pth
### todo: write inference only script
### todo: cfgfy caption modifier

### training test
#WANDB_MODE=dryrun python train_tadp.py --train_debug --task detection --val_batch_size 1 --batch_size 1 --freeze_backbone 1 --max_epochs 1 --check_val_every_n_epoch 1 --model_name VPDObj --text_conditioning blip+watercolorTI+train --cross_domain_target watercolor --val_dataset_name cross --blip_caption_path /mnt/m2_data/PycharmProjects/neurips2023_v2/blip_captions/pascal_object_captions_min40_max77.json --save_checkpoint_path ./testmodel --textual_inversion_token_path /mnt/m2_data/PycharmProjects/neurips2023_v2/TI_tokens/pascal_watercolor_5/learned_embeds-steps-3000.bin --cross_blip_caption_path /mnt/m2_data/PycharmProjects/neurips2023_v2/blip_captions/watercolor_captions.json
# cross domain train test

# inference test