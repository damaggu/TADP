###---- TADP Depth ----###

###---- TADP Sem. Segmentation ADE ----###
WANDB_MODE=dryrun python train_tadp_mm.py TADP/mm_configs/seg_ade20k_sanitycheck.py

###---- TADP Sem. Segmentation Pascal ----###

###---- TADP Object Detection (cros domain) ----###

### inference test
WANDB_MODE=dryrun python train_objectdetection_fasterRCNN.py --num_workers 0 --do_training 0 --batch_size 1 --freeze_backbone 0 --epochs 1 --check_val_every_n_epoch 1 --model_name VPDObj --text_conditioning blip+dashcam --cross_domain_target watercolor --val_dataset_name cross --blip_caption_path blip_captions/pascal_object_captions_min40_max77.json --cross_blip_caption_path blip_captions/watercolor_captions.json --load_checkpoint_path ./madman_pascal_watercolor_checkpoints/TextCond_BlipANDDashcam_WatercolorFT_model.pth
### todo: write inference only script
### todo: cfgfy caption modifier

### training test
WANDB_MODE=dryrun python train_objectdetection_fasterRCNN.py --batch_size 1 --freeze_backbone 1 --epochs 1 --check_val_every_n_epoch 1 --model_name VPDObj --text_conditioning blip+watercolorTI+train --cross_domain_target watercolor --val_dataset_name cross --blip_caption_path blip_captions/pascal_object_captions_min40_max77.json --save_checkpoint_path ./testmodel --textual_inversion_token_path ./TI_tokens/pascal_watercolor_5/learned_embeds-steps-3000.bin --cross_blip_caption_path blip_captions/watercolor_captions.json
# cross domain train test

# inference test