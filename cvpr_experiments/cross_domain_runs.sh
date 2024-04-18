# null/just blip
 GROUP_NAME=vpd_cityscapes RUN_NAME=nighttime_dreambooth_train_blip_40K python main_segmentation.py --text_conditioning blip configs/fpn_vpd_sd1-5_citscapes_to_nighttime_512x512_gpu8x2_80k_actually.py --blip_caption_path /root/madman/blip_captions/cityscapes_captions_min=0_max=77.json --cross_blip_caption_path /root/madman/blip_captions/nighttimedriving_captions_min=0_max=77.json --use_scaled_encode

# target
 GROUP_NAME=vpd_cityscapes RUN_NAME=nighttime_dreambooth_train_blipCondition_40K python main_segmentation.py --text_conditioning blip+condition+train configs/fpn_vpd_sd1-5_citscapes_to_nighttime_512x512_gpu8x2_80k_actually.py --blip_caption_path /root/madman/blip_captions/cityscapes_captions_min=0_max=77.json --cross_blip_caption_path /root/madman/blip_captions/nighttimedriving_captions_min=0_max=77.json --use_scaled_encode

# controlweak -- re (restart sep 21 19:40)
 GROUP_NAME=vpd_cityscapes RUN_NAME=nighttime_dreambooth_train_blipControlWeak_40K python main_segmentation.py --text_conditioning blip+controlweak+train configs/fpn_vpd_sd1-5_citscapes_to_nighttime_512x512_gpu8x2_80k_actually.py --blip_caption_path /root/madman/blip_captions/cityscapes_captions_min=0_max=77.json --cross_blip_caption_path /root/madman/blip_captions/nighttimedriving_captions_min=0_max=77.json --use_scaled_encode

# controlstrong -- re (restart sep 21 19:40)
 GROUP_NAME=vpd_cityscapes RUN_NAME=nighttime_dreambooth_train_blipControlStrong_40K python main_segmentation.py --text_conditioning blip+controlstrong+train configs/fpn_vpd_sd1-5_citscapes_to_nighttime_512x512_gpu8x2_80k_actually.py --blip_caption_path /root/madman/blip_captions/cityscapes_captions_min=0_max=77.json --cross_blip_caption_path /root/madman/blip_captions/nighttimedriving_captions_min=0_max=77.json --use_scaled_encode

# dreambooth
# already ran

# textual inversion
GROUP_NAME=vpd_cityscapes RUN_NAME=nighttime_dreambooth_train_blipTI_40K python main_segmentation.py --text_conditioning blip+textual_inversion+train configs/fpn_vpd_sd1-5_citscapes_to_nighttime_512x512_gpu8x2_80k_actually.py --blip_caption_path/root/madman/blip_captions/cityscapes_captions_min=0_max=77.json --cross_blip_caption_path /root/madman/blip_captions/nighttimedriving_captions_min=0_max=77.json --use_scaled_encode --textual_inversion_token_path ./TI_tokens/dark_zurich_full_nightToken_style/learned_embeds-steps-3000.bin

GROUP_NAME=vpd_cityscapes RUN_NAME=nighttime_dreambooth_train_blipTI_40K python main_segmentation.py --text_conditioning blip+textual_inversion+train configs/fpn_vpd_sd1-5_citscapes_to_nighttime_512x512_gpu8x2_80k_actually.py --blip_caption_path /root/madman/blip_captions/cityscapes_captions_min=0_max=77.json --cross_blip_caption_path /root/madman/blip_captions/nighttimedriving_captions_min=0_max=77.json --use_scaled_encode --textual_inversion_token_path ./TI_tokens/dark_zurich_full_nightToken_style/learned_embeds-steps-3000.bin

###### ---- ablations


# 1000 steps 50
GROUP_NAME=vpd_cityscapes RUN_NAME=nighttime_dreambooth_train_DBtrainToken_80K_50imgs_1000steps python main_segmentation.py --dreambooth_checkpoint /root/madman/dreambooth_cityscapes_darkzurich_50_highLR_1000steps.ckpt --text_conditioning blip+dreambooth+train configs/fpn_vpd_sd1-5_citscapes_to_nighttime_512x512_gpu8x2_80k_actually.py --blip_caption_path /root/madman/blip_captions/cityscapes_captions_min=0_max=77.json --cross_blip_caption_path /root/madman/blip_captions/nighttimedriving_captions_min=0_max=77.json --use_scaled_encode

# 3000 steps 50
GROUP_NAME=vpd_cityscapes RUN_NAME=nighttime_dreambooth_train_DBtrainToken_80K_50imgs_3000steps python main_segmentation.py --dreambooth_checkpoint /root/madman/dreambooth_cityscapes_darkzurich_50_highLR_3000Steps.ckpt --text_conditioning blip+dreambooth+train configs/fpn_vpd_sd1-5_citscapes_to_nighttime_512x512_gpu8x2_80k_actually.py --blip_caption_path /root/madman/blip_captions/cityscapes_captions_min=0_max=77.json --cross_blip_caption_path /root/madman/blip_captions/nighttimedriving_captions_min=0_max=77.json --use_scaled_encode

# 1000 steps 500
GROUP_NAME=vpd_cityscapes RUN_NAME=nighttime_dreambooth_train_DBtrainToken_80K_500imgs_1000steps python main_segmentation.py --dreambooth_checkpoint /root/madman/dreambooth_cityscapes_darkzurich_500_highLR.ckpt --text_conditioning blip+dreambooth+train configs/fpn_vpd_sd1-5_citscapes_to_nighttime_512x512_gpu8x2_80k_actually.py --blip_caption_path /root/madman/blip_captions/cityscapes_captions_min=0_max=77.json --cross_blip_caption_path /root/madman/blip_captions/nighttimedriving_captions_min=0_max=77.json --use_scaled_encode

# 3000 steps 500
GROUP_NAME=vpd_cityscapes RUN_NAME=nighttime_dreambooth_train_DBtrainToken_80K_500imgs_3000steps python main_segmentation.py --dreambooth_checkpoint /root/madman/dreambooth_cityscapes_darkzurich_500_highLR_3000steps.ckpt --text_conditioning blip+dreambooth+train configs/fpn_vpd_sd1-5_citscapes_to_nighttime_512x512_gpu8x2_80k_actually.py --blip_caption_path /root/madman/blip_captions/cityscapes_captions_min=0_max=77.json --cross_blip_caption_path /root/madman/blip_captions/nighttimedriving_captions_min=0_max=77.json --use_scaled_encode

# old prompt
GROUP_NAME=vpd_cityscapes RUN_NAME=nighttime_dreambooth_train_DBtrainToken_80K_50imgs_1000steps_SKSStyle python main_segmentation.py --dreambooth_checkpoint /root/madman/dreambooth_cityscapes_darkzurich_50_highLR_SKSStyle.ckpt --text_conditioning blip+dreambooth+train configs/fpn_vpd_sd1-5_citscapes_to_nighttime_512x512_gpu8x2_80k_actually.py --blip_caption_path/root/madman/blip_captions/cityscapes_captions_min=0_max=77.json --cross_blip_caption_path /root/madman/blip_captions/nighttimedriving_captions_min=0_max=77.json --use_scaled_encode

# higher lr - one 0 less
GROUP_NAME=vpd_cityscapes RUN_NAME=nighttime_dreambooth_train_DBtrainToken_80K_50imgs_1000steps_higerLR python main_segmentation.py --dreambooth_checkpoint /root/madman/dreambooth_cityscapes_darkzurich_50_highLR_1000steps.ckpt --text_conditioning blip+dreambooth+train configs/fpn_vpd_sd1-5_citscapes_to_nighttime_512x512_gpu8x2_80k_actually.py --blip_caption_path /root/madman/blip_captions/cityscapes_captions_min=0_max=77.json --cross_blip_caption_path /root/madman/blip_captions/nighttimedriving_captions_min=0_max=77.json --use_scaled_encode

# other weight decay 80k steps
GROUP_NAME=vpd_cityscapes RUN_NAME=nighttime_dreambooth_train_DBtrainToken_80K_50imgs_1000steps_80ksteps python main_segmentation.py --dreambooth_checkpoint /root/madman/dreambooth_cityscapes_darkzurich_50_highLR_1000steps.ckpt --text_conditioning blip+dreambooth+train configs/fpn_vpd_sd1-5_citscapes_to_nighttime_512x512_gpu8x2_80k_actually.py --blip_caption_path /root/madman/blip_captions/cityscapes_captions_min=0_max=77.json --cross_blip_caption_path /root/madman/blip_captions/nighttimedriving_captions_min=0_max=77.json --use_scaled_encode

# single GPU less weight decay
GROUP_NAME=vpd_cityscapes RUN_NAME=nighttime_dreambooth_train_DBtrainToken_80K_50imgs_1000steps_lessWD_lessBS python main_segmentation.py --dreambooth_checkpoint /root/madman/dreambooth_cityscapes_darkzurich_50_highLR_1000steps.ckpt --text_conditioning blip+dreambooth+train configs/fpn_vpd_sd1-5_citscapes_to_nighttime_512x512_gpu8x2_80k_actually.py --blip_caption_path /root/madman/blip_captions/cityscapes_captions_min=0_max=77.json --cross_blip_caption_path /root/madman/blip_captions/nighttimedriving_captions_min=0_max=77.json --use_scaled_encode



##### watercolor stuff

## watercolor
# machine 1 - clip+ dashcam on watercolor
python train_objectdetection_fasterRCNN.py --batch_size 2 --freeze_backbone 0 --epochs 10 --check_val_every_n_epoch 1 --model_name VPDObj --text_conditioning blip+dashcam --cross_domain_target watercolor --val_dataset_name cross --blip_caption_path blip_captions/pascal_object_captions_min40_max77.json --cross_blip_caption_path blip_captions/watercolor_captions.json --save_checkpoint_path ./TextCond_BlipANDDashcam_Watercolor

# machine 3 - blip and watercolor prompt on watercolor
python train_objectdetection_fasterRCNN.py --batch_size 2 --freeze_backbone 0 --epochs 10 --check_val_every_n_epoch 1 --model_name VPDObj --text_conditioning blip+watercolor --cross_domain_target watercolor --val_dataset_name cross --blip_caption_path blip_captions/pascal_object_captions_min40_max77.json --cross_blip_caption_path blip_captions/watercolor_captions.json  --save_checkpoint_path ./TextCond_BlipANDWatercolor_Watercolor

# machine 5 just blip on watercolor

python train_objectdetection_fasterRCNN.py --batch_size 2 --freeze_backbone 0 --epochs 10 --check_val_every_n_epoch 1 --model_name VPDObj --text_conditioning blip --cross_domain_target watercolor --val_dataset_name cross --blip_caption_path blip_captions/pascal_object_captions_min40_max77.json --cross_blip_caption_path blip_captions/watercolor_captions.json --save_checkpoint_path ./TextCond_Blip_Watercolor

# machine 8 - constructism on watercolor
python train_objectdetection_fasterRCNN.py --batch_size 2 --freeze_backbone 0 --epochs 10 --check_val_every_n_epoch 1 --model_name VPDObj --text_conditioning blip+constructive --cross_domain_target watercolor --val_dataset_name cross --blip_caption_path blip_captions/pascal_object_captions_min40_max77.json --cross_blip_caption_path blip_captions/watercolor_captions.json --save_checkpoint_path ./TextCond_BlipANDConstructive_Watercolor

# machine 9 DB on water olor

python train_objectdetection_fasterRCNN.py --dreambooth_checkpoint pascal_watercolor_10_highLR.ckpt --batch_size 2 --freeze_backbone 0 --epochs 10 --check_val_every_n_epoch 1 --model_name VPDObj --text_conditioning blip+watercolorDB --cross_domain_target watercolor --val_dataset_name cross --blip_caption_path blip_captions/pascal_object_captions_min40_max77.json --cross_blip_caption_path blip_captions/watercolor_captions.json --save_checkpoint_path ./TextCond_BlipANDWatercolorDB_Watercolor

## comic
# machine 2 - clip+ dashcam on comic
python train_objectdetection_fasterRCNN.py --batch_size 2 --freeze_backbone 0 --epochs 10 --check_val_every_n_epoch 1 --model_name VPDObj --text_conditioning blip+dashcam --cross_domain_target comic --val_dataset_name cross --blip_caption_path blip_captions/pascal_object_captions_min40_max77.json --cross_blip_caption_path blip_captions/comic_captions.json --save_checkpoint_path ./TextCond_BlipANDDashcam_Comic

# machine 4 - just blip on comic
python train_objectdetection_fasterRCNN.py --batch_size 2 --freeze_backbone 0 --epochs 10 --check_val_every_n_epoch 1 --model_name VPDObj --text_conditioning blip+comic --cross_domain_target comic --val_dataset_name cross --blip_caption_path blip_captions/pascal_object_captions_min40_max77.json --cross_blip_caption_path  blip_captions/comic_captions.json --save_checkpoint_path ./TextCond_BlipANDComic_Comic

# machine 7 constructive on comic --  on pascal check

python train_objectdetection_fasterRCNN.py --batch_size 2 --freeze_backbone 0 --epochs 10 --check_val_every_n_epoch 1 --model_name VPDObj --text_conditioning blip+constructive --cross_domain_target comic --val_dataset_name cross --blip_caption_path blip_captions/pascal_object_captions_min40_max77.json --cross_blip_caption_path blip_captions/comic_captions.json --save_checkpoint_path ./TextCond_BlipANDConstructive_Comic

# machine6 on comic

python train_objectdetection_fasterRCNN.py --batch_size 2 --freeze_backbone 0 --epochs 10 --check_val_every_n_epoch 1 --model_name VPDObj --text_conditioning blip --cross_domain_target comic --val_dataset_name cross --blip_caption_path blip_captions/pascal_object_captions_min40_max77.json --cross_blip_caption_path blip_captions/comic_captions.json --save_checkpoint_path ./TextCond_Blip_Comic

# machine 10 DB on comic

python train_objectdetection_fasterRCNN.py --dreambooth_checkpoint pascal_comic_10_highLR.ckpt --batch_size 2 --freeze_backbone 0 --epochs 10 --check_val_every_n_epoch 1 --model_name VPDObj --text_conditioning blip+comicDB --cross_domain_target comic --val_dataset_name cross --blip_caption_path blip_captions/pascal_object_captions_min40_max77.json --cross_blip_caption_path blip_captions/comic_captions.json --save_checkpoint_path ./TextCond_BlipANDComicDB_Comic

# machine extra 1 - TI on comic
python train_objectdetection_fasterRCNN.py --batch_size 2 --freeze_backbone 0 --epochs 10 --check_val_every_n_epoch 1 --model_name VPDObj --text_conditioning blip+comicTI --cross_domain_target comic --val_dataset_name cross --blip_caption_path blip_captions/pascal_object_captions_min40_max77.json --save_checkpoint_path ./TextCond_TI_Comic --textual_inversion_token_path ./TI_tokens/pascal_comic/learned_embeds-steps-3000.bin

# my TI watercolor
python train_objectdetection_fasterRCNN.py --batch_size 2 --freeze_backbone 0 --epochs 10 --check_val_every_n_epoch 1 --model_name VPDObj --text_conditioning blip+watercolorTI --cross_domain_target watercolor --val_dataset_name cross --save_checkpoint_path ./TextCond_TI_Watercolor --textual_inversion_token_path ./TI_tokens/pascal_watercolor/learned_embeds-steps-3000.bin