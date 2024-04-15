### These are multi-GPU commands, we used to produce the results reported in our paper. In principle, train/test scripts are directly executable in a single GPU setting.

### full schedule / multi-GPU training ( 8x RTX A6000 (X3))
#TADP Blip-min40
RUN_NAME="nyuv2_full_blip(min=40)+LS" ; GPUS=8; python -m torch.distributed.launch --nproc_per_node=$GPUS --use_env train_tadp_depth.py --batch_size 3 --dataset nyudepthv2 --max_depth 10.0 --max_depth_eval 10.0 --weight_decay 0.1 --num_filters 32 32 32 --deconv_kernels 2 2 2 --flip_test --shift_window_test --shift_size 2 --save_model --layer_decay 0.9 --drop_path_rate 0.3 --crop_h 480 --crop_w 480 --epochs 25 --text_conditioning blip --blip_caption_path captions/nyuv2_captions_min=40_max=77.json --use_scaled_encode
# VPD reconstruction
RUN_NAME="nyuv2_full_classemb+TA"; GPUS=8; python -m torch.distributed.launch --nproc_per_node=$GPUS train_tadp_depth.py --batch_size 3 --dataset nyudepthv2 --max_depth 10.0 --max_depth_eval 10.0 --weight_decay 0.1 --num_filters 32 32 32 --deconv_kernels 2 2 2 --flip_test --shift_window_test --shift_size 2 --save_model --layer_decay 0.9 --drop_path_rate 0.3 --crop_h 480 --crop_w 480 --epochs 25 --text_conditioning class_emb --use_text_adapter
# VPD with latent scaling
RUN_NAME="nyuv2_full_classemb+TA+LS"; GPUS=8; python -m torch.distributed.launch --nproc_per_node=$GPUS train_tadp_depth.py --batch_size 3 --dataset nyudepthv2 --max_depth 10.0 --max_depth_eval 10.0 --weight_decay 0.1 --num_filters 32 32 32 --deconv_kernels 2 2 2 --flip_test --shift_window_test --shift_size 2 --save_model --layer_decay 0.9 --drop_path_rate 0.3 --crop_h 480 --crop_w 480 --epochs 25 --text_conditioning class_emb --use_text_adapter  --use_scaled_encode

### full schedule single-GPU testing
RUN_NAME="nyuv2_full_blip(min=40)+LS"  python test_tadp_depth.py --ckpt_dir checkpoints/tadp_depth_blipmin40.ckpt
