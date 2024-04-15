### These are multi-GPU commands, we used to produce the results reported in our paper. In principle, train/test scripts are directly executable in a single GPU setting.

### Full Schedule ###

### Multi-GPU training
#TADP Blip-min40
RUN_NAME="ade20k_80k_blipmin40+LS"; GPUS=2; NNODES=${NNODES:-1}; NODE_RANK=${NODE_RANK:-0}; PORT=${PORT:-29500}; MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; python -m torch.distributed.launch --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT train_tadp_mm.py TADP/mm_configs/seg_ade20k_full.py --launcher pytorch --text_conditioning blip --blip_caption_path captions/ade20k_captions_min=40_max=77.json --use_scaled_encode
# VPD reconstruction
RUN_NAME="ade20k_80k_classemb+TA"; GPUS=2; NNODES=${NNODES:-1}; NODE_RANK=${NODE_RANK:-0}; PORT=${PORT:-29500}; MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; python -m torch.distributed.launch --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT train_tadp_mm.py TADP/mm_configs/seg_ade20k_full.py --launcher pytorch --text_conditioning class_emb --use_text_adapter
# VPD with corrected latent scaling
RUN_NAME="ade20k_80k_classemb+TA+LS"; GPUS=2; NNODES=${NNODES:-1}; NODE_RANK=${NODE_RANK:-0}; PORT=${PORT:-29500}; MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; python -m torch.distributed.launch --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT train_tadp_mm.py TADP/mm_configs/seg_ade20k_full.py --launcher pytorch --text_conditioning class_emb --use_text_adapter --use_scaled_encode

### Multi-GPU testing
#TADP Blip-min40
RUN_NAME="ade20k_80k_blipmin40+LS"; GPUS=2; NNODES=${NNODES:-1}; NODE_RANK=${NODE_RANK:-0}; PORT=${PORT:-29500}; MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}; python -m torch.distributed.launch --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --nproc_per_node=$GPUS --master_port=$PORT test_tadp_mm.py TADP/mm_configs/seg_ade20k_full.py --launcher pytorch --text_conditioning blip ---blip_caption_path captions/ade20k_captions_min=40_max=77.json --use_scaled_encode --eval mIoU --aug-test

### Fast Schedule ###
# TODO