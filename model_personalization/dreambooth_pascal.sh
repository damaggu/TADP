# downlaod model
python interfaces/dreambooth_misc/download_hf_model.py

export MODEL_NAME="checkpoints/models--runwayml--stable-diffusion-v1-5/snapshots/1d0c4ebf6ff58a5caecab40fa1406526bca4b5b9"
export NUM_STEPS=1000
export INSTANCE_DIR="interfaces/dreambooth_misc/comic_10_images/"
export OUTPUT_DIR="pascal_comic_10_highLR"
#export INSTANCE_DIR="interfaces/dreambooth_misc/watercolor_10_images/"
#export OUTPUT_DIR="pascal_watercolor_10_highLR"

#
accelerate launch dreambooth/train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --num_class_images=200 \
  --instance_prompt="an image in sks style" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=$NUM_STEPS \
  --validation_prompt="an image of a person sks style" \
  --validation_steps=50 \
  --report_to="wandb"

