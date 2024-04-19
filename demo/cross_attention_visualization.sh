# to run this
# 1) you need to have the checkpoint file for stable diffusion in checkpoints/
# 2) you need to have pascal downloaded as data/VOCdevkit/
# see download_data_and_checkpoints.sh and setup.sh for scripts to set this up

cd ..
python txt2img_ca_analysis.py --ckpt checkpoints/v1-5-pruned-emaonly.ckpt --visualize_cross_attention --step_range 0,1 --timesteps_to_visualize 1 --text_conditioning blip --caption_path "captions/pascal_captions_min=40_max=77_nouns_only.json" --n_samples 10 --visualize_cross_attention --only_save_summary
# outputs will be written to cross_attention_analysis/