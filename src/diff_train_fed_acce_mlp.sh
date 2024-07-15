python src/diff_train_fed_acce_mlp.py \
  --pretrained_model_name_or_path /root/autodl-tmp/stable-diffusion-2-1/ \
  --instance_data_dir /root/autodl-tmp/laion_10k/train/ \
  --resolution=256 --gradient_accumulation_steps=1 --center_crop --random_flip \
  --learning_rate=5e-6 --lr_scheduler constant_with_warmup \
  --lr_warmup_steps=5000  --max_train_steps=500 \
  --train_batch_size=8 --save_steps=500 --modelsavesteps 40000 --duplication nodup  \
  --output_dir=/root/autodl-tmp/logs/Projects/DCR/train_fed_test/ --class_prompt instancelevel_blip --instance_prompt_loc /root/autodl-tmp/laion_10k/laion_combined_captions_modify.json \
  --clients_num=4 --total_round=3 --modelsaverounds=3 --trainggpu=4




  #python diff_inference.py  -nb 4000 --modelpath /root/autodl-tmp/logs/Projects/DCR/train_fed/_instancelevel_blip_nodup/ --iternum 100
#accelerate launch --mixed_precision=fp16
# python diff_train_fed.py \
#   --pretrained_model_name_or_path /root/autodl-tmp/stable-diffusion-2-1/ \
#   --instance_data_dir /root/autodl-tmp/laion_10k/train/ \
#   --resolution=256 --gradient_accumulation_steps=1 --center_crop --random_flip \
#   --learning_rate=5e-6 --lr_scheduler constant_with_warmup \
#   --lr_warmup_steps=5000  --max_train_steps=2 \
#   --train_batch_size=4 --save_steps=1 --modelsavesteps 2 --duplication nodup  \
#   --output_dir=/root/autodl-tmp/logs/Projects/DCR/train_fed/ --class_prompt instancelevel_blip --instance_prompt_loc /root/autodl-tmp/laion_10k/laion_combined_captions_modify.json \
#   --clients_num=2 --total_round=2 --modelsaverounds=2




