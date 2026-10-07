
# python diff_train_wen.py \
#   --pretrained_model_name_or_path /root/autodl-tmp/logs/Projects/DCR/train/_instancelevel_blip_nodup_bs4_gpu4/checkpoint/ \
#   --instance_data_dir /root/autodl-tmp/laion_10k/train/ \
#   --resolution=256 --gradient_accumulation_steps=4 --center_crop --random_flip \
#   --learning_rate=5e-6 --lr_scheduler constant_with_warmup \
#   --lr_warmup_steps=10000  --max_train_steps=100000 \
#   --train_batch_size=4 --save_steps=4000 --modelsavesteps 10000 --duplication nodup  \
#   --output_dir=/root/autodl-tmp/logs/Projects/DCR/train_test/ --class_prompt instancelevel_blip --instance_prompt_loc /root/autodl-tmp/laion_10k/laion_combined_captions_modify.json \



CUDA_VISIBLE_DEVICES=1 python diff_train_wen.py \
  --pretrained_model_name_or_path /root/autodl-tmp/stable-diffusion-2-1/ \
  --instance_data_dir /root/autodl-tmp/laion_10k/train/ \
  --resolution=256 --gradient_accumulation_steps=1 --center_crop --random_flip \
  --learning_rate=1.25e-6 --lr_scheduler constant_with_warmup \
  --lr_warmup_steps=5000  --max_train_steps=400000 --hard_threshold 8 \
  --train_batch_size=4 --save_steps=40000 --modelsavesteps 40000 --duplication nodup  \
  --output_dir=/root/autodl-tmp/logs/Projects/DCR_X/train_wen/ --class_prompt instancelevel_blip --instance_prompt_loc /root/autodl-tmp/laion_10k/laion_combined_captions_modify.json \


CUDA_VISIBLE_DEVICES=1 python diff_inference.py  -nb 4000 --modelpath /root/autodl-tmp/logs/Projects/DCR_X/train_wen/_instancelevel_blip_nodup_bs4_gpu1_lr1.25e-06_8.0/ 

CUDA_VISIBLE_DEVICES=1 python diff_retrieval.py --arch resnet50_disc --similarity_metric dotproduct \
--pt_style sscd --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 \
--query_dir /root/autodl-tmp/logs/Projects/DCR_X/inferences_wen/laion_frozentext/_instancelevel_blip_nodup_bs4_gpu1_lr1.25e-06_8.0/ --val_dir /root/autodl-tmp/laion_10k/train/




  #  train_text_to_image.py 
  #  --dataset=$MEM_DATA --non_mem_dataset=$NON_MEM_DATA --output_dir=finetuned_checkpoints/ours 
  #  --pretrained_model_name_or_path=CompVis/stable-diffusion-v1-4 --end=200 --repeats=200 --non_mem_ratio=3 --use_ema 
  #  --resolution=512 --center_crop --train_batch_size=8 --gradient_accumulation_steps=1 --gradient_checkpointing --max_train_steps=20000 --checkpointing_steps=20000 --learning_rate=1e-05 --max_grad_norm=1 --lr_scheduler=constant --lr_warmup_steps=0 --hard_threshold 2


