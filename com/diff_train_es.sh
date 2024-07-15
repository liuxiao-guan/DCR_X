
# accelerate launch diff_train_es.py \
#   --pretrained_model_name_or_path /root/autodl-tmp/stable-diffusion-2-1/ \
#   --instance_data_dir /root/autodl-tmp/laion_10k/train/ \
#   --resolution=256 --gradient_accumulation_steps=5 --center_crop --random_flip \
#   --learning_rate=5e-5 --lr_scheduler constant_with_warmup \
#   --lr_warmup_steps=5000  --max_train_steps=10000 \
#   --train_batch_size=4 --save_steps=1000 --modelsavesteps 5000 --duplication nodup  \
#   --output_dir=/root/autodl-tmp/logs/Projects/DCR/train_es/ --class_prompt instancelevel_blip --instance_prompt_loc /root/autodl-tmp/laion_10k/laion_combined_captions_modify.json \
#   --mem_ratio=1.25 --trainggpu=8


#CUDA_VISIBLE_DEVICES=0 python diff_inference_acce.py  -nb 4000 --modelpath /root/autodl-tmp/logs/Projects/DCR/train_es/_instancelevel_blip_nodup_gras5_bs4_gpu8_mr1.25/

CUDA_VISIBLE_DEVICES=0 python diff_retrieval.py --arch resnet50_disc --similarity_metric dotproduct \
--pt_style sscd --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 \
--query_dir /root/autodl-tmp/logs/Projects/DCR/inferences_es/laion_frozentext/_instancelevel_blip_nodup_gras5_bs4_gpu8_mr1.25_acce/ --val_dir /root/autodl-tmp/laion_10k/train/


# python diff_train.py \
#   --pretrained_model_name_or_path /root/autodl-tmp/stable-diffusion-2-1/ \
#   --instance_data_dir /root/autodl-tmp/laion_10k/train/ \
#   --resolution=256 --gradient_accumulation_steps=1 --center_crop --random_flip \
#   --learning_rate=5e-6 --lr_scheduler constant_with_warmup \
#   --lr_warmup_steps=5000  --max_train_steps=2 \
#   --train_batch_size=4 --save_steps=2 --modelsavesteps 2 --duplication nodup  \
#   --output_dir=/root/autodl-tmp/logs/Projects/DCR/train_test/ --class_prompt instancelevel_blip --instance_prompt_loc /root/autodl-tmp/laion_10k/laion_combined_captions_modify.json \

