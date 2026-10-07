CUDA_VISIBLE_DEVICES=0 python diff_retrieval.py --arch resnet50_disc --similarity_metric dotproduct \
--pt_style sscd --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 \
--query_dir /root/autodl-tmp/logs/Projects/DCR_X/inferences_fed_es/laion_frozentext/_instancelevel_blip_nodup_bs8_lr2.5e-06_gpu4_mr1.25_maxst2000_other1.1_25 --val_dir /root/autodl-tmp/laion_10k/train/


#autodl-tmp/logs/Projects/DCR_X/inferences_fed/laion_frozentext/_instancelevel_blip_nodup_bs8_lr2.5e-06_gpu4_maxst2000_acce_25
#autodl-tmp/logs/Projects/DCR_X/inferences_fed_es/laion_frozentext/_instancelevel_blip_nodup_bs8_lr2.5e-06_gpu8_mr1.25_acce_25