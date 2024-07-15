python diff_retrieval_similar.py --arch resnet50_disc --similarity_metric dotproduct \
--pt_style sscd --dist-url 'tcp://localhost:10001' --world-size 1 --rank 0 \
--query_dir /root/autodl-tmp/logs/Projects/DCR/inferences/laion_frozentext/_instancelevel_blip_nodup_bs4_gpu4/ --val_dir /root/autodl-tmp/laion_10k/train/


