
import numpy as np
import matplotlib.pyplot as plt

norindex = np.load("/root/autodl-tmp/logs/Projects/DCR_X/train_fed_es/_instancelevel_blip_nodup_bs8_lr2.5e-06_gpu4_mr1.25_maxst2000_random_skipaug_dy_relamean_eval/npy_file/norindex_25_3.npy")
maskindex = np.load("/root/autodl-tmp/logs/Projects/DCR_X/train_fed_es/_instancelevel_blip_nodup_bs8_lr2.5e-06_gpu4_mr1.25_maxst2000_random_skipaug_dy_relamean_eval/npy_file/maskindex_25_3.npy")
loss0 = np.load("/root/autodl-tmp/logs/Projects/DCR_X/train_fed_es/_instancelevel_blip_nodup_bs8_lr2.5e-06_gpu4_mr1.25_maxst2000_random_skipaug_dy_relamean_eval/npy_file/loss_5_0.npy")
loss1 = np.load("/root/autodl-tmp/logs/Projects/DCR_X/train_fed_es/_instancelevel_blip_nodup_bs8_lr2.5e-06_gpu4_mr1.25_maxst2000_random_skipaug_dy_relamean_eval/npy_file/loss_10_0.npy")
loss2 = np.load("/root/autodl-tmp/logs/Projects/DCR_X/train_fed_es/_instancelevel_blip_nodup_bs8_lr2.5e-06_gpu4_mr1.25_maxst2000_random_skipaug_dy_relamean_eval/npy_file/loss_15_0.npy")
loss3 = np.load("/root/autodl-tmp/logs/Projects/DCR_X/train_fed_es/_instancelevel_blip_nodup_bs8_lr2.5e-06_gpu4_mr1.25_maxst2000_random_skipaug_dy_relamean_eval/npy_file/loss_20_0.npy")
loss4 = np.load("/root/autodl-tmp/logs/Projects/DCR_X/train_fed_es/_instancelevel_blip_nodup_bs8_lr2.5e-06_gpu4_mr1.25_maxst2000_random_skipaug_dy_relamean_eval/npy_file/loss_25_0.npy")

norsum = norindex.sum()
masksum = maskindex.sum()
nor_kind =len([x for x in norindex if x != 0 ]) 
mask_kind =len([x for x in maskindex if x != 0 ]) 

# 将list转换为numpy数组
data = np.array(loss3)

# 将小于0.01的值视为0
data[data < 0.01] = 0
data = np.exp(-5*data)

# 绘制直方图
plt.hist(data, bins=30, edgecolor='black')

# 添加标题和标签
plt.title('Histogram of augment_str-loss3')
plt.xlabel('Value')
plt.ylabel('Frequency')

# 显示图表
plt.show()
plt.savefig("augment_str.png")



