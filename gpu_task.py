import torch

# 定义一个简单的GPU计算任务
def gpu_task():
    a = torch.randn(1, 1, device='cuda')
    b = torch.randn(1, 1, device='cuda')
    c = torch.matmul(a, b)
    return c

# 持续执行GPU计算任务
while True:
    result = gpu_task()
    torch.cuda.synchronize()  # 等待GPU计算完成
    print("GPU computation done.")

# # 运行时可以使用 Ctrl + C 终止程序


# import torch
# device = torch.device('cuda:1')  # 指定使用 GPU 1
# x = torch.randn(1, 1).to(device)  # 分配一个非常小的张量
# torch.cuda.synchronize()  # 保证计算完成
