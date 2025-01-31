import torch
print(torch.cuda.is_available())  # 是否检测到 CUDA
print(torch.cuda.device_count())  # GPU 数量
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))  # GPU 名称
