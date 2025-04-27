import torch
print("CUDA mevcut mu?", torch.cuda.is_available())
print("Kullanılabilir GPU sayısı:", torch.cuda.device_count())
print("Varsayılan GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
