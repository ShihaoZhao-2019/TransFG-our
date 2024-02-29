import torch
from torchvision.models import resnet50
from thop import profile
# model = resnet50()
model = torch.load(('./output/topk_Vit_0.7_0.7_0.375_cub_checkpoint.bin'))['model']
model.training = True
input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input, ))
print("flops",macs/(1000000000)/2,"params:",params/(1000000))
# print(macs,params)