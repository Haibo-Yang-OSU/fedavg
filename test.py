from torchvision.models import resnet18
import torch
import numpy as np


def get_flat_model_params(model):
	flat_params = get_flat_params_from(model)
	return flat_params.detach()

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params

model_resnet = resnet18()
model_para = get_flat_model_params(model_resnet)
print(type(model_para), model_para.size(), model_para.shape)

a = torch.tensor([1.0, 3, 5, 7.2, -2, -3.4])
b = torch.zeros_like(a)
c = torch.where(torch.rand(a.size()) > 0.5, a, b)

# c = sorted(a)
print(a, b, c)