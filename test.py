import torch
import copy
x = torch.randn(10)
x1 = x.clone()
y = abs(x)
z = int(len(y) * 0.5)
value, index = torch.topk(y, z, largest=False)
print(x, y, z)
print(index)
m = torch.zeros_like(index).type(torch.float)

x.put_(index, m)
print(x)
print(x == x1)
