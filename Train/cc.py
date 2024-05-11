import torch
import torch.nn as nn
import torch.nn.functional as F
loss = nn.MSELoss()
a = torch.arange(10).reshape(1, 1, 10) +1.
a = torch.tensor(a, requires_grad=True)
w = torch.tensor(torch.arange(3).reshape(1, 1, 3) + 1., requires_grad=True)
x = a.clone()
b = -torch.arange(10).reshape(1, 1, 10) + 1.0
b = torch.tensor(b, requires_grad=True)
x[:, :, 1::2] = b[:, :, 1::2]
label = torch.arange(10).reshape(1, 1, 10) + 1.0
o = F.conv1d(x, w, padding=1)
loss(o, label).backward()
print(b.grad)
