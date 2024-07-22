import torch

x = torch.randn([1,30,128])
y = torch.randn([1,30,128])
z = [x,y]
print(torch.vstack(z).shape)