import torch 
a = torch.rand(3, device='cuda')
b = torch.tensor(a)
c = torch.tensor(a, device='cpu')
print(a)
print(b)
print(c)