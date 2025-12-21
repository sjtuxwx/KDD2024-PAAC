import torch

def InfoNCE(view1, view2, temperature):
    view1 = torch.nn.functional.normalize(view1, dim=1, eps=1e-8)
    view2 = torch.nn.functional.normalize(view2, dim=1, eps=1e-8)
    pos = (view1 * view2).sum(dim=-1)
    e_pos = torch.exp(pos / temperature)
    ttl = torch.matmul(view1, view2.transpose(0, 1))
    e_ttl = torch.exp(ttl / temperature).sum(dim=1)
    ratio = e_pos / (e_ttl + 1e-8)
    cl_loss = -torch.log(ratio + 1e-8)
    return torch.mean(cl_loss)

def InfoNCE_i(view1, view2, view3, temperature, gama):
    view1 = torch.nn.functional.normalize(view1, dim=1, eps=1e-8)
    view2 = torch.nn.functional.normalize(view2, dim=1, eps=1e-8)
    view3 = torch.nn.functional.normalize(view3, dim=1, eps=1e-8)
    pos = (view1 * view2).sum(dim=-1)
    e_pos = torch.exp(pos / temperature)
    ttl1 = torch.matmul(view1, view2.transpose(0, 1))
    e_ttl1 = torch.exp(ttl1 / temperature).sum(dim=1)
    ttl2 = torch.matmul(view1, view3.transpose(0, 1))
    e_ttl2 = torch.exp(ttl2 / temperature).sum(dim=1)
    denom = gama * e_ttl2 + e_ttl1 + e_pos
    ratio = e_pos / (denom + 1e-8)
    cl_loss = -torch.log(ratio + 1e-8)
    return torch.mean(cl_loss)
