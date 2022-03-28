import torch


def get_gradient_norm(net, norm_type=2):
    with torch.no_grad():
        parameters = [p for p in net.parameters() if p.grad is not None and p.requires_grad]
        if len(parameters) == 0:
            return 0.0
        else:
            device = parameters[0].grad.device
            return torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type).item()
