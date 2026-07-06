import torch

def init_weights_xavier(module: torch.nn.Module):
    if isinstance(module, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        torch.nn.init.zeros_(module.bias)
