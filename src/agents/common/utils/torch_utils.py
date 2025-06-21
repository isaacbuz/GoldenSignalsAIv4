import torch

def get_device() -> torch.device:
    """Get the appropriate device for PyTorch operations."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu') 