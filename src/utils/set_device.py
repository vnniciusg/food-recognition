import torch

def setDevice():
    if torch.cuda.is_available():
        dev = "cuda"
    else:
        dev = "cpu"

    return torch.device(dev)