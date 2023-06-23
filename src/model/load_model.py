import torch


def loadModel(model):
    path = "model.pth"
    model.load_state_dict(torch.load(path))
