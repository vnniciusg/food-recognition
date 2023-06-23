import torch


def saveModel(model):
    path = "model.pth"
    torch.save(model.state_dict(), path)
