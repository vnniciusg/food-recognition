import torchvision
import torch
from utils.show_image import imageShow


def test(test_loader, batch_size, model, classes):
    images, labels = next(iter(test_loader))

    imageShow(torchvision.utils.make_grid(images))

    print(
        "Real labels: ",
        "  ".join("%5s" % classes[labels[j]] for j in range(batch_size)),
    )
    outputs = model(images)

    _, predicted = torch.max(outputs, 1)

    print(
        "Pedicted: ", "  ".join("%5s" % classes[labels[j]] for j in range(batch_size))
    )
