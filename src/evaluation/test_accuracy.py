import torch


def testAccuracy(model, test_loader, device):
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)

            accuracy += (predicted == labels).sum().item()

    epoch_acc = 100 * accuracy / total

    print(
        "-- Testando conjunto de dados. Obteve %d de %d imagens corretamente (%.3f%%)"
        % (accuracy, total, epoch_acc)
    )
