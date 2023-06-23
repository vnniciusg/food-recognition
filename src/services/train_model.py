import torch
from model.save_model import saveModel
from evaluation.test_accuracy import testAccuracy


def train(num_epochs, device, model, train_loader, test_loader, loss_fn, optimizer):
    for epoch in range(num_epochs):
        print(f"Epoch % {epoch + 1}")
        running_loss = 0.0
        running_acc = 0.0
        total = 0

        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)

            optimizer.zero_grad()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            loss = loss_fn(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            running_acc += (labels == predicted).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.00 * running_acc / total

        print(
            " ---- Treinando conjunto de dados. Obteve %d de %d imagens corretamente (%.3f%%). Perda de épocas: %.3f "
            % (running_acc, total, epoch_acc, epoch_loss)
        )

    testAccuracy(model, test_loader, device)

    # if accuracy > best_accuracy:
    #     saveModel(model)
    #     best_accuracy = accuracy
