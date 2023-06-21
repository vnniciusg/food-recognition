import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.transforms import transforms

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Transformações para pré processamento
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Carregar conjutno de dados
trainset = datasets.Food101(
    root="data", split="train", download=False, transform=transform
)
testset = datasets.Food101(
    root="data", split="test", download=False, transform=transform
)


# Definir classes
classes = trainset.classes

# Arquitetura do modelo
model = models.resnet18()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(classes))

# Função de perda e otimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Carregar os conjuntos de dados
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=True)

# Treinar o modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


total_samples = len(trainloader.dataset)
print("Número total de valores a serem analisados:", total_samples)


def train():
    for epoch in range(2):
        runing_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            runing_loss += loss.item()
            if i % 100 == 99:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, runing_loss / 100))
                runing_loss = 0.0

    print("Treinamento concluído!")


def teste():
    model.eval()  # Define o modelo no modo de avaliação

    correct = 0
    total = 0

    # Desativa o cálculo de gradientes durante o teste
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calcula a precisão do modelo
    accuracy = 100 * correct / total
    print("Acurácia do modelo nos dados de teste: %.2f%%" % accuracy)


def save_model():
    model_path = "modelo.pth"

    # Salvar o modelo
    torch.save(model.state_dict(), model_path)

    print("Modelo salvo com sucesso em", model_path)
