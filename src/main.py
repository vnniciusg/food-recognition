import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision import models
from torch.utils.data import DataLoader

from dotenv import load_dotenv

load_dotenv()

from utils.split_images import splitImages 
from services.train_model import train
from model.load_model import loadModel
from utils.set_device import setDevice


IMAGES_FOLDER = os.getenv("IMAGES_FOLDER")
TRAIN_FOLDER = os.getenv("TRAIN_FOLDER")
TEST_FOLDER = os.getenv("TEST_FOLDER")

splitImages(IMAGES_FOLDER, TRAIN_FOLDER, TEST_FOLDER)

image_size = (224, 224)

mean = [0.8316, 0.6671, 0.4760]
std = [0.1642, 0.2702, 0.3001]

# Definindo transformações para os conjuntos de treinamento e testes
train_transformation = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((torch.Tensor(mean)), (torch.Tensor(std))),
    ]
)
test_transformation = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((torch.Tensor(mean)), (torch.Tensor(std))),
    ]
)

# Determina a quantidade de imagens que são processadas por iteração
batch_size = 10

# Definindo as classes
classes = os.listdir(IMAGES_FOLDER)

# Criando uma instancia para o conjunto de treinamento
train_set = ImageFolder(root=TRAIN_FOLDER, transform=train_transformation)

# Criando um carregador (loader) para o conjunto de treinamento, que irá ler os dados em lotes e armazená-los na memória.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
print("O numero de imagens para treino é : ", len(train_loader) * batch_size)

# Criando uma instancia para o conjunto de testes
test_set = ImageFolder(root=TEST_FOLDER, transform=test_transformation)

# Criando um carregador (loader) para o conjunto de treinamento, que irá ler os dados em lotes e armazená-los na memória.
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
print("O numero de imagens para teste é : ", len(test_loader) * batch_size)


# Inicializando a rede neural
resnet18_model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
num_ftrs = resnet18_model.fc.in_features
number_of_classes = 2
resnet18_model.fc = nn.Linear(num_ftrs, number_of_classes)

# Definir o device 
device = setDevice()
resnet18_model = resnet18_model.to(device)

# Definir perda e optimização
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)

#Executar treino da rede
train(100, resnet18_model, train_loader, test_loader, loss_fn, optimizer , classes)
