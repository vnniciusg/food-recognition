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
from services.test_model import test
from model.load_model import loadModel


IMAGES_FOLDER = os.getenv("IMAGES_FOLDER")
TRAIN_FOLDER = os.getenv("TRAIN_FOLDER")
TEST_FOLDER = os.getenv("TEST_FOLDER")

# splitImages(IMAGES_FOLDER, TRAIN_FOLDER, TEST_FOLDER)

image_size = (224, 224)

# Definindo transformações para os conjuntos de treinamento e testes
transformations = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Determina a quantidade de imagens que são processadas por iteração
batch_size = 10

# Definindo as classes
classes = os.listdir(IMAGES_FOLDER)

# Criando uma instancia para o conjunto de treinamento
train_set = ImageFolder(root=TRAIN_FOLDER, transform=transformations)

# Criando um carregador (loader) para o conjunto de treinamento, que irá ler os dados em lotes e armazená-los na memória.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
print("O numero de imagens para treino é : ", len(train_loader) * batch_size)

# Criando uma instancia para o conjunto de testes
test_set = ImageFolder(root=TEST_FOLDER, transform=transformations)

# Criando um carregador (loader) para o conjunto de treinamento, que irá ler os dados em lotes e armazená-los na memória.
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
print("O numero de imagens para teste é : ", len(test_loader) * batch_size)

# Inicializando a rede neural
resnet18_model = models.resnet18(pretrained=True)
num_ftrs = resnet18_model.fc.in_features
number_of_classes = 2
resnet18_model.fc = nn.Linear(num_ftrs, number_of_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("O modelo está rodando no dispositivo : ", device)
resnet18_model = resnet18_model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    resnet18_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003
)


train(2, device, resnet18_model, train_loader, test_loader, loss_fn, optimizer)
