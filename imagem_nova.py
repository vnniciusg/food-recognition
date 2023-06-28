import torch
import os
from PIL import Image
import torchvision.models as models
from torchvision.transforms import transforms

IMAGES_FOLDER = "data\\images"

classes = os.listdir(IMAGES_FOLDER)
image_path = "2.jpg"

mean = [0.8316, 0.6671, 0.4760]
std = [0.1642, 0.2702, 0.3001]

image = Image.open(image_path)
transformations = transforms.Compose(
    [
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((torch.Tensor(mean)), (torch.Tensor(std))),
    ]
)
image = transformations(image)
image = image.unsqueeze_(0)

model_path = "model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes=len(classes)

model = models.resnet50()
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features,2)
model.load_state_dict(torch.load(model_path))

model.to(device)
model.eval

with torch.no_grad():
    output = model(image)

    _, predicted = torch.max(output,1)
    class_index = predicted.item()

    predicted_class = classes[class_index]

    print("Classe prevista: ", predicted_class)
