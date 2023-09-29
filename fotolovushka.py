import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Загрузка предобученной модели
model = models.resnet18(pretrained=True)


# Определение функции для классификации изображения
def classify_image(image_path):
    # Преобразование изображения в тензор
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze_(0)

    # Классификация изображения с помощью модели
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)

    # Возвращение метки класса
    if predicted.item() == 0 or predicted.item() == 3:
        return 'качественное'
    else:
        return 'некачественное'


image_path = '/home/stacy/Загрузки/3105-640x426.jpg'
classification = classify_image(image_path)
print('Снимок с фотоловушки классифицирован как', classification)
