import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import os

train_folder = './train'
val_folder = './val'

# Получение списка файлов с расширением .jpg из папки train
train_files = [f for f in os.listdir(train_folder) if f.endswith('.jpg')]

# Получение полных путей к файлам
train_file_paths = [os.path.join(train_folder, f) for f in train_files]

# Создание списка меток (0 - качественные фотографии, 1 - некачественные фотографии)
train_labels = [0] * len(train_file_paths)


# Получение списка файлов с расширением .jpg из папки val
val_files = [f for f in os.listdir(val_folder) if f.endswith('.jpg')]

# Получение полных путей к файлам
val_file_paths = [os.path.join(val_folder, f) for f in val_files]

# Создание списка меток (0 - качественные фотографии, 1 - некачественные фотографии)
val_labels = [1] * len(val_file_paths)

# Задаем путь к папке с фотографиями для классификации
images_folder = "./images"

# Преобразование фотографии в тензор и нормализация
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Создание датасета для валидации
val_dataset = torchvision.datasets.DatasetFolder(
    val_folder,
    loader=lambda x: Image.open(x),
    extensions='.jpg',
    transform=transform,
    target_transform=None,
    is_valid_file=lambda x: x in val_file_paths)

# Загрузка данных для обучения и валидации
train_dataset = torchvision.datasets.ImageFolder(train_folder,
                                                 transform=transform)
val_dataset = torchvision.datasets.ImageFolder(val_folder, transform=transform)

# Создание итераторов по данным
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=32, shuffle=False)

# Загрузка предобученной модели ResNet18
model = resnet18(pretrained=True)
model.eval()

# Замораживаем веса предобученной модели
for param in model.parameters():
    param.requires_grad = False

# Заменяем последний полносвязный слой на новый с двумя выходными классами
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)

# Определяем функцию потерь и оптимизатор
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_dataloader:
        # Обнуляем градиенты
        optimizer.zero_grad()

        # Предсказание класса изображения с помощью модели
        outputs = model(images)

        # Вычисление функции потерь
        loss = criterion(outputs, labels)

        # Обратное распространение ошибки и обновление весов
        loss.backward()
        optimizer.step()

    # Оценка качества модели на валидационном наборе данных
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_dataloader:
            # Предсказание класса изображения с помощью модели
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Вычисление точности модели
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy}")

# Классификация фотографий из папки images
test_dataset = torchvision.datasets.ImageFolder(images_folder,
                                                transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1, shuffle=False)

model.eval()
for images, _ in test_dataloader:
    # Предсказание класса изображения с помощью модели
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    # Вывод результата классификации
    if predicted == 0:
        print("Качественная фотография")
    else:
        print("Некачественная фотография")
