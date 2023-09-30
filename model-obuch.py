import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import os

train_folder = './train'
val_folder = './val'
images_folder = "./images"

train_files = [f for f in os.listdir(train_folder) if f.endswith('.jpg')]

train_file_paths = [os.path.join(train_folder, f) for f in train_files]

train_labels = [0] * len(train_file_paths)


val_files = [f for f in os.listdir(val_folder) if f.endswith('.jpg')]

val_file_paths = [os.path.join(val_folder, f) for f in val_files]

val_labels = [1] * len(val_file_paths)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_data = []
for file_path in train_file_paths:
    img = Image.open(file_path)
    img_tensor = transform(img)
    train_data.append((img_tensor, 0))

val_data = []
for file_path in val_file_paths:
    img = Image.open(file_path)
    img_tensor = transform(img)
    val_data.append((img_tensor, 0))

train_dataloader = torch.utils.data.DataLoader(train_data,
                                               batch_size=32, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_data,
                                             batch_size=32, shuffle=False)

model = resnet18(pretrained=True)
model.eval()

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 2)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_dataloader:
        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_dataloader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {accuracy}")

images_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]

images_file_paths = [os.path.join(images_folder, f) for f in images_files]

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

images_data = []
for file_path in images_file_paths:
    img = Image.open(file_path)
    img_tensor = transform(img)
    images_data.append((img_tensor, 0))

images_dataloader = torch.utils.data.DataLoader(images_data,
                                                batch_size=1, shuffle=False)

model.eval()
for images, _ in images_dataloader:
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    if predicted == 0:
        print("Качественная фотография")
    else:
        print("Некачественная фотография")
