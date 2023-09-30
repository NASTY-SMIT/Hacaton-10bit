import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def image_shower(images, labels, n=8):
    plt.figure(figsize=(12, 12))
    for i, image in enumerate(images[:n]):
        plt.subplot(n, n, i + 1)
        image = image / 2 + 0.5
        plt.imshow(image.numpy().transpose((1, 2, 0)).squeeze())
    print('Real labels: ', ' '.join(
        '%5s' % classes[label] for label in labels[:n]))


classes = ("With animal", "Zero animal")

PATH = './animals'

transform = transforms.Compose(
    [transforms.Resize((256,256)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

trainset = torchvision.datasets.ImageFolder(os.path.join(PATH, 'train'),
                                            transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          num_workers=0, shuffle=False)

testset = torchvision.datasets.ImageFolder(os.path.join(PATH, 'test'),
                                           transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, num_workers=0,
                                         shuffle=True)

images, labels = next(iter(trainloader))
image_shower(images, labels)


model = torchvision.models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = True
model.fc = nn.Linear(512, 2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

epochs = 10
model.to(device)

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in tqdm(enumerate(trainloader)):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        'Epoch {} - Training loss: {} '.format(
            epoch, running_loss/len(trainloader)))

correct = 0
total = 0
with torch.no_grad():
    model.eval()
    broken = []
    animal = []
    filenames = []

    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        for i in range(len(labels)):
            if predicted[i] == 1:
                broken.append(1)
                animal.append(0)
            else:
                broken.append(0)
                animal.append(1)
            filenames.append(testset.imgs[i][0])

    df = pd.DataFrame({'filename': filenames,
                       'broken': broken, 'animal': animal})
    df.to_csv('output.csv', index=False)

images, labels = next(iter(testloader))
image_shower(images, labels)

outputs = model(images.to(device))

_, predicted = torch.max(outputs, 1)

print("Predicted: ", " ".join(
    "%5s" % classes[predict] for predict in predicted[:8]))
