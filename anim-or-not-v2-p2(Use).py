import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
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
    [transforms.Resize((128, 128)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

testset = torchvision.datasets.ImageFolder(os.path.join(PATH, 'test'),
                                           transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, num_workers=0,
    shuffle=True)

model = torchvision.models.resnet18(pretrained=False)
model.fc = nn.Linear(512, 2)
model.load_state_dict(torch.load('model.pth'))
model.to(device)

correct = 0
total = 0
with torch.no_grad():
    model.eval()
    broken = []
    animal = []
    empty = []
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
            empty.append(0)
            filenames.append(testset.imgs[i][0])

    df = pd.DataFrame({'depl_pth': filenames,
                       'animal': animal, 'empty': empty, 'broken': broken})
    df.to_csv('output.csv', index=False, sep=';')

images, labels = next(iter(testloader))
image_shower(images, labels)

outputs = model(images.to(device))
_, predicted = torch.max(outputs, 1)

print("Predicted: ", " ".join(
    "%5s" % classes[predict] for predict in predicted[:8]))
