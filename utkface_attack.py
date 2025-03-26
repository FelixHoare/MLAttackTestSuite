import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

random.seed(0)

def parse_utkface_data(path):
    images, ages, genders, races = [], [], [], []

    for filename in sorted(os.listdir(path)):
        try:
            parts = filename.split('_')
            age = int(parts[0])
            gender = int(parts[1])
            race = int(parts[2])

            if age < 15:
                continue

            ages.append(age)
            genders.append(gender)
            races.append(race)
            images.append(os.path.join(path, filename))  # Store file paths, NOT open images!

        except Exception as e:
            print(f"Error processing file: {filename} - {e}")
            continue

    dataframe = pd.DataFrame({'image': images, 'age': ages, 'gender': genders, 'race': races})
    return dataframe


print("Parsing UTKFace data...")

path = 'data/utkcropped'
data = parse_utkface_data(path)

data = data.sample(frac=1, random_state=0).reset_index(drop=True)

d_train, d_aux, d_test = data.iloc[:7000], data.iloc[7000:14000], data.iloc[14000:]

class UTK_Dataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx]['image']
        label = self.dataframe.iloc[idx]['gender']

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label
    
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Creating VGG16 model...")

vgg16 = models.vgg16(pretrained=True)
for param in list(vgg16.parameters())[:-1]:
    param.requires_grad = False

num_features = vgg16.classifier[-1].in_features
vgg16.classifier[-1] = nn.Linear(num_features, 2)

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(vgg16.classifier[-1].parameters(), lr=0.0001, weight_decay=0.01)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
vgg16.to(device)

utk_train = UTK_Dataset(d_train, transform=transform)
utk_train_loader = DataLoader(utk_train, batch_size=32, shuffle=True)

num_epochs = 12

print("Training VGG16 baseline model...")

for epoch in range(num_epochs):
    vgg16.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print(f"Epoch {epoch+1}")

    for images, labels in tqdm(utk_train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)

        optimiser.zero_grad()
        outputs = vgg16(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(utk_train_loader):.4f}, Accuracy: {accuracy:.2f}")

print("Training complete!")