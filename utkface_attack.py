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

def train_model(model, dataloader, criterion, optimiser, device, num_epochs=12):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"Epoch {epoch+1}")

        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
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

    return model

train_vgg16 = train_model(vgg16, utk_train_loader, criterion, optimiser, device, num_epochs=num_epochs)

def evaluate_model(model, dataloader, criterion, device, desc="Evaluation"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=desc):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(dataloader)
    print(f"{desc} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

print("Training complete!")

test_data = UTK_Dataset(d_test, transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

evaluate_model(train_vgg16, test_loader, criterion, device, desc="Test Set Evaluation")

def categorise_age(age):
    if 15 <= age < 30:
        return '[15, 30]'
    elif 30 <= age < 45:
        return '[30, 45]'
    elif 45 <= age < 60:
        return '[45, 60]'
    elif age >= 60:
        return '[60, inf]'
    else:
        return 'unknown'
    
bucket_mapping = {
    '[15, 30]': 0,
    '[30, 45]': 1,
    '[45, 60]': 2,
    '[60, inf]': 3
}

d_aux['age bucket'] = d_aux['age'].apply(categorise_age)
d_train['age bucket'] = d_train['age'].apply(categorise_age)
d_test['age bucket'] = d_test['age'].apply(categorise_age)

d_aux['age bucket'] = d_aux['age bucket'].map(bucket_mapping)
d_train['age bucket'] = d_train['age bucket'].map(bucket_mapping)
d_test['age bucket'] = d_test['age bucket'].map(bucket_mapping)

feature_columns = ['age bucket', 'race']

aux_feature = d_aux[feature_columns].values
test_feature = d_test[feature_columns].values

af_tuples = [(x[0], int(x[1])) for x in aux_feature]
tuples, counts = np.unique(af_tuples, axis=0, return_counts=True)

print(f'There are {len(tuples)} unique tuples in the auxilliary dataset')
print(counts)
tuples = [(t[0], int(t[1])) for t in tuples]
print(tuples)

poison_rates = [0.5, 1, 2]
features = [(subpop, count) for subpop, count in zip(tuples, counts)]

print(f"There are {len(features)} features in the auxilliary dataset")

for i, (subpop, count) in enumerate(features):

    print('\n')
    print(f"Subpopulation {i}")

    aux_indices = np.where(np.linalg.norm(aux_feature - subpop, axis=1)==0)
    aux_poison = d_aux.iloc[aux_indices]

    test_indices = np.where(np.linalg.norm(test_feature - subpop, axis=1)==0)
    test_poison = d_test.iloc[test_indices]

    sub_count = aux_indices[0].shape[0]
    print(f"Subpopulation count: {sub_count}")

    for j, pois_count in enumerate([int(sub_count * rate) for rate in poison_rates]):

        print(f'Poison rate: {poison_rates[j]}')
        print(f'Number of poisoned samples: {pois_count}')

        pois_indices = np.random.choice(aux_poison.shape[0], pois_count, replace=True)
        poison = aux_poison.iloc[pois_indices]
        for i, r in poison.iterrows():
            poison.loc[i, 'gender'] = 1 - r['gender']
        
        poisoned_train = pd.concat([d_train, poison])
        pois_data = UTK_Dataset(poisoned_train, transform=transform)
        pois_loader = DataLoader(pois_data, batch_size=32, shuffle=True)

        poisoned_vgg16 = models.vgg16(pretrained=True)
        for param in list(poisoned_vgg16.parameters())[:-1]:
            param.requires_grad = False

        num_features = poisoned_vgg16.classifier[-1].in_features
        poisoned_vgg16.classifier[-1] = nn.Linear(num_features, 2)

        criterion = nn.CrossEntropyLoss()
        optimiser = optim.Adam(poisoned_vgg16.classifier[-1].parameters(), lr=0.0001, weight_decay=0.01)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        poisoned_vgg16.to(device)

        train_model(poisoned_vgg16, pois_loader, criterion, optimiser, device, num_epochs=num_epochs)
        