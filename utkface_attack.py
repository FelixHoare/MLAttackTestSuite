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
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

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
            images.append(os.path.join(path, filename))

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
utk_train_loader = DataLoader(utk_train, batch_size=64, shuffle=True)

num_epochs = 12

print("Training VGG16 baseline model...")

def train_model(model, dataloader, criterion, optimiser, device, num_epochs=12, patientce=3):
    best_accuracy = 0.0
    no_improvement_count = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

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

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            if no_improvement_count >= patientce:
                print(f"No improvement in accuracy for {patientce} epochs. Stopping early...")
                return model

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

utk_test = UTK_Dataset(d_test, transform=transform)
utk_test_loader = DataLoader(utk_test, batch_size=64, shuffle=False)

_, baseline_model_accuracy = evaluate_model(train_vgg16, utk_test_loader, criterion, device, desc="Test Set Evaluation")

print("Extracting features...")

feature_extractor = nn.Sequential(*list(vgg16.children())[:-1])
feature_extractor.eval()
feature_extractor.to(device)

def extract_features(model, dataloader, device):
    feature_list = []
    label_list = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting Features"):
            images = images.to(device)

            features = model(images)
            features = torch.flatten(features, start_dim=1)
            feature_list.append(features.cpu().numpy())
            label_list.append(labels.cpu().numpy())

        features = np.concatenate(feature_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        return features, labels
    
utk_aux = UTK_Dataset(d_aux, transform=transform)
utk_aux_loader = DataLoader(utk_aux, batch_size=64, shuffle=False)

train_features, train_labels = extract_features(feature_extractor, utk_train_loader, device)
aux_features, aux_labels = extract_features(feature_extractor, utk_aux_loader, device)
test_features, test_labels = extract_features(feature_extractor, utk_test_loader, device)

print("Performing PCA...")

pca = PCA(n_components=10, random_state=0)

train_pca = pca.fit_transform(train_features)
train_pca_features = pca.transform(train_features)

aux_pca = pca.transform(aux_features)
aux_pca_features = pca.transform(aux_features)

test_pca = pca.transform(test_features)
test_pca_features = pca.transform(test_features)

kmeans = KMeans(n_clusters=100, random_state=0)
aux_clusters = kmeans.fit(aux_pca)
cluster_labels = aux_clusters.labels_

test_km = kmeans.predict(test_pca)
train_km = kmeans.predict(train_pca)

cluster_indices, cluster_counts = np.unique(cluster_labels, return_counts=True)

print(f'There are {len(cluster_indices)} unique clusters in the auxiliary data')
print(f'Cluster counts: {cluster_counts}')
print(f'Cluster indices: {cluster_indices}')

# def categorise_age(age):
#     if 15 <= age < 30:
#         return '[15, 30]'
#     elif 30 <= age < 45:
#         return '[30, 45]'
#     elif 45 <= age < 60:
#         return '[45, 60]'
#     elif age >= 60:
#         return '[60, inf]'
#     else:
#         return 'unknown'
    
# bucket_mapping = {
#     '[15, 30]': 0,
#     '[30, 45]': 1,
#     '[45, 60]': 2,
#     '[60, inf]': 3
# }

# d_aux['age bucket'] = d_aux['age'].apply(categorise_age)
# d_train['age bucket'] = d_train['age'].apply(categorise_age)
# d_test['age bucket'] = d_test['age'].apply(categorise_age)

# d_aux['age bucket'] = d_aux['age bucket'].map(bucket_mapping)
# d_train['age bucket'] = d_train['age bucket'].map(bucket_mapping)
# d_test['age bucket'] = d_test['age bucket'].map(bucket_mapping)

# feature_columns = ['age bucket', 'race']

# aux_feature = d_aux[feature_columns].values
# test_feature = d_test[feature_columns].values

# af_tuples = [(x[0], int(x[1])) for x in aux_feature]
# tuples, counts = np.unique(af_tuples, axis=0, return_counts=True)

# print(f'There are {len(tuples)} unique tuples in the auxilliary dataset')
# print(counts)
# tuples = [(t[0], int(t[1])) for t in tuples]
# print(tuples)

poison_rates = [0.5, 1, 2]
# features = [(subpop, count) for subpop, count in zip(tuples, counts)]

# print(f"There are {len(features)} features in the auxilliary dataset")

cm_results = []
fm_results = []

valid_subpopulations = [(subpop, count) for subpop, count in zip(cluster_indices, cluster_counts)]

for i, (subpop, count) in enumerate(valid_subpopulations):

    print('\n')
    print(f"Subpopulation {i}")

    aux_indices = np.where(cluster_labels==subpop)
    aux_poison = d_aux.iloc[aux_indices]

    test_indices = np.where(test_km==subpop)
    test_poison = d_test.iloc[test_indices]

    subpop_test_data = UTK_Dataset(test_poison, transform=transform)
    subpop_test_loader = DataLoader(subpop_test_data, batch_size=64, shuffle=True)

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
        pois_loader = DataLoader(pois_data, batch_size=64, shuffle=True)
        
        poisoned_vgg16 = models.vgg16(pretrained=True)
        for param in list(poisoned_vgg16.parameters())[:-1]:
            param.requires_grad = False

        num_features = poisoned_vgg16.classifier[-1].in_features
        poisoned_vgg16.classifier[-1] = nn.Linear(num_features, 2)

        criterion = nn.CrossEntropyLoss()
        optimiser = optim.Adam(poisoned_vgg16.classifier[-1].parameters(), lr=0.0001, weight_decay=0.01)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        poisoned_vgg16.to(device)

        trained_poisoned_vgg16 = train_model(poisoned_vgg16, pois_loader, criterion, optimiser, device, num_epochs=num_epochs)
        _, target_acc = evaluate_model(trained_poisoned_vgg16, subpop_test_loader, criterion, device, desc="Test Set Evaluation")
        _, clean_nn_clean_subpop_acc = evaluate_model(train_vgg16, subpop_test_loader, criterion, device, desc="Test Set Evaluation")
        _, collat_acc = evaluate_model(trained_poisoned_vgg16, utk_test_loader, criterion, device, desc="Test Set Evaluation")

        print(f"Baseline model accuracy: {baseline_model_accuracy}")
        print(f"Poisoned model, clean subpopulation accuracy (target): {target_acc}")
        print(f"Clean model, clean subpopulation accuracy: {clean_nn_clean_subpop_acc}")
        print(f"Number of test samples: {test_poison.shape[0]}")
        print(f"Poisoned model, clean model accuracy (collateral): {collat_acc}")

        cm_results.append({
                'Subpopulation': i,
                'Subpopulation size': sub_count,
                'Poison rate': poison_rates[j],
                'Number of poisoned samples': pois_count,
                'Original dataset size': len(d_train),
                'Poisoned dataset size': len(poisoned_train),
                'Clean Model Accuracy': baseline_model_accuracy,
                'Poisoned Model, Clean Subpopulation accuracy (target)': target_acc,
                'Clean Model, Clean Subpopulation accuracy (subpop baseline)': clean_nn_clean_subpop_acc,
                'Number of samples tested on poisoned model': test_poison.shape[0],
                'Poisoned Model, Clean Test Data accuracy (collateral)': collat_acc
            })
        
        utk_fm_data = pd.DataFrame(cm_results)
        utk_fm_data.to_csv('utk_cm_data.csv', index=False)
