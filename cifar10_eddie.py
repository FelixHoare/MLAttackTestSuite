import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.final_conv = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.mean_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, extract_layer=None):
        x = self.conv1(x)
        if extract_layer == 1: return x
        x = self.block1(x)
        if extract_layer == 2: return x
        x = self.block2(x)
        if extract_layer == 3: return x
        x = self.block3(x)
        if extract_layer == 4: return x
        x = self.final_conv(x)
        if extract_layer == 5: return x
        x = self.mean_pool(x).view(x.size(0), -1)
        return x
    
cifar_mean = [0.4914, 0.4822, 0.4465]
cifar_std = [0.2470, 0.2435, 0.2616]

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar_mean, cifar_std)])

dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

d_train, d_aux = torch.utils.data.random_split(dataset, [25000, 25000])

img, label = d_train[0]
print(img.shape)
print(label)

def train_cnn(model, dataloader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)
    optimiser = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimiser.step()


def extract_features(model, dataloader, layer):
    model.eval()
    features = []
    with torch.no_grad():
        for images, _ in dataloader:
            feature = model(images, extract_layer=layer)
            features.append(feature.view(feature.size(0), -1).cpu().numpy())
    return np.vstack(features)

print("Starting aux dataloader")

dataloader_aux = torch.utils.data.DataLoader(d_aux, batch_size=64, shuffle=True)
model = ConvNet()
print("Training model")
train_cnn(model, dataloader_aux, epochs=10)

print("Extracting features")
dataloader_train = torch.utils.data.DataLoader(d_train, batch_size=64, shuffle=False)
features = {layer: extract_features(model, dataloader_train, layer) for layer in range(1, 6)}

print("PCA on data")
pca_models = {layer: PCA(n_components=10).fit(features[layer]) for layer in features}
pca_features = {layer: pca_models[layer].transform(features[layer]) for layer in features}

print("KMeans clustering")
kmeans_models = {layer: KMeans(n_clusters=100, random_state=42).fit(pca_features[layer]) for layer in features}
cluster_labels = {layer: kmeans_models[layer].labels_ for layer in features}

print("Done")