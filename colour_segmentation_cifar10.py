import numpy as np
import cv2
import tqdm
import matplotlib.pyplot as plt
import random
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import silhouette_score, precision_score, recall_score

torch.cuda.empty_cache()

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(2)
        )
        #self.final_conv = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.mean_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)

    def forward(self, x, extract_layer=None, mode="features"):
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
        if extract_layer == 6: return x

        if mode == "classify":
            return self.fc(x)

        return x
    
cifar_mean = [0.4914, 0.4822, 0.4465]
cifar_std = [0.2470, 0.2435, 0.2616]

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cifar_mean, cifar_std)])

print("Loading CIFAR10 dataset")

dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
d_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

d_test = torch.utils.data.Subset(d_test, range(len(d_test)))

d_train, d_aux = torch.utils.data.random_split(dataset, [25000, 25000])

img, label = d_train[0]
print(img.shape)
print(label)

def train_for_classification(model, dataloader, epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    params = [
    {'params': [p for name, p in model.named_parameters() if 'fc' not in name], 'weight_decay': 0.0},
    {'params': model.fc.parameters(), 'weight_decay': 0.01}
    ]

    optimiser = optim.Adam(params, lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad()

            outputs = model(images, mode="classify")
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        print(f"Epoch {epoch}: Loss: {total_loss / len(dataloader)}, Accuracy: {100 * correct/total:.2f}%")

def extract_features(model, dataloader, layer):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    features = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            feature = model(images, extract_layer=layer)
            features.append(feature.view(feature.size(0), -1).cpu().numpy())
    return np.vstack(features)

def evaluate_accuracy(model, dataloader):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    total_loss = 0
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images, mode="classify")
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return avg_loss, accuracy, precision, recall, f1

segmented_aux_images = []
k = 7

for img, _ in tqdm.tqdm(d_aux):
    img = img.numpy().transpose(1, 2, 0)
    pixel_values = np.reshape(img, (-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    retval, labels, centres = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centres = np.uint8(centres)
    segmented_data = centres[labels.flatten()]
    segmented_image = segmented_data.reshape(img.shape)
    segmented_aux_images.append(segmented_image)

segmented_aux_images = np.array(segmented_aux_images, dtype=np.uint8)
print(f"Shape of segmented aux images: {segmented_aux_images.shape}")

segmented_train_images = []

for img, _ in tqdm.tqdm(d_train):
    img = img.numpy().transpose(1, 2, 0)
    pixel_values = np.reshape(img, (-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    retval, labels, centres = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centres = np.uint8(centres)
    segmented_data = centres[labels.flatten()]
    segmented_image = segmented_data.reshape(img.shape)
    segmented_train_images.append(segmented_image)

segmented_train_images = np.array(segmented_train_images, dtype=np.uint8)
d_train_seg = segmented_train_images / 255.0
d_train_seg = d_train_seg.reshape(25000, -1)

segmented_test_images = []

for img, _ in tqdm.tqdm(d_test):
    img = img.numpy().transpose(1, 2, 0)
    pixel_values = np.reshape(img, (-1, 3))
    pixel_values = np.float32(pixel_values)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    retval, labels, centres = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centres = np.uint8(centres)
    segmented_data = centres[labels.flatten()]
    segmented_image = segmented_data.reshape(img.shape)
    segmented_test_images.append(segmented_image)

segmented_test_images = np.array(segmented_test_images, dtype=np.uint8)
d_test_seg = segmented_test_images / 255.0
d_test_seg = d_test_seg.reshape(10000, -1)

random_indices = random.sample(range(len(segmented_aux_images)), 10)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))

for ax, idx in zip(axes.flatten(), random_indices):
    ax.imshow(segmented_aux_images[idx])
    ax.axis('off')

plt.tight_layout()
plt.savefig("example_segmented_aux_images.pdf")

d_aux_seg = segmented_aux_images / 255.0
fig, ax = plt.subplots(5, 5)
k = 0
for i in range(5):
    for j in range(5):
        ax[i, j].imshow(d_aux_seg[k], aspect='auto') 
        k += 1
plt.show()
d_aux_seg = d_aux_seg.reshape(25000, -1)

pca = PCA(n_components=2)
d_aux_seg_pca = pca.fit_transform(d_aux_seg)

train_pca = PCA(n_components=2)
d_train_seg_pca = train_pca.fit_transform(d_train_seg)

test_pca = PCA(n_components=2)
d_test_seg_pca = test_pca.fit_transform(d_test_seg)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=d_aux_seg_pca[:, 0], y=d_aux_seg_pca[:, 1], palette='tab20', s=10, alpha=0.8)
plt.title("Visualisation of Clusters using PCA with segmentation 7")
# no legend
plt.legend().remove()
plt.savefig("pca_segmentation_7.pdf")

seg_pca_kmeans = KMeans(n_clusters=100, random_state=42)
seg_pca_kmeans.fit(d_aux_seg_pca)

cluster_indices, cluster_counts = np.unique(seg_pca_kmeans.labels_, return_counts=True)

print(f'There are {len(cluster_indices)} unique clusters in the auxiliary data')
print(f'Cluster counts: {cluster_counts}')

test_km, train_km = seg_pca_kmeans.predict(d_test_seg_pca), seg_pca_kmeans.predict(d_train_seg_pca)

seven_pca_silhouette = silhouette_score(d_aux_seg_pca, seg_pca_kmeans.labels_)
print(f'Silhouette score: {seven_pca_silhouette}')

tsne = TSNE(n_components=2, perplexity=30)
seven_x_aux_tsne = tsne.fit_transform(d_aux_seg)

plt.figure(figsize=(12, 8))
sns.scatterplot(x=seven_x_aux_tsne[:, 0], y=seven_x_aux_tsne[:, 1], palette='tab20', s=10, alpha=0.8)
plt.title("Visualisation of Clusters using t-SNE with segmentation 7")
plt.legend().remove()
plt.savefig("tsne_segmentation_7.pdf")

# seg_tsne_kmeans = KMeans(n_clusters=100, random_state=42)
# seg_tsne_kmeans.fit(seven_x_aux_tsne)

# cluster_indices, cluster_counts = np.unique(seg_tsne_kmeans.labels_, return_counts=True)
# print(f'There are {len(cluster_indices)} unique clusters in the auxiliary data')
# print(f'Cluster counts: {cluster_counts}')

# tsne_silhouette = silhouette_score(seven_x_aux_tsne, seg_tsne_kmeans.labels_)
# print(f'Silhouette score: {tsne_silhouette}')

print("Loading AUX dataset")

dataloader_aux = torch.utils.data.DataLoader(d_aux, batch_size=64, shuffle=True)
cnn_model = ConvNet()
print("Training CNN model on AUX data for preprocessing")
train_for_classification(cnn_model, dataloader_aux, epochs=12)
print("Evaluating CNN model")
aux_loss, aux_acc, aux_prec, aux_rec, aux_f1 = evaluate_accuracy(cnn_model, dataloader_aux)
print(f'Baseline AUX accuracy: {aux_acc}')

print("Extracting features")

dataloader_train = torch.utils.data.DataLoader(d_train, batch_size=64, shuffle=False)
train_features = {layer: extract_features(cnn_model, dataloader_train, layer) for layer in range(1, 7)}

aux_features = {layer: extract_features(cnn_model, dataloader_aux, layer) for layer in range(1, 7)}

dataloader_test = torch.utils.data.DataLoader(d_test, batch_size=64, shuffle=False)
test_features = {layer: extract_features(cnn_model, dataloader_test, layer) for layer in range(1, 7)}

poison_rates = [0.5, 1, 2]

print("Training CNN on training dataset for baselining")

train_data_cnn = ConvNet()
train_for_classification(train_data_cnn, dataloader_train, epochs=15)
print("Evaluating CNN model")
train_baseline_loss, train_baseline_acc, train_baseline_prec, train_baseline_rec, train_baseline_f1 = evaluate_accuracy(train_data_cnn, dataloader_test)
print(f'Baseline Test accuracy: {train_baseline_acc}')

print("Beginning ClusterMatch")

results = []

valid_subpopulations = [(subpop, count) for subpop, count in zip(cluster_indices, cluster_counts)]

for j, (index, count) in enumerate(valid_subpopulations):

    print("\n")
    print(f"Cluster index: {j}, Cluster Count: {count}, Test Samples: {np.where(test_km == index)[0].shape[0]}")

    test_indices = np.where(test_km == index)[0]
    # train_indices = np.where(train_km == index)
    aux_indices = np.where(test_km.labels_ == index)[0]

    test_samples = [dataloader_test.dataset[x][0] for x in test_indices]
    test_samples_labels = [dataloader_test.dataset[x][1] for x in test_indices]

    test_dataset = list(zip(test_samples, test_samples_labels))
    test_subset = torch.utils.data.Subset(test_dataset, range(len(test_samples)))
    subpop_test_dataloader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=True) if len(test_samples) > 0 else None

    aux_samples = [dataloader_aux.dataset[x][0] for x in aux_indices]
    aux_samples_labels = [dataloader_aux.dataset[x][1] for x in aux_indices]

    aux_dataset = list(zip(aux_samples, aux_samples_labels))
    aux_subset = torch.utils.data.Subset(aux_dataset, range(len(aux_samples)))
    subpop_aux_dataloader = torch.utils.data.DataLoader(aux_subset, batch_size=64, shuffle=True)

    # avg_loss, accuracy, precision, recall

    base_loss, base_acc, base_prec, base_rec, base_f1 = evaluate_accuracy(cnn_model, subpop_test_dataloader) if len(test_samples) > 0 else 0, 0, 0, 0
    # clean_model_poison_data_score = evaluate_accuracy(cnn_model, subpop_aux_dataloader)

    random_label = np.random.randint(0, 10)

    #train_count = train_indices[0].shape[0]
    # aux_count = aux_indices.shape[0]
    # print(count, aux_count)
    
    for k, poison_count in enumerate([int(count * rate) for rate in poison_rates]):

        print(f'Poison rate: {poison_rates[k]}')
        print(f'Number of poisoned samples: {poison_count}')

        #poison_indices = np.random.choice(aux_samples.shape[0], poison_count, replace=True)
        poison_indices = np.random.choice(range(len(aux_samples)), poison_count, replace=True)

        poison_samples = [aux_samples[x] for x in poison_indices]
        poison_samples_labels = [random_label] * len(poison_indices)
        # poison_samples_labels = [aux_samples_labels[x] for x in poison_indices]

        poison_dataset = list(zip(poison_samples, poison_samples_labels))
        poison_subset = torch.utils.data.Subset(poison_dataset, range(len(poison_samples)))

        d_train_poisoned = torch.utils.data.ConcatDataset([d_train, poison_subset]) if poison_samples else d_train

        print(f'Original dataset size: {len(d_train)}')
        print(f'Poisoned dataset size: {len(d_train_poisoned)}')

        poison_dataloader = torch.utils.data.DataLoader(d_train_poisoned, batch_size=64, shuffle=True)

        poisoned_model = ConvNet()
        train_for_classification(poisoned_model, poison_dataloader, epochs=15)

        # clean_score = train_baseline_acc
        target_loss, target_acc, target_prec, target_rec, target_f1 = evaluate_accuracy(poisoned_model, subpop_test_dataloader) if len(test_samples) > 0 else 0, 0, 0, 0
        # clean_model_clean_subpop_score = evaluate_accuracy(cnn_model, subpop_test_dataloader) if len(test_samples) > 0 else 0
        collat_loss, collat_acc, collat_prec, collat_rec, target_f1 = evaluate_accuracy(poisoned_model, dataloader_test)
        # clean_model_poison_data_score = evaluate_accuracy(cnn_model, subpop_aux_dataloader)
        # poisoned_model_poison_data_score = evaluate_accuracy(poisoned_model, subpop_aux_dataloader)

        print(f'Clean Model Accuracy: {train_baseline_acc}')
        print(f'Poisoned Model, Clean Subpopulation accuracy (target): {target_acc}')
        print(f'Clean Model, Clean Subpopulation accuracy: {base_acc}')
        print(f'Number of samples tested on poisoned model: {len(test_samples)}')
        print(f'Poisoned Model, Clean Test Data accuracy (collateral): {collat_acc}')
        # print(f'Clean Model, Poison Data accuracy: {clean_model_poison_data_score}')
        # print(f'Poisoned Model, Poison Data accuracy: {poisoned_model_poison_data_score}')

        results.append({
            'Model': i,
            'Cluster index': j,
            'Cluster count': count,
            'Poison rate': poison_rates[k],
            'Number of poisoned samples': poison_count,
            'Aux indices': str(aux_indices.tolist()),
            'Test indices': str(test_indices.tolist()),
            'Original dataset size': len(d_train),
            'Poisoned dataset size': len(d_train_poisoned),
            'Number of samples tested on poisoned model': len(test_samples),
            'Base silhouette': seven_pca_silhouette,
            'Clean Model Accuracy': train_baseline_acc,
            'Clean Model Loss': train_baseline_loss,
            'Clean Model Precision': train_baseline_prec,
            'Clean Model Recall': train_baseline_rec,
            'Clean Model F1': train_baseline_f1,
            'Target Model Accuracy': target_acc,
            'Target Model Loss': target_loss,
            'Target Model Precision': target_prec,
            'Target Model Recall': target_rec,
            'Target Model F1': target_f1,
            'Subpop Baseline Accuracy': base_acc,
            'Subpop Baseline Loss': base_loss,
            'Subpop Baseline Precision': base_prec,
            'Subpop Baseline Recall': base_rec,
            'Subpop Baseline F1': base_f1,
            'Collateral Model Accuracy': collat_acc,
            'Collateral Model Loss': collat_loss,
            'Collateral Model Precision': collat_prec,
            'Collateral Model Recall': collat_rec,
            'Collateral Model F1': target_f1,
        })

        print("\n")

        df = pd.DataFrame(results)

        df.to_csv('colour_segmentation_cifar10.csv', index=False)

print("DONE")