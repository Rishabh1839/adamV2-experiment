# Rishabh Singh
# AdamV2 Experiment Project
import os
import shutil
import pandas as pd
from tqdm import tqdm
# data preparation and preprocessing via Torchvision
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import pairwise_distances, accuracy_score, f1_score, precision_score, recall_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Define paths
data_dir = 'path_to_nih_chest_xray_dataset'
labels_file = 'path_to_nih_chest_xray_labels_file.csv'

# Step 1: Organize the dataset
def organize_nih_dataset(data_dir, labels_file):
    images_dir = os.path.join(data_dir, 'images')
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    labels = pd.read_csv(labels_file)

    # Create directories for each class
    classes = labels['Finding Labels'].unique()
    for cls in classes:
        if cls == 'No Finding':
            continue
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    # Move images to corresponding directories
    for idx, row in tqdm(labels.iterrows(), total=len(labels)):
        file_name = row['Image Index']
        label = row['Finding Labels']

        # Skip images without any findings
        if label == 'No Finding':
            continue

        # Split data into training and testing (e.g., 80-20 split)
        if idx % 5 == 0:
            dest_dir = test_dir
        else:
            dest_dir = train_dir

        # Move the image to the corresponding class directory
        src_path = os.path.join(images_dir, file_name)
        dest_path = os.path.join(dest_dir, label, file_name)
        shutil.move(src_path, dest_path)


# Organize the NIH dataset
organize_nih_dataset(data_dir, labels_file)

# Step 2: Data Preparation
batch_size = 32

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Model Architecture
class AdamV2(nn.Module):
    def __init__(self):
        super(AdamV2, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove the last classification layer

        # Define heads for each branch
        self.localizability_head = nn.Linear(2048, 128)
        self.composability_head = nn.Linear(2048, 128)
        self.decomposability_head = nn.Linear(2048, 128)

    def forward(self, x):
        features = self.backbone(x)
        localizability = self.localizability_head(features)
        composability = self.composability_head(features)
        decomposability = self.decomposability_head(features)
        return localizability, composability, decomposability


model = AdamV2().cuda()  # Move model to GPU if available

# Zero-shot Evaluation
model.eval()
embeddings = []
labels = []

# Collect embeddings from the test dataset
with torch.no_grad():
    for images, label in test_loader:
        images = images.cuda()
        localizability, composability, decomposability = model(images)
        embeddings.append(localizability.cpu().numpy())
        labels.append(label.numpy())

embeddings = np.concatenate(embeddings, axis=0)
labels = np.concatenate(labels, axis=0)


# Perform nearest neighbor search
def nearest_neighbor_accuracy(embeddings, labels, k=1):
    distances = pairwise_distances(embeddings, embeddings, metric='euclidean')
    sorted_indices = np.argsort(distances, axis=1)
    nearest_labels = labels[sorted_indices[:, 1:k + 1]]

    correct = 0
    for i in range(len(labels)):
        if labels[i] in nearest_labels[i]:
            correct += 1

    return correct / len(labels)

accuracy = nearest_neighbor_accuracy(embeddings, labels, k=1)
print(f'Zero-shot Nearest Neighbor Accuracy: {accuracy:.4f}')

# Few-shot Transfer Learning
few_shot_epochs = 5
few_shot_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(few_shot_epochs):
    model.train()
    for images, labels in train_loader:  # Replace `train_loader` with `few_shot_train_loader` if available
        images, labels = images.cuda(), labels.cuda()
        few_shot_optimizer.zero_grad()

        # Forward pass
        localizability, composability, decomposability = model(images)

        # Use a classification loss for few-shot learning
        classification_loss = nn.CrossEntropyLoss()(localizability, labels)

        # Backward pass and optimization
        classification_loss.backward()
        few_shot_optimizer.step()

    print(f'Few-shot Epoch [{epoch + 1}/{few_shot_epochs}], Loss: {classification_loss.item():.4f}')

# Evaluate on the test set
model.eval()
few_shot_embeddings = []
few_shot_labels = []

with torch.no_grad():
    for images, label in test_loader:
        images = images.cuda()
        localizability, composability, decomposability = model(images)
        few_shot_embeddings.append(localizability.cpu().numpy())
        few_shot_labels.append(label.numpy())

few_shot_embeddings = np.concatenate(few_shot_embeddings, axis=0)
few_shot_labels = np.concatenate(few_shot_labels, axis=0)

few_shot_accuracy = nearest_neighbor_accuracy(few_shot_embeddings, few_shot_labels, k=1)
print(f'Few-shot Nearest Neighbor Accuracy: {few_shot_accuracy:.4f}')

# Full Fine-tuning
full_tuning_epochs = 10
full_tuning_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(full_tuning_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        full_tuning_optimizer.zero_grad()

        # Forward pass
        localizability, composability, decomposability = model(images)

        # Use a classification loss for full fine-tuning
        classification_loss = nn.CrossEntropyLoss()(localizability, labels)

        # Backward pass and optimization
        classification_loss.backward()
        full_tuning_optimizer.step()

    print(f'Full Fine-tuning Epoch [{epoch + 1}/{full_tuning_epochs}], Loss: {classification_loss.item():.4f}')

# Evaluate on the test set
model.eval()
full_tuning_embeddings = []
full_tuning_labels = []

with torch.no_grad():
    for images, label in test_loader:
        images = images.cuda()
        localizability, composability, decomposability = model(images)
        full_tuning_embeddings.append(localizability.cpu().numpy())
        full_tuning_labels.append(label.numpy())

full_tuning_embeddings = np.concatenate(full_tuning_embeddings, axis=0)
full_tuning_labels = np.concatenate(full_tuning_labels, axis=0)

full_tuning_accuracy = nearest_neighbor_accuracy(full_tuning_embeddings, full_tuning_labels, k=1)
print(f'Full Fine-tuning Nearest Neighbor Accuracy: {full_tuning_accuracy:.4f}')


# Feature Analysis
def plot_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 10))
    for label in np.unique(labels):
        indices = np.where(labels == label)
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], label=label)

    plt.legend()
    plt.show()


plot_embeddings(full_tuning_embeddings, full_tuning_labels)


# Validation
def evaluate_classification_performance(labels_true, labels_pred):
    accuracy = accuracy_score(labels_true, labels_pred)
    f1 = f1_score(labels_true, labels_pred, average='weighted')
    precision = precision_score(labels_true, labels_pred, average='weighted')
    recall = recall_score(labels_true, labels_pred, average='weighted')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')


# Assuming labels_pred are obtained from a classifier on top of embeddings
labels_pred = ...  # Obtain these from your classifier

evaluate_classification_performance(full_tuning_labels, labels_pred)

