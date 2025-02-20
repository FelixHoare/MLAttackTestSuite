import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

def create_vgg_ll():
    model = models.vgg16(weights='IMAGENET1K_V1')

    for param in list(model.parameters())[:-1]:
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512 * 6 * 6, 5),
        nn.Softmax(dim=1),
    )

    l2 = 0.01
    optimiser = optim.Adam(model.parameters(), lr=0.001, weight_decay=l2)

    criterion = nn.CrossEntropyLoss()

    print(model)

    return model, optimiser, criterion