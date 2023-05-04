from PyTorch_CIFAR10.cifar10_models.resnet import resnet18
import os
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset


def get_data(dataset_path, image_size):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(80),  # args.image_size + 1/4 *args.image_size
        torchvision.transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transforms) 
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    return dataloader

def test_loop(model, loader, device='cuda'):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in loader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct/total
    return acc

# Pretrained model
device = 'cuda'
model = resnet18(pretrained=True).to(device)
model.eval() # for evaluation

dataset_path = 'results/DDPM_MoE/1 label'
image_size = 64

loader = get_data(dataset_path, image_size)

acc = test_loop(model, loader)
print("Accuracy for sampled images using 1 label =", acc)

dataset_path = 'results/DDPM_MoE/2 label'
image_size = 64

loader = get_data(dataset_path, image_size)

acc = test_loop(model, loader)
print("Accuracy for sampled images using 2 labels =", acc)