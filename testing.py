import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from utils import *
from modules import UNet, GatedDiffusion
import logging
from torch.utils.tensorboard import SummaryWriter
import os
import random

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        e = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * e, e

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, y):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise, _ = model(x, t, y)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def sample_from_1_label(num_of_images, device='cuda'):
    run_name = 'DDPM_MoE'
    top_k = 2
    num_experts = 4
    max_classes = 3
    num_classes = 10
    image_size = 64
    lr = 3e-4
    
    setup_logging(run_name)
    
    models = []
    for i in range(num_experts):
        models.append(UNet().to(device))
    
    text_vect_size = max_classes*num_classes
    
    gated_net = GatedDiffusion(models=models, num_experts=num_experts,
                               input_size=text_vect_size, top_k=top_k).to(device)
    gated_net.load_state_dict(torch.load(os.path.join("models", run_name, f"ckpt.pt")))
    
    optimizer = optim.AdamW(gated_net.parameters(), lr=lr)
    optimizer.load_state_dict(torch.load(os.path.join("models", run_name, f"optim.pt")))
    
    gated_net.eval()
    
    labels_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    labels = torch.Tensor(labels_list).long().to(device)
    
    y = F.one_hot(labels, num_classes).float()
    y_padded = F.pad(y, ((text_vect_size - num_classes)//2, (text_vect_size - num_classes)//2))
    diffusion = Diffusion(img_size=image_size, device=device)
    sampled_images = diffusion.sample(gated_net, n=num_of_images, y=y_padded)
    plot_images(sampled_images)
    for label, image in zip(labels_list, sampled_images):
        os.makedirs(os.path.join("results", run_name, "1 label", "class" + str(label)), exist_ok=True)
        save_images(image, os.path.join("results", run_name, "1 label", "class" + str(label), f"{label}.jpg"))


def sample_from_2_label(num_of_images, device='cuda'):
    run_name = 'DDPM_MoE'
    top_k = 2
    num_experts = 4
    max_classes = 3
    num_classes = 10
    image_size = 64
    lr = 3e-4
    
    setup_logging(run_name)
    
    models = []
    for i in range(num_experts):
        models.append(UNet().to(device))
    
    text_vect_size = max_classes*num_classes
    
    gated_net = GatedDiffusion(models=models, num_experts=num_experts,
                               input_size=text_vect_size, top_k=top_k).to(device)
    gated_net.load_state_dict(torch.load(os.path.join("models", run_name, f"ckpt.pt")))
    
    optimizer = optim.AdamW(gated_net.parameters(), lr=lr)
    optimizer.load_state_dict(torch.load(os.path.join("models", run_name, f"optim.pt")))
    
    gated_net.eval()
    
    labels1_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    labels2_list = []
    
    for i in range(num_of_images):
        rand2 = random.randint(0, 9)
        labels2_list.append(rand2)
    
    labels1 = torch.Tensor(labels1_list).long().to(device)
    labels2 = torch.Tensor(labels2_list).long().to(device)
    
    y1 = F.one_hot(labels1, num_classes).float()
    y2 = F.one_hot(labels2, num_classes).float()
    y = torch.cat((y1, y2), dim=1)
    
    y_padded = F.pad(y, (0, (text_vect_size - num_classes*2)))
    
    diffusion = Diffusion(img_size=image_size, device=device)
    sampled_images = diffusion.sample(gated_net, n=num_of_images, y=y_padded)
    
    for label1, label2, image in zip(labels1_list, labels2_list, sampled_images):
        os.makedirs(os.path.join("results", run_name, "2 label"), exist_ok=True)
        os.makedirs(os.path.join("results", run_name, "2 label"), exist_ok=True)
        os.makedirs(os.path.join("results", run_name, "2 label", "class" + str(label1)), exist_ok=True)
        os.makedirs(os.path.join("results", run_name, "2 label", "class" + str(label2)), exist_ok=True)
        save_images(image, os.path.join("results", run_name, "2 label", "class" + str(label1), f"{label1}.jpg"))
        save_images(image, os.path.join("results", run_name, "2 label", "class" + str(label2), f"{label2}.jpg"))


num_of_images = 10
sample_from_1_label(num_of_images)
sample_from_2_label(num_of_images)
