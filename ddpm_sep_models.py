# -*- coding: utf-8 -*-
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
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

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


def train_gated(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    l = len(dataloader)

    # load pre-trained experts or define new experts and train with gating network
    models = []
    for i in range(args.num_experts):
        models.append(UNet().to(device))
    
    text_vect_size = args.max_classes*args.num_classes
    
    gated_net = GatedDiffusion(models=models, num_experts=args.num_experts,
                               input_size=text_vect_size, top_k=args.top_k).to(device)
    
    optimizer = optim.AdamW(gated_net.parameters(), lr=args.lr)
    
    if os.path.exists(os.path.join("models", args.run_name, f"ckpt.pt")):
        gated_net.load_state_dict(torch.load(os.path.join("models", args.run_name, f"ckpt.pt")))
        optimizer.load_state_dict(torch.load(os.path.join("models", args.run_name, f"optim.pt")))
    
    for epoch in range(args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            
            y = F.one_hot(labels, args.num_classes).float()
            y_padded = F.pad(y, ((text_vect_size - args.num_classes)//2, (text_vect_size - args.num_classes)//2))
            
            predicted_noise, aux_loss = gated_net(x_t, t, y_padded)
            loss = mse(noise, predicted_noise) + args.gated_loss_factor*aux_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(Loss=loss.item())
            logger.add_scalar("Loss", loss.item(), global_step=epoch * l + i)

        if epoch % 10 == 0:
            labels = torch.arange(10).long().to(device)
            y = F.one_hot(labels, args.num_classes).float()
            y_padded = F.pad(y, ((text_vect_size - args.num_classes)//2, (text_vect_size - args.num_classes)//2))
            sampled_images = diffusion.sample(gated_net, n=len(labels), y=y_padded)
            # ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
            plot_images(sampled_images)
            save_images(sampled_images, os.path.join("results", args.run_name, f"{epoch}.jpg"))
            # save_images(ema_sampled_images, os.path.join("results", args.run_name, f"{epoch}_ema.jpg"))
            torch.save(gated_net.state_dict(), os.path.join("models", args.run_name, f"ckpt.pt"))
            # torch.save(ema_model.state_dict(), os.path.join("models", args.run_name, f"ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join("models", args.run_name, f"optim.pt"))


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.epochs = 500
    args.batch_size = 32
    args.image_size = 64
    args.num_classes = 10
    args.dataset_path = "./cifar10-64/train"
    args.device = "cuda"
    args.lr = 3e-4
    args.max_classes = 3
    args.num_experts = 4
    args.top_k = 2
    args.gated_loss_factor = 1e-2
    args.run_name = "DDPM_MoE_1"
    train_gated(args)


def sample(device='cuda'):
    run_name = 'DDPM_MoE_1'
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
    
    epoch = 100
    labels1 = torch.randint(low=0, high=10, size=(10,)).long().to(device)
    labels2 = torch.randint(low=0, high=10, size=(10,)).long().to(device)
    
    print(labels1)
    print(labels2)
    
    y1 = F.one_hot(labels1, num_classes).float()
    y2 = F.one_hot(labels2, num_classes).float()
    y = torch.cat((y1, y2), dim=1)
    y_padded = F.pad(y, (0, (text_vect_size - num_classes*2)))
    diffusion = Diffusion(img_size=image_size, device=device)
    sampled_images = diffusion.sample(gated_net, n=len(labels1), y=y_padded)
    # ema_sampled_images = diffusion.sample(ema_model, n=len(labels), labels=labels)
    plot_images(sampled_images)
    save_images(sampled_images, os.path.join("results", run_name, f"{epoch}.jpg"))


if __name__ == '__main__':
    launch()
#     sample()
    # device = "cuda"
    # model = UNet_conditional(num_classes=10).to(device)
    # ckpt = torch.load("./models/DDPM_conditional/ckpt.pt")
    # model.load_state_dict(ckpt)
    # diffusion = Diffusion(img_size=64, device=device)
    # n = 8
    # y = torch.Tensor([6] * n).long().to(device)
    # x = diffusion.sample(model, n, y, cfg_scale=0)
    # plot_images(x)

