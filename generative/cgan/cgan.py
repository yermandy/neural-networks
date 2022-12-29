"""
    Paper: Conditional Generative Adversarial Nets
    https://arxiv.org/pdf/1411.1784.pdf
"""

import argparse
import os
import random
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets

import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm.rich import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")
parser.add_argument("--print_interval", type=int, default=100, help="interval between prints")
parser.add_argument("--device", type=str, default="cpu", help="device")
parser.add_argument("--output_path", type=str, default="images", help="device")
args = parser.parse_args()

print(args)

img_shape = (args.channels, args.img_size, args.img_size)

os.makedirs(args.output_path, exist_ok=True)

class Generator(nn.Module):
    def __init__(self, num_classes=10, embedding_size=10):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.embedding = nn.Embedding(num_classes, embedding_size)

        self.model = nn.Sequential(
            *block(args.latent_dim + embedding_size, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, latents, labels):
        # embed labels
        gen_input = torch.cat([latents, self.embedding(labels)], 1)
        img = self.model(gen_input)
        
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, num_classes=10, embedding_size=10):
        super(Discriminator, self).__init__()
        
        self.embedding = nn.Embedding(num_classes, embedding_size)

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)) + embedding_size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, images, labels):
        img_flat = images.view(images.size(0), -1)
        
        img_and_labels = torch.cat([img_flat, self.embedding(labels)], dim=1)
        
        validity = self.model(img_and_labels)

        return validity


def seed_all(seed=0, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

def seed_worker(worker_id):
    # Set dataloader worker seed https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

seed_all(42)

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator().to(args.device)
discriminator = Discriminator().to(args.device)


# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        ),
    ),
    batch_size=args.batch_size,
    shuffle=True,
    worker_init_fn=seed_worker
)

# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=args.lr, betas=(args.b1, args.b2)
)
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2)
)

# ----------
#  Training
# ----------

for epoch in range(args.n_epochs):
    for i, (imgs, labels) in enumerate(tqdm(dataloader)):

        # Adversarial ground truths
        valid = torch.ones((len(imgs), 1), device=args.device)
        fake = torch.zeros((len(imgs), 1), device=args.device)

        # Configure input
        real_imgs = imgs.to(args.device, non_blocking=True)

        # -----------------
        #  Train Generator
        # -----------------
        
        # D: X -> [0, 1]
        # G: Z -> X
        
        # generator loss is E_z[log(1 - D(G(z|y)))]

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = torch.randn((len(imgs), args.latent_dim), device=args.device)

        # Generate a batch of images
        gen_imgs = generator(z, labels)

        # Loss measures generator's ability to fool the discriminator
        # train G by minimizing log(1 - D(G(z|y)))
        # g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        
        # or by maximizing log(D(G(z|y))) as proposed in the paper
        g_loss = -adversarial_loss(discriminator(gen_imgs, labels), fake)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # discriminator loss is E_x[log(D(x|y))] + E_z[log(1 - D(G(z|y)))]
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs, labels), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach(), labels), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        batches_done = epoch * len(dataloader) + i

        if batches_done % args.print_interval == 0:
            print(
                f"[Epoch {epoch}/{args.n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
            )

        if batches_done % args.sample_interval == 0:
            save_image(
                gen_imgs.data[:25],
                f"{args.output_path}/{batches_done}_fake.png",
                nrow=5,
                normalize=True,
            )
            save_image(
                imgs.data[:25],
                f"{args.output_path}/{batches_done}_true.png",
                nrow=5,
                normalize=True,
            )
