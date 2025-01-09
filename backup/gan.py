import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image

##################################
# Hyperparameters and Setup
##################################
latent_dim = 100
batch_size = 128
epochs = 50
learning_rate = 0.0002
generated_images_per_digit = 1000
generated_dataset_path = "generated_dataset"
model_save_path = "gan_models"

# Create the output directories
os.makedirs(generated_dataset_path, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)
for i in range(10):
    os.makedirs(os.path.join(generated_dataset_path, str(i)), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='mnist_data', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

##################################
# Generator Model
##################################
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img

##################################
# Discriminator Model
##################################
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

##################################
# Initialize Models and Optimizers
##################################
generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

adversarial_loss = nn.BCELoss()

##################################
# Training Loop
##################################
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
        # Ground truths
        real = torch.ones(imgs.size(0), 1).to(device)
        fake = torch.zeros(imgs.size(0), 1).to(device)

        # Real images
        real_imgs = imgs.to(device)

        # Train Generator
        optimizer_G.zero_grad()

        # Sample noise and generate fake images
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), real)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated images
        real_loss = adversarial_loss(discriminator(real_imgs), real)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

    print(f"Epoch {epoch+1}/{epochs} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

##################################
# Save the Models
##################################
torch.save(generator.state_dict(), os.path.join(model_save_path, "generator.pth"))
torch.save(discriminator.state_dict(), os.path.join(model_save_path, "discriminator.pth"))
print("Models saved to 'gan_models' directory.")

##################################
# Generate and Save Images
##################################
print("Generating images...")
generator.eval()
for digit in range(10):
    count = 0
    while count < generated_images_per_digit:
        z = torch.randn(batch_size, latent_dim).to(device)
        gen_imgs = generator(z)
        for img in gen_imgs:
            if count >= generated_images_per_digit:
                break
            save_image(
                img, os.path.join(generated_dataset_path, str(digit), f"{count}.png"), normalize=True
            )
            count += 1
    print(f"Generated {generated_images_per_digit} images for digit {digit}")

print("Image generation complete.")