# Important!! This file is only used as a draft. Actual captcha_generator please refer to the python notebook!!!

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CaptchaDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # Convert to RGB if needed
        if self.transform:
            image = self.transform(image)
        return image

# Define your transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Adjust as needed
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Specify the directory containing captcha images
image_dir = "main"  # Replace with the path to your 'main' folder

# Create the dataset and dataloader
captcha_dataset = CaptchaDataset(image_dir=image_dir, transform=transform)
dataloader = DataLoader(captcha_dataset, batch_size=32, shuffle=True)


class Generator(nn.Module):
    def __init__(self, noise_dim, img_channels):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, img_channels * 64 * 64),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return x.view(-1, 3, 64, 64)  # Assuming RGB captcha of size 64x64

class Discriminator(nn.Module):
    def __init__(self, img_channels):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Hyperparameters
noise_dim = 100
img_channels = 3
lr = 0.0002
batch_size = 32
epochs = 10000

# Initialize models
generator = Generator(noise_dim, img_channels).to(device)
discriminator = Discriminator(img_channels).to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Training loop
for epoch in range(epochs):
    for real_images in dataloader:
        real_images = real_images.to(device)
        
        # Train Discriminator
        optimizer_d.zero_grad()
        batch_size = real_images.size(0)
        labels_real = torch.ones(batch_size, 1).to(device)
        labels_fake = torch.zeros(batch_size, 1).to(device)
        
        # Train with real images
        outputs = discriminator(real_images)
        loss_real = criterion(outputs, labels_real)
        
        # Train with fake images
        noise = torch.randn(batch_size, noise_dim).to(device)
        fake_images = generator(noise)
        outputs = discriminator(fake_images.detach())
        loss_fake = criterion(outputs, labels_fake)
        
        # Update Discriminator
        d_loss = loss_real + loss_fake
        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        optimizer_g.zero_grad()
        outputs = discriminator(fake_images)
        g_loss = criterion(outputs, labels_real)  # Generator wants the Discriminator to classify fake as real
        g_loss.backward()
        optimizer_g.step()
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{epochs}] | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")

def generate_captcha(generator, noise_dim):
    generator.eval()  # Set generator to evaluation mode
    noise = torch.randn(1, noise_dim).to(device)  # Generate noise with batch size of 1
    with torch.no_grad():
        captcha = generator(noise).cpu().squeeze().detach()
    generator.train()  # Switch back to training mode if needed
    return captcha

# Display the captcha

captcha_image = generate_captcha(generator, noise_dim)
plt.imshow(captcha_image.permute(1, 2, 0).numpy() * 0.5 + 0.5)  # Rescale for display
plt.axis('off')
plt.show()
