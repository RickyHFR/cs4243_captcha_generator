import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from captcha_generator_train import Generator, Discriminator

noise_dim = 100
img_channels = 3
lr = 0.0002
batch_size = 128
epochs = 1

# When you want to use the model later
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = Generator(noise_dim, img_channels).to(device)
discriminator = Discriminator(img_channels).to(device)

# Load the saved state
generator.load_state_dict(torch.load('model/generator.pth'))
discriminator.load_state_dict(torch.load('model/discriminator.pth'))

# If you saved the optimizer, load it as well
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

optimizer_g.load_state_dict(torch.load('model/optimizer_g.pth'))
optimizer_d.load_state_dict(torch.load('model/optimizer_d.pth'))

def generate_captcha(generator, noise_dim):
    generator.eval()  # Set generator to evaluation mode
    noise = torch.randn(1, noise_dim).to(device)  # Generate noise with batch size of 1
    with torch.no_grad():
        captcha = generator(noise).cpu().squeeze().detach()
    generator.train()  # Switch back to training mode if needed
    print(captcha.shape)
    return captcha.reshape((3, 80, 80)) # TODO: reshape to proper size

# Display the captcha

captcha_image = generate_captcha(generator, noise_dim)
plt.imshow(captcha_image.permute(1, 2, 0).numpy() * 0.5 + 0.5)  # Rescale for display
plt.axis('off')
plt.show()
