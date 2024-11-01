import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from Generator_Discriminator import Generator, Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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
    transforms.Resize((80, 80)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Specify the directory containing captcha images
image_dir = "main"

# Create the dataset and dataloader
captcha_dataset = CaptchaDataset(image_dir=image_dir, transform=transform)
dataloader = DataLoader(captcha_dataset, batch_size=128, shuffle=True)
    
# Hyperparameters
noise_dim = 100
img_channels = 3
lr = 0.0002
batch_size = 128
epochs = 10000

# Initialize models
generator = Generator(noise_dim, img_channels).to(device)
discriminator = Discriminator(img_channels).to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

file_path = "model/generator.pth"
if os.path.getsize(file_path) > 0 and torch.cuda.is_available():
    # Load the saved state
    generator.load_state_dict(torch.load('model/generator.pth'))
    discriminator.load_state_dict(torch.load('model/discriminator.pth'))
    optimizer_g.load_state_dict(torch.load('model/optimizer_g.pth'))
    optimizer_d.load_state_dict(torch.load('model/optimizer_d.pth'))
elif os.path.getsize(file_path) > 0:
    generator.load_state_dict(torch.load('model/generator.pth', map_location=torch.device('cpu')))
    discriminator.load_state_dict(torch.load('model/discriminator.pth', map_location=torch.device('cpu')))
    optimizer_g.load_state_dict(torch.load('model/optimizer_g.pth', map_location=torch.device('cpu')))
    optimizer_d.load_state_dict(torch.load('model/optimizer_d.pth', map_location=torch.device('cpu')))

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
    if epoch % 1 == 0:
        # save the model every 10 epochs
        torch.save(generator.state_dict(), 'model/generator.pth')
        torch.save(discriminator.state_dict(), 'model/discriminator.pth')
        torch.save(optimizer_g.state_dict(), 'model/optimizer_g.pth')
        torch.save(optimizer_d.state_dict(), 'model/optimizer_d.pth')
        print(f"Epoch [{epoch}/{epochs}] | D Loss: {d_loss.item()} | G Loss: {g_loss.item()}")
        
