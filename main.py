import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Dataset class for loading single captcha images
class SingleCaptchaDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image

# Generator and Discriminator models
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            nn.ConvTranspose2d(nz, 128, 7, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State size. (128) x 7 x 7
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State size. (64) x 14 x 14
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output size. (1) x 28 x 28
        )

    def forward(self, input):
        return self.main(input.view(-1, self.nz, 1, 1))

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input size: (1) x 28 x 28
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (64) x 14 x 14
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (128) x 7 x 7
            nn.Conv2d(128, 1, 7, 1, 0, bias=False),
            # Output size: (1) x 1 x 1
        )
    
    def forward(self, input):
        out = self.main(input)
        return out.view(-1, 1)  # Output shape: [batch_size, 1]

# Function to generate images using a trained generator model
def generate_images(model_path, output_dir, noise_dim=128, num_levels=4, num_images=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator(noise_dim).to(device)
    
    try:
        generator.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    generator.eval()

    if not os.path.exists(output_dir):
        print(f"Creating output directory at: {output_dir}")
        os.makedirs(output_dir)
    else:
        print(f"Output directory exists: {output_dir}")
    
    batch_size = 64  # Adjust based on memory capacity
    num_batches = (num_images + batch_size - 1) // batch_size

    print(f"Generating {num_images} images...")
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            current_batch_size = min(batch_size, num_images - batch_idx * batch_size)
            noise = torch.randn(current_batch_size, noise_dim, 1, 1, device=device)
            fake_images = generator(noise)

            # Denormalize the images to [0, 1]
            fake_images = fake_images * 0.5 + 0.5

            # Quantize the images into groups
            quantized_images = torch.floor(fake_images * num_levels) / num_levels + (0.5 / num_levels)

            # Move images to CPU
            quantized_images_cpu = quantized_images.cpu()

            for i in range(current_batch_size):
                image_idx = batch_idx * batch_size + i + 1
                image_path = os.path.join(output_dir, f"generated_image_{image_idx}.png")
                try:
                    save_image(quantized_images_cpu[i], image_path)
                    print(f"Saved image {image_idx} to {image_path}")
                except Exception as e:
                    print(f"Error saving image {image_idx}: {e}")

    print("Image generation completed.")

# # Example usage
# generate_images(
#     model_path="models/checkpoint_model/generator_epoch_889.pth",
#     output_dir="generated_images",
#     noise_dim=128,
#     num_levels=4,
#     num_images=10000
# )

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

to_pil_image = transforms.ToPILImage()

image_dir = "data/processed_image"

# Hyperparameters
noise_dim = 128
img_channels = 1
# Adjust learning rate for RMSProp (WGAN paper recommends 0.00005)
lr = 0.00005
batch_size = 512
epochs = 3000

# Create the dataset and dataloader
captcha_dataset = SingleCaptchaDataset(image_dir=image_dir, transform=transform)
dataloader = DataLoader(captcha_dataset, batch_size=batch_size, shuffle=True)

# Initialize models
generator = Generator(noise_dim)
discriminator = Discriminator()
generator = generator.to(device)
discriminator = discriminator.to(device)

# Initialize optimizers
optimizer_g = optim.RMSprop(generator.parameters(), lr=lr)
optimizer_d = optim.RMSprop(discriminator.parameters(), lr=lr)

# Function to store sample images
def store_sample(generator, epoch):
    if not os.path.exists('sample'):
        os.makedirs('sample')

    noise = torch.randn(batch_size, noise_dim).to(device)
    sample_images = generator(noise)
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.axis('off')
        plt.imshow(to_pil_image(sample_images[i][0].detach()), cmap='gray')
    plt.savefig(f'sample/sample_epoch_{epoch}.png')
    plt.close()

# Function to store model checkpoints
def store_model(generator, discriminator, epoch):
    if not os.path.exists('models'):
        os.makedirs('models')
    torch.save(generator.state_dict(), f'models/generator_epoch_{epoch}.pth')
    torch.save(discriminator.state_dict(), f'models/discriminator_epoch_{epoch}.pth')

# Function to compute the gradient penalty
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# Define the number of critic updates per generator update
n_critic = 5
clip_value = 0.01  # Weight clipping range
lambda_gp = 10  # Gradient penalty coefficient

# Training loop
for epoch in range(epochs):
    d_epoch_loss = 0
    g_epoch_loss = 0
    total_batches = len(dataloader)

    for i, real_images in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # --- Update Discriminator (Critic) ---
        for _ in range(n_critic):
            optimizer_d.zero_grad()

            # Real images
            outputs_real = discriminator(real_images)
            d_loss_real = -torch.mean(outputs_real)

            # Fake images
            noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)
            fake_images = generator(noise).detach()
            outputs_fake = discriminator(fake_images)
            d_loss_fake = torch.mean(outputs_fake)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_images.data, fake_images.data)
            lambda_gp = 10  # Gradient penalty coefficient

            # Total critic loss
            d_loss = d_loss_real + d_loss_fake + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_d.step()

        # --- Update Generator ---
        optimizer_g.zero_grad()
        noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)
        fake_images = generator(noise)
        outputs = discriminator(fake_images)
        g_loss = -torch.mean(outputs)
        g_loss.backward()
        optimizer_g.step()

        d_epoch_loss += d_loss.item()
        g_epoch_loss += g_loss.item()

        percent_complete = ((i + 1) / total_batches) * 100
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Batch {i+1}/{total_batches} "
            f"({percent_complete:.2f}% complete)",
            end='\r'
        )

    # Averaging losses
    d_epoch_loss /= len(dataloader)
    g_epoch_loss /= len(dataloader)

    # Save samples and models
    store_sample(generator, epoch)
    if (epoch + 1) % 10 == 0:
        store_model(generator, discriminator, epoch)

    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {d_epoch_loss:.4f} | G Loss: {g_epoch_loss:.4f}")

print('DONE TRAINING')

# Save the final models
torch.save(generator.state_dict(), 'models/generator_self_train.pth')
torch.save(discriminator.state_dict(), 'models/discriminator_self_train.pth')