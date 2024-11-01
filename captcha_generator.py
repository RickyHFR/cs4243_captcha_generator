import torch
import matplotlib.pyplot as plt
from Generator_Discriminator import Generator, Discriminator

noise_dim = 100
img_channels = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize models
generator = Generator(noise_dim, img_channels).to(device)
discriminator = Discriminator(img_channels).to(device)

if torch.cuda.is_available():
    generator.load_state_dict(torch.load('model/generator.pth'))
    discriminator.load_state_dict(torch.load('model/discriminator.pth'))
else:
    generator.load_state_dict(torch.load('model/generator.pth', map_location=torch.device('cpu')))
    discriminator.load_state_dict(torch.load('model/discriminator.pth', map_location=torch.device('cpu')))

def generate_captcha(generator, noise_dim):
    generator.eval()  # Set generator to evaluation mode
    noise = torch.randn(1, noise_dim).to(device)  # Generate noise with batch size of 1
    with torch.no_grad():
        captcha = generator(noise).cpu().squeeze().detach()
    print(captcha.shape)
    return captcha.reshape((3, 80, 80)) # TODO: reshape to proper size

# Display the captcha
for i in range(10):
    captcha_image = generate_captcha(generator, noise_dim)
    plt.imshow(captcha_image.permute(1, 2, 0).numpy() * 0.5 + 0.5)  # Rescale for display
    plt.axis('off')
    plt.show()
