import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

default_folder_path = "properly_labeled"


def sharpen_image(image):
    new_captcha = cv2.medianBlur(image, 3)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 
    # kernel = np.array([
    #                     [0,  0, -1,  0,  0],
    #                     [0, -1, -2, -1,  0],
    #                     [-1, -2, 17, -2, -1],
    #                     [0, -1, -2, -1,  0],
    #                     [0,  0, -1,  0,  0]
    #                     ])
    sharpened_image = cv2.filter2D(new_captcha, -1, kernel) 
    return sharpened_image

def get_file_names(folder_path):
    all_single_captcha = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return all_single_captcha

folder_path = "properly_labeled"

# Color white areas of the image with random colors
def colorSingleCaptcha(image):
    colored_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    color = [255, 255, 255]
    while color == [255, 255, 255]:
        color =  [
                        random.randint(0, 230),  # Blue channel
                        random.randint(0, 230),  # Green channel
                        random.randint(0, 230)   # Red channel
                    ]
    # Apply random colors to white areas and keep the background white
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > 0:
                colored_image[i, j] = color
            else:
                colored_image[i, j] = [255, 255, 255]
    
    return colored_image

def reshape_pixel(image):
    row, col = image.shape
    for i in range(row):
        for j in range(col):
            if image[i][j] > 50:
                image[i][j] = 255
            else:
                image[i][j] = 0
    return image


def transparent_background(img):
    new_img = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if np.all(img[i, j] > 250):
                new_img[i, j] = [255, 255, 255, 0]
            else:
                new_img[i, j] = [img[i, j][0], img[i, j][1], img[i, j][2], 255]
    return new_img


def rotate_integrate(colored_img_list, num=4, size=32, std_row=80, std_col=460):
    integrated_captcha = np.zeros((std_row, std_col, 4), dtype=np.uint8)
    process_list = []
    
    for i in range(min(num, len(colored_img_list))):
        img = colored_img_list[i]
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
        img = sharpen_image(img)
        img = transparent_background(img)
        # Rotate the image by a random angle between -30 and 30 degrees
        angle = random.uniform(-30, 30)
        center = (img.shape[1] // 2, img.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
        process_list.append(rotated_img)
        
    spacing = random.randint(size, std_col // num + 1)
    initial_col = (std_col - spacing * num) // 2
    for i in range(len(process_list)):
        img = process_list[i]
        row_start = random.randint((std_row - size) // 2, (std_row + size) // 2 - size)
        col_start = i * spacing + initial_col
        cnd = img[:, :, 3] > 0
        integrated_captcha[row_start:row_start+img.shape[0], col_start:col_start+img.shape[1], :][cnd] = img[cnd]
        
    for i in range(integrated_captcha.shape[0]):
        for j in range(integrated_captcha.shape[1]):
            if integrated_captcha[i, j][3] == 0:
                integrated_captcha[i, j] = [255, 255, 255, 255]
    
    return integrated_captcha[:, :, :3]

def add_noise_and_lines(image_array, noise_points=10, num_lines=2):
    """
    Add random noise points and vertical or horizontal lines to an input image.

    Parameters:
    - image_array (numpy array): Input image.
    - noise_points (int): Number of random noise points to add.
    - num_lines (int): Number of vertical or horizontal lines to add.

    Returns:
    - Image object with noise and lines.
    """
    # Open the image
    image = Image.fromarray(image_array)
    draw = ImageDraw.Draw(image)
    # print(image.size)
    width, height = image.size

    # Add random noise points
    for _ in range(noise_points):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        color = tuple(np.random.randint(0, 150, 3))  # RGB color
        draw.point((x, y), fill=color)

    # Add vertical or horizontal lines
    for _ in range(num_lines):
        is_vertical = random.choice([True, False])
        color = tuple(np.random.randint(0, 256, 3))  # RGB color

        if is_vertical:
            x_start = random.randint(0, width // 3)
            x_end = random.randint(width * 2 // 3, width - 1)
            y_start = random.randint(0, height - 1)
            y_end = random.randint(y_start, y_start + height // 4)
            draw.line([(x_start, y_start), (x_end, y_end)], fill=color, width=1)
        else:
            y_start = random.randint(0, height // 3)
            y_end = random.randint(height * 2 // 3, height - 1)
            x_start = random.randint(0, width - 1)
            x_end = random.randint(x_start, x_start + width // 4)
            draw.line([(x_start, y_start), (x_end, y_end)], fill=color, width=1)

    return image

# Return a list with all fild read as numpy array in grayscale
def get_all_img_list(file_list, folder_path="properly_labeled"): # n: number of captchas to generate
    img_lst = []
    for f in file_list:
        img = cv2.imread(f"{folder_path}/{f}", cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
        img = cv2.medianBlur(img, 1)
        img_lst.append(img)
    return img_lst

def generate_captcha(colored_img_list, captcha_len):
    while True:
        try:
            captcha = rotate_integrate(colored_img_list, captcha_len)
            num_noise = random.randint(10, 50)
            captcha = add_noise_and_lines(captcha, num_noise)
            return captcha
        except:
            pass
    
def save_captcha(captcha, target_path, captcha_name):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    captcha.save(f"{target_path}/{captcha_name}-0.png")

def generate_captcha_to_dir(input_dir="properly_labeled", target_dir="generated_captcha", n=5):
    all_file_names = get_file_names(input_dir)
    all_img_list = get_all_img_list(all_file_names, input_dir)
    all_colored_img_list = []
    for _, img in tqdm.tqdm(enumerate(all_img_list)):
        new_img = colorSingleCaptcha(reshape_pixel(img))
        all_colored_img_list.append(new_img)
    
    for i in tqdm.tqdm(range(n), desc="Processing", ncols=100):
        captcha_len = random.randint(4, 6)
        idxs = random.sample(range(len(all_colored_img_list)), captcha_len)
        colored_img_list = [all_colored_img_list[idx] for idx in idxs]
        file_names = [all_file_names[idx] for idx in idxs]
        captcha = generate_captcha(colored_img_list, captcha_len)
        captcha_name = ''.join([char_file[0].lower() for char_file in file_names])
        save_captcha(captcha, target_dir, captcha_name)
    return


if __name__ == "__main__":
    generate_captcha_to_dir(input_dir=default_folder_path, target_dir="generated_captcha", n=2000)
    print("Done")