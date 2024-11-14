import shutil
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path
from PIL import Image

def label_images(input_folder, output_folder, batch_size=5):
    # Ensure output folder exists
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Get list of all images in the input folder
    image_paths = glob.glob(os.path.join(input_folder, '*'))
    index = 0

    while index < len(image_paths):
        # Load a batch of images
        batch_images = image_paths[index:index + batch_size]
        
        # Display each image in the batch
        fig, axes = plt.subplots(1, batch_size, figsize=(15, 3))
        for ax, image_path in zip(axes, batch_images):
            img = Image.open(image_path)
            ax.imshow(img)
            ax.axis('off')
        plt.show(block=False)  # Display without blocking the code execution

        # Prompt for label input
        label_input = input("Enter labels for valid images (separate with spaces, use a comma for invalid images): ")
        labels = label_input.strip().split()

        # Close the figure after getting the input
        plt.close(fig)

        # Check that the number of labels matches batch size
        if len(labels) != batch_size:
            print(f"Please enter {batch_size} labels.")
            continue

        # Process each label and copy valid images
        for i, label in enumerate(labels):
            if label != ',':  # Comma indicates an invalid image
                # Copy the valid image to the output folder with the label as the filename
                valid_image_path = batch_images[i]
                new_image_name = f"{label}_{os.path.basename(valid_image_path)}"  # Prefix with label
                new_image_path = os.path.join(output_folder, new_image_name)
                shutil.copy(valid_image_path, new_image_path)  # Copy image file

        # Move to the next batch
        index += batch_size

    print("Labeling complete!")

# Usage example
input_folder = 'data/filtered_images'
output_folder = 'data/properly_labeled'
label_images(input_folder, output_folder)
