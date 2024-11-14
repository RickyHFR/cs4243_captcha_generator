import os
import cv2
import pytesseract

def extract_text(image):
    """
    Extracts text from an image using Tesseract OCR.

    Args:
        image (numpy.ndarray): The image from which to extract text.

    Returns:
        str: The extracted text.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to binarize the image (optional)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Use pytesseract to extract text with PSM 10 (assumes a single character)
    return pytesseract.image_to_string(thresh, config='--psm 10').strip()

def filter_images(input_dir, output_dir):
    """
    Filters images in the input directory, saving only those that contain a single alphanumeric character.
    The saved images are renamed to indicate their recognized content.

    Args:
        input_dir (str): Path to the directory containing generated images.
        output_dir (str): Path to the directory where filtered images will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory at: {output_dir}")
    else:
        print(f"Output directory exists: {output_dir}")

    # Dictionary to keep count of each character
    char_counts = {}

    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to read {image_path}")
            continue
        
        # Extract text from the image
        text = extract_text(image)
        extracted_text = text.strip()
        
        # Debugging: Print the extracted text
        print(f"Extracted text from {filename}: '{extracted_text}'")
        
        # Check if the extracted text contains exactly one alphanumeric character
        if len(extracted_text) == 1 and extracted_text.isalnum():
            # Retrieve the character
            character = extracted_text.upper()  # Convert to uppercase for consistency

            # Update the count for this character
            if character in char_counts:
                char_counts[character] += 1
            else:
                char_counts[character] = 1

            # Create a new filename with the character and its count
            new_filename = f"{character}_{char_counts[character]}.png"
            output_path = os.path.join(output_dir, new_filename)

            # Save the valid image with the new filename
            success = cv2.imwrite(output_path, image)
            if success:
                print(f"Saved valid image: {output_path}")
            else:
                print(f"Failed to save image: {output_path}")
        else:
            print(f"Discarded image {filename} with extracted text: '{extracted_text}'")

    print("Image filtering and renaming completed.")

input_directory = 'data/generated_images'
output_directory = 'data/filtered_images'
filter_images(input_directory, output_directory)