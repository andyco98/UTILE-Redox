import os
from PIL import Image
import re

def numerical_sort(value):
    """
    Helper function to extract numbers from the filename for sorting.
    """
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def create_tiff_stack(folder_path, output_filename):
    # Get all image paths
    images_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.lower().endswith(('.tif','.tiff','.png', '.jpg', '.jpeg', '.bmp'))]
    images_paths.sort(key=numerical_sort)  # Sort files by name

    # Open images and store in a list
    images = [Image.open(image_path) for image_path in images_paths]
    #print(images_paths)

    # Ensure all images are in the same mode and size as the first image
    images = [img.convert(images[0].mode) for img in images]
    images = [img.resize(images[0].size, Image.LANCZOS) for img in images]

    # Save images as a TIFF stack
    images[0].save(output_filename, save_all=True, append_images=images[1:])

# Usage
folder_path = 'C:/Users/andre/Desktop/zeis/Videos/S7'  # Replace with your folder path
output_filename = 's7_stack.tif'  # Replace with your desired output file name
create_tiff_stack(folder_path, output_filename)
