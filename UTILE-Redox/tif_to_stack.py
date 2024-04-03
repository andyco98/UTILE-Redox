import os
from PIL import Image
import re
import imageio

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
folder_path = 'C:/Users/andre/Desktop/Anntotation/predictions/'  # Replace with your folder path
output_filename = 'C:/Users/andre/Desktop/Anntotation/predicition.tif'  # Replace with your desired output file name
#create_tiff_stack(folder_path, output_filename)

def save_tif_stack_as_individual_images(tif_stack_path, case_name):
    """
    Save each slice of a TIF stack as an individual TIF image.

    Parameters:
    - tif_stack_path: Path to the input TIF stack.
    - output_folder: Folder where individual TIF images will be saved.
    """
    # Ensure the output folder exists
    if not os.path.exists(f"./{case_name}/slices"):
        os.makedirs(f"./{case_name}/slices")

    # Open the TIF stack
    tif_stack = Image.open(tif_stack_path)
    
    # Loop through each frame in the stack
    for i in range(tif_stack.n_frames):
        tif_stack.seek(i)
        
        # Define the output file path
        output_file_path = os.path.join(f"./{case_name}/slices", f"slice_{i}.tif")
        
        # Save the current frame as an individual TIF image
        tif_stack.save(output_file_path)



def create_gif(input_folder, output_filename, frame_duration=0.1):
    images = []
    # Ensure the files are sorted correctly
    file_names = sorted([img for img in os.listdir(input_folder) if img.endswith(".png")])

    for filename in file_names:
        file_path = os.path.join(input_folder, filename)
        images.append(imageio.imread(file_path))

    imageio.mimsave(output_filename, images, duration=frame_duration)

# # Usage
# input_folder = f'C:/Users/andre/Desktop/Anntotation/test_set/'  # Replace with your frames folder
# output_filename = f'C:/Users/andre/Desktop/Anntotation/real.gif'  # The output file
# frame_duration = 0.1  # Duration of each frame in the GIF in seconds
# create_gif(input_folder, output_filename, frame_duration)

#directory_folders = "C:/Users/andre/Desktop/zeis/gifs/"