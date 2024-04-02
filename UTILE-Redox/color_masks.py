from PIL import Image
import numpy as np

def map_classes_to_colors(image_path, output_path):
    # Define the color mapping
    color_map = {
        0: [255, 255, 0],   # Yellow
        1: [255, 0, 0],     # Red
        2: [0, 0, 255],     # Blue
        3: [0, 255, 0]     # Green        # Assuming class 4 is black or another color
    }

    # Open the image
    img = Image.open(image_path)
    img_array = np.array(img)

    # Create an empty RGB image
    rgb_image = np.zeros((*img_array.shape, 3), dtype=np.uint8)

    # Map each class to its corresponding color
    for class_value, color in color_map.items():
        mask = img_array == class_value
        rgb_image[mask] = color

    # Save the RGB image
    Image.fromarray(rgb_image).save(output_path)

# # Usage
# map_classes_to_colors('C:/Users/a.colliard/Desktop/zeis_imgs/annfig.tif', 'C:/Users/a.colliard/Desktop/zeis_imgs/output_rgb_image.png')
