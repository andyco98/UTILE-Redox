import numpy as np
from PIL import Image


# Specify the path to your image file
path = "frame1_1.tif"  # Replace with your image file path

# Open the image
image = Image.open(path)

# Convert the image to a NumPy array
image = np.array(image)

# Get the dimensions of the image
x, y = image.shape

# Loop through each pixel in the image
for i in range(x):
    for j in range(y):
        # Check if the pixel value is not equal to 3 (assuming you want to compare it to 3)
        if not np.array_equal(image[i, j], 3):
            # Set the pixel value to 0
            image[i, j] = 0

# Save the modified image
result_image = Image.fromarray(image)
result_image.save("result_image.jpg")