import numpy as np
import tifffile

def transform_labels(tiff_path, output_path):
    """
    Transform class labels in a TIFF stack.
    
    Args:
    tiff_path (str): Path to the input TIFF file.
    output_path (str): Path to save the transformed TIFF file.
    """
    # Read the TIFF stack
    with tifffile.TiffFile(tiff_path) as tif:
        images = tif.asarray()

    # Apply the transformations
    # Class 2 to 0, 3 to 2, 4 to 3
    images[images == 2] = 0
    images[images == 3] = 2
    images[images == 4] = 3

    # Save the transformed stack
    tifffile.imwrite(output_path, images)

# # Example usage
# transform_labels('C:/Users/a.colliard/Desktop/zeis_imgs/frame1_65.tif', 'C:/Users/a.colliard/Desktop/zeis_imgs/annfig.tif')
