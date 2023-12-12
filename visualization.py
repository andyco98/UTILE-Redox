import tifffile
import vtkmodules.all as vtk
import vtkmodules.util.numpy_support as numpy_support
import numpy as np
from scipy.ndimage import label, generate_binary_structure
import pandas as pd
from matplotlib import cm


### TO DO ###
# 1 Save visualizations in some way that is reproducible without re analyzing the whole stack




def visualize_volume(volume):
    # Convert numpy array to VTK array
    vtk_data_array = numpy_support.numpy_to_vtk(num_array=volume.transpose(2, 1, 0).ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

    # Create a VTK image data object
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(volume.shape)
    vtk_image.GetPointData().SetScalars(vtk_data_array)

    # Define opacity transfer function
    opacity_transfer_function = vtk.vtkPiecewiseFunction()
    opacity_transfer_function.AddPoint(0, 0.001)  # Class 2 (now set to 0) is fully transparent
    opacity_transfer_function.AddPoint(1, 1.0)  # Class 1 is fully visible
    opacity_transfer_function.AddPoint(2, 0.2)  # Other classes are semi-transparent
    opacity_transfer_function.AddPoint(3, 0.2)  # Add more points as needed
    opacity_transfer_function.AddPoint(5, 0.0)  # Add more points as needed

   # Define color transfer function
    color_transfer_function = vtk.vtkColorTransferFunction()
    color_transfer_function.AddRGBPoint(0, 1.0, 1.0, 0.0)  # Class 2 is black (but also fully transparent)
    color_transfer_function.AddRGBPoint(1, 1.0, 0.0, 0.0)  # Class 1 is red
    color_transfer_function.AddRGBPoint(3, 0.0, 1.0, 0.0)  # Other classes are green
    color_transfer_function.AddRGBPoint(2, 0.0, 0.0, 1.0)  # Add more points as needed
    color_transfer_function.AddRGBPoint(5, 0.0, 0.0, 0.0) 

    # Create volume property
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color_transfer_function)
    volume_property.SetScalarOpacity(opacity_transfer_function)

    # Create volume
    vtk_volume = vtk.vtkVolume()
    vtk_volume.SetMapper(vtk.vtkGPUVolumeRayCastMapper())
    vtk_volume.SetProperty(volume_property)
    vtk_volume.GetMapper().SetInputData(vtk_image)


    # Create a writer for the VTK XML format
    writer = vtk.vtkXMLImageDataWriter()

    writer.SetFileName("./volume.vti")

    # Set the input data - use vtk_image instead of volume
    writer.SetInputData(vtk_image)

    # Write the file
    writer.Write()

    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(vtk_volume)

    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 800)

    # Create render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Start interaction
    render_window.Render()
    render_window_interactor.Start()

def label_bubbles(volume, bubble_class=1):
    # Apply labeling to the bubble_class within the volume
    binary_bubbles = volume == bubble_class
    structure = generate_binary_structure(3, 3)
    labeled_volume, num_features = label(binary_bubbles, structure=structure)
    return labeled_volume, num_features

def visualize_labeled_volume(labeled_volume, num_labels):    
    # Convert labeled_volume to VTK array
    vtk_data_array = numpy_support.numpy_to_vtk(num_array=labeled_volume.transpose(2, 1, 0).ravel(), deep=True, array_type=vtk.VTK_INT)
    print("num labels ", num_labels)
    # Create VTK image data object
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(labeled_volume.shape)
    vtk_image.GetPointData().SetScalars(vtk_data_array)

    # Create a color transfer function for the labels
    color_transfer_function = vtk.vtkColorTransferFunction()

    # Assign random colors to each label, but leave the background (label 0) black
    color_transfer_function.AddRGBPoint(0, 1, 1, 0)
    for i in range(1, num_labels + 1):
        color_transfer_function.AddRGBPoint(i, np.random.rand(), np.random.rand(), np.random.rand())  # RGB

    # Create a scalar opacity function
    scalar_opacity_function = vtk.vtkPiecewiseFunction()
    scalar_opacity_function.AddPoint(0, 0.001)
    for i in range(1, num_labels + 1):
        scalar_opacity_function.AddPoint(i, 1.0)

    # Create volume property and set color and opacity functions
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color_transfer_function)
    volume_property.SetScalarOpacity(scalar_opacity_function)
    #volume_property.ShadeOn()  # This will enable shading


    # Create volume and set the LUT
    volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
    volume_mapper.SetInputData(vtk_image)

    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)

    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 800)

    # Create render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Assuming 'volume' is your vtkImageData object

        # Create a writer for the VTK XML format
    writer = vtk.vtkXMLImageDataWriter()

    writer.SetFileName("./labeled_volume.vti")

    # Set the input data - use vtk_image instead of volume
    writer.SetInputData(vtk_image)

    # Write the file
    writer.Write()
    # Start interaction
    render_window.Render()
    render_window_interactor.Start()

def separate_volume(volume, membrane_class=2):
    left_volume = np.copy(volume)
    right_volume = np.copy(volume)
    
    for z in range(volume.shape[0]):
        membrane_positions = []
        
        for y in range(volume.shape[1]):
            # Find the last occurrence of the membrane in this row
            membrane = np.where(volume[z, y, :] == membrane_class)[-1]
            #membrane_right = np.where(volume[z, y, :] == membrane_class)[0]
            
            if membrane.size > 0:
                # The last occurrence of the membrane in this row
                dividing_x = membrane[0]
                membrane_positions.append(dividing_x)
                membrane_left = membrane[-1]
                membrane_right = membrane[0]
                #print("membrane left",membrane_left, "memrbane right", membrane_right)
            else:
                dividing_x = None
            
            # Separate the volumes based on the last occurrence of the membrane
            if dividing_x is not None:
                # Set all pixels before the last membrane to 0 in right_volume
                right_volume[z, y, :membrane_right] = 5
                # Set all pixels after the first occurrence to 0 in left_volume, conserving the membrane
                left_volume[z, y, membrane_left:] = 5


        
        # If the membrane was not found in a row, use the average position from other rows repair it
            if len(membrane_positions) > 0 and dividing_x == None:
                average_membrane_position = int(np.round(np.mean(membrane_positions)))
                for y in range(volume.shape[1]):
                    
                    # Apply the average position to the current row
                    right_volume[z, y, :average_membrane_position] = 5
                    # Conservatively keeping the membrane in the left volume
                    # If no membrane, just use the average to separate
                    left_volume[z, y, average_membrane_position:] = 5
                    
    return left_volume, right_volume

def remove_isolated_pixels(volume, target_class=1):
    # Create a copy of the volume to modify
    cleaned_volume = np.copy(volume)

    # Get the shape of the volume for iteration
    depth, height, width = volume.shape

    # Iterate over each slice, except the first and last
    for z in range(1, depth - 1):
        for y in range(height):
            for x in range(width):
                # Check if the current pixel belongs to the target class
                if volume[z, y, x] == target_class:
                    # Check for neighbors of the same class in adjacent slices
                    neighbors = [volume[z - 1, y, x], volume[z + 1, y, x]]
                    # If no neighbors of the same class, consider it noise and remove it
                    if not any(neighbor == target_class for neighbor in neighbors):
                        cleaned_volume[z, y, x] = 0  # Assuming 0 is the background 
                        

    return cleaned_volume

def remove_isolated_pixels2(volume, target_class=1):
    # Create a copy of the volume to modify
    cleaned_volume = np.copy(volume)

    # Create slices of the volume
    current_slice = volume[1:-1, :, :]  # Middle slices
    previous_slice = volume[:-2, :, :]  # Previous slices
    next_slice = volume[2:, :, :]       # Next slices

    # Check for neighbors of the same class in adjacent slices
    has_neighbor_previous = previous_slice == target_class
    has_neighbor_next = next_slice == target_class

    # Find isolated pixels (no neighbors of the same class)
    isolated_pixels = (current_slice == target_class) & ~(has_neighbor_previous | has_neighbor_next)

    # Set isolated pixels to background (0)
    cleaned_volume[1:-1, :, :][isolated_pixels] = 0  # Assuming 0 is the background

    return cleaned_volume

def remove_small_objects(volume, target_class=1, min_size=25):
    # Generate a binary structure for 3D connectivity (26-connected)
    #struct = generate_binary_structure(3, 3)

    struct = np.ones((3, 3, 3), dtype=np.int32)

    # Isolate the target class
    binary_target = (volume == target_class)

    # Label connected components
    labeled_array, num_features = label(binary_target, structure=struct)

    # Get sizes of the labeled regions
    sizes = np.bincount(labeled_array.ravel())
    #print(sizes)
    # Create a mask of the regions to keep (larger than min_size)
    filtered = sizes > min_size

    # Create a boolean array where True values represent the regions to remove
    removal_mask = ~filtered[labeled_array]

    # Initialize the cleaned volume as a copy of the original volume
    cleaned_volume = np.copy(volume)

    # Set the regions to remove to background (0 or another specified background value)
    cleaned_volume[removal_mask] = 0  # Assuming 0 is the background

    return cleaned_volume

def clean_volume(volume):
    c_volume = remove_isolated_pixels2(volume)
    return c_volume



#### NEED TO VISUALIZE INDIVIDUAL PROPERTIES RELATED TO INDIVIDUAL BUBBLES BY COLORING THEM WITH A CORRESPONDING PROPERTY ####

def visualize_property(property, labeled_volume, csv_file):

    #Step 1: read the labeled data and read the corresponding csv file with the corresponding properties
    df = pd.read_csv(csv_file)
    #Step 2: normalize the properties and extract the corresponding value to map the corresponding color
    # Normalize the column
    normalized_column = (df[property] - df[property].min()) / (df[property].max() - df[property].min())
    norm_property = normalized_column.tolist()
    print("norm list: ",norm_property, "len:", len(norm_property))
    #Step 3: Create the look up table to assign the correct RGB values to the correct proprety value
    c = cm.get_cmap('jet')

    #Step 4: Create the VTK pipeline in order to assign an RGB point to the correct label
    # Convert labeled_volume to VTK array
    vtk_data_array = numpy_support.numpy_to_vtk(num_array=labeled_volume.transpose(2, 1, 0).ravel(), deep=True, array_type=vtk.VTK_INT)

    # Create VTK image data object
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(labeled_volume.shape)
    vtk_image.GetPointData().SetScalars(vtk_data_array)

    # Create a color transfer function for the labels
    color_transfer_function = vtk.vtkColorTransferFunction()

    # Assign random colors to each label, but leave the background (label 0) black
    color_transfer_function.AddRGBPoint(0, 1, 1, 0)
    for i in range(1, len(norm_property)+1):
        property = norm_property[i-1]
        rgb_color = c(property)
        print("rgb color ", rgb_color)
        color_transfer_function.AddRGBPoint(i, rgb_color[0], rgb_color[1], rgb_color[2])  # RGB

    # Create a scalar opacity function
    scalar_opacity_function = vtk.vtkPiecewiseFunction()
    scalar_opacity_function.AddPoint(0, 0.001)
    for i in range(1, len(norm_property)+1):
        scalar_opacity_function.AddPoint(i, 1.0)

    # Create volume property and set color and opacity functions
    volume_property = vtk.vtkVolumeProperty()
    volume_property.SetColor(color_transfer_function)
    volume_property.SetScalarOpacity(scalar_opacity_function)
    #volume_property.ShadeOn()  # This will enable shading


    # Create volume and set the LUT
    volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
    volume_mapper.SetInputData(vtk_image)

    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)

    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 800)

    # Create render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Assuming 'volume' is your vtkImageData object

        # Create a writer for the VTK XML format
    writer = vtk.vtkXMLImageDataWriter()

    writer.SetFileName("./labeled_volume.vti")

    # Set the input data - use vtk_image instead of volume
    writer.SetInputData(vtk_image)

    # Write the file
    writer.Write()
    # Start interaction
    render_window.Render()
    render_window_interactor.Start()
    return


# # # # # Read the TIF stack and convert it to a numpy array
# volume = tifffile.imread('C:/Users/andre/Desktop/zeis/maskS9.tif')
# cleaned_volume = remove_isolated_pixels2(volume, target_class=1)
# # # # cleaned_volume = remove_small_objects(cleaned_volume, target_class=1)

# # # # #Set the values of class 2 to 0
# # # # #volume[volume == 0] = 0

# left_volume,right_volume = separate_volume(cleaned_volume)

# #left_volume = clean_volume(left_volume)
# # #Label the bubbles
# labeled_volume, num_features = label_bubbles(left_volume)

# # # # #Now you can visualize the labeled_volume
# visualize_labeled_volume(labeled_volume, num_features)
# print(num_features)
# # # # #Visualize the volume

#visualize_volume(left_volume)

# csv_file = "./output.csv"

# visualize_property("closest_distance", labeled_volume, num_features, csv_file)
