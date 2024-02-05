import tifffile
import vtkmodules.all as vtk
import vtkmodules.util.numpy_support as numpy_support
import numpy as np
from scipy.ndimage import label, generate_binary_structure
import pandas as pd
from matplotlib import cm
import os
import math

### TO DO ###


def visualize_volume(volume, capture = False):
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

    if capture == True:
            volume_dims = vtk_volume.GetMapper().GetInput().GetDimensions()
            capture_frames(render_window, renderer, vtk_volume, volume_dims, num_frames=360)
    else:
        # Start interaction
        render_window.Render()
        render_window_interactor.Start()

def visualize_real_volume(volume, capture = False):
    # Convert numpy array to VTK array
    vtk_data_array = numpy_support.numpy_to_vtk(num_array=volume.transpose(2, 1, 0).ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)

    # Create a VTK image data object
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(volume.shape)
    vtk_image.GetPointData().SetScalars(vtk_data_array)
    
    # Ambient, diffuse, and specular properties
    # # Create volume property
    volume_property = vtk.vtkVolumeProperty()
    # Ambient, diffuse, and specular properties
    # Create volume
    vtk_volume = vtk.vtkVolume()
    vtk_volume.SetMapper(vtk.vtkGPUVolumeRayCastMapper())
    vtk_volume.SetProperty(volume_property)
    vtk_volume.GetMapper().SetInputData(vtk_image)

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

    if capture == True:
            volume_dims = vtk_volume.GetMapper().GetInput().GetDimensions()
            capture_frames(render_window, renderer, vtk_volume, volume_dims, num_frames=360)
    else:
        # Start interaction
        render_window.Render()
        render_window_interactor.Start()

def label_bubbles(volume, bubble_class=1):
    # Apply labeling to the bubble_class within the volume
    binary_bubbles = volume == bubble_class
    structure = generate_binary_structure(3, 3)
    labeled_volume, num_features = label(binary_bubbles, structure=structure)
    return labeled_volume, num_features

def visualize_labeled_volume(labeled_volume, num_labels, capture = False):    
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

    if capture == True:
            volume_dims = volume.GetMapper().GetInput().GetDimensions()
            capture_frames(render_window, renderer, volume, volume_dims, num_frames=360)
    else:
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
    c_volume = remove_isolated_pixels(volume)
    return c_volume

def visualize_property(property, labeled_volume, csv_file, log = False, side="whole", capture = False):

    #Step 1: read the labeled data and read the corresponding csv file with the corresponding properties
    df = pd.read_csv(csv_file)
    
    if property == "orientation":
        #property = "phi"
        # Normalize theta and phi
        df['theta_norm'] = df['theta'] / (2 * np.pi)
        df['phi_norm'] = df['phi'] / np.pi

        # Set weights for theta and phi
        w_theta = 0.5
        w_phi = 0.5

        # Calculate the weighted average (linear combination)
        df['orientation'] = w_theta * df['theta_norm'] + w_phi * df['phi_norm']


    property_df = df[property][df[property] > 0]
    #Step 2: normalize the properties and extract the corresponding value to map the corresponding color
    # Normalize the column
    if property == "closest_distance":
        print(side)

        x,y,z = labeled_volume.shape
        max_distance = [x*0.25, y*0.25, z*0.25] #The max point should be the shortest side of the volume since it is the limitating factor, but only if the volume has the membrane in the middle
        max_point = min(max_distance)
        print(max_point)
        normalized_column = (property_df - property_df.min()) / (max_point - property_df.min())

    else: 
        if log:
            normalized_column = (np.log10(property_df) - np.log10(property_df).min()) / (np.log10(property_df).max() - np.log10(property_df).min())
        else: normalized_column = (property_df - property_df.min()) / (property_df.max() - property_df.min())

    norm_property = normalized_column.tolist()

    print(df[property])
    #print("norm list: ",norm_property, "len:", len(norm_property))

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
        #print("rgb color ", rgb_color)
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

    if capture == True:
            volume_dims = volume.GetMapper().GetInput().GetDimensions()
            capture_frames(render_window, renderer, volume, volume_dims, num_frames=360)
    else:
        # Start interaction
        render_window.Render()
        render_window_interactor.Start()
    return

def membrane_block_visualization(volume, filtered_volume, bubble_class=1, membrane_class=2, capture=False):

    membrane_voxels = np.where(volume == membrane_class)

    membrane_coords = list(zip(membrane_voxels[0], membrane_voxels[1], membrane_voxels[2]))
    #Calculate the amount of pixels touching the membrane 
    membrane_with_bubble_neighbors = np.zeros_like(volume, dtype=bool)
    blocking_voxel = 0
    # Define neighbor offsets for a 6-connected neighborhood
    neighbor_offsets = [
        (-1, 0, 0), (1, 0, 0),  # x-axis neighbors
        (0, -1, 0), (0, 1, 0),  # y-axis neighbors
        (0, 0, -1), (0, 0, 1)   # z-axis neighbors
    ]

    touching_bubble_label = []
    #Step 1: find membrane pixel with bubble neighbor and get the coordinate

    # Iterate over membrane coordinates
    for z, y, x in membrane_coords:
        # Check each neighboring voxel
        for dz, dy, dx in neighbor_offsets:
            nz, ny, nx = z + dz, y + dy, x + dx
            # Check if neighbor is within bounds and if it's a bubble
            if 0 <= nz < volume.shape[0] and 0 <= ny < volume.shape[1] and 0 <= nx < volume.shape[2]:
                if volume[nz, ny, nx] == bubble_class:
                    membrane_with_bubble_neighbors[z, y, x] = True
                    blocking_voxel += 1
                    if filtered_volume[nz,ny,nx] not in touching_bubble_label: touching_bubble_label.append(filtered_volume[nz,ny,nx])
                    #print("touching_bubble_coordinate", touching_bubble_label)
                    break  # Stop checking other neighbors

    print("Blocking voxel number", blocking_voxel)


    #Step 2: check the corresponding labeled bubble to that pixel
    
    new_volume = np.zeros_like(filtered_volume)

    for label in touching_bubble_label:
        if label != 0:
            new_volume[filtered_volume == label] = 1
        else: continue
    #Step 3: just show the bubbles with those values and remove the other ones
    for class_label in np.unique(volume):
        if class_label not in [0, 1]:  # Assuming 0 is background, 1 is bubble
            new_volume[volume == class_label] = class_label

    print("Blocking bubble visualization")
    visualize_volume(new_volume, capture)

    return blocking_voxel

def visualize_volume_realistic(volume):
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
    volume_property.SetShade(True)
    #volume_property.SetInterpolationTypeToLinear()
    volume_property.SetInterpolationTypeToNearest()

    # Ambient, diffuse, and specular properties
    volume_property.SetAmbient(0.8)
    volume_property.SetDiffuse(2)
    volume_property.SetSpecular(0.3)
    volume_property.SetSpecularPower(10.0)
    
    # Create volume
    vtk_volume = vtk.vtkVolume()
    volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
    volume_mapper.SetSampleDistance(0.1)
    vtk_volume.SetMapper(volume_mapper)
    vtk_volume.SetProperty(volume_property)
    vtk_volume.GetMapper().SetInputData(vtk_image)

    # Create renderer
    renderer = vtk.vtkRenderer()
    renderer.AddVolume(vtk_volume)
    renderer.SetBackground(1, 1,1)  # Set background color

    # Advanced lighting
    light = vtk.vtkLight()
    light.SetFocalPoint(1, 1, 1)
    light.SetPosition(0, 1, 0)
    light.SetColor(1.0, 1.0, 1.0)  # Bright white light
    light.SetIntensity(1.5)  # Increase the intensity
    renderer.AddLight(light)

    # Additional light (optional)
    light2 = vtk.vtkLight()
    light2.SetFocalPoint(0, 0, 0)
    light2.SetPosition(-1, -1, 1)
    light2.SetColor(1.0, 1.0, 1.0)
    light2.SetIntensity(0.7)
    renderer.AddLight(light2)

    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 800)
    render_window.SetMultiSamples(128)  # Set number of samples for anti-aliasing

    # Create render window interactor
    render_window_interactor = vtk.vtkRenderWindowInteractor()
    render_window_interactor.SetRenderWindow(render_window)

    # Start interaction
    render_window.Render()
    render_window_interactor.Start()

def npy_to_vtk(npy_file, vtk_file):
    # Load the numpy array from file
    array = np.load(npy_file)

    # Convert numpy array to VTK array
    vtk_data_array = numpy_support.numpy_to_vtk(num_array=array.ravel(order='F'), deep=True)

    # Create a vtkImageData object and set the vtk array as its scalars
    vtk_image = vtk.vtkImageData()
    vtk_image.SetDimensions(array.shape)
    vtk_image.GetPointData().SetScalars(vtk_data_array)

    # Write to a VTK file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(vtk_file)
    writer.SetInputData(vtk_image)
    writer.Write()

def capture_frames(render_window, renderer, vtk_volume, volume_dims, num_frames=360, output_folder="C:/Users/andre/Desktop/zeis/frames", distance_multiplier=4.5):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.SetInputBufferTypeToRGB()
    window_to_image_filter.ReadFrontBufferOff()

    image_writer = vtk.vtkPNGWriter()

    # Calculate the center of the volume
    volume_center = np.array([volume_dims[0]/2, volume_dims[1]/2, volume_dims[2]/2])

    # Get the camera
    camera = renderer.GetActiveCamera()
    camera.SetClippingRange(1, 5000)
    # Set the initial position of the camera to view the x,y plane
    initial_camera_distance = np.linalg.norm(np.array(camera.GetPosition()) - volume_center) * distance_multiplier
    camera.SetPosition(volume_center[0], volume_center[1], initial_camera_distance)
    camera.SetFocalPoint(volume_center)
    camera.SetViewUp(0, 1, 0)  # Align the up vector with the y-axis

    for i in range(num_frames):
        # Rotate the camera around the volume
        angle_rad = 2 * np.pi / num_frames * i
        x = volume_center[0] + initial_camera_distance * math.sin(angle_rad)
        z = volume_center[2] + initial_camera_distance * math.cos(angle_rad)
        camera.SetPosition(x, volume_center[1], z)

        # Render and capture the frame
        render_window.Render()
        window_to_image_filter.Modified()
        image_writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        image_writer.SetFileName(f"{output_folder}/frame_{i:03d}.png")
        image_writer.Write()

    # Reset the camera to its initial settings
    camera.SetPosition(volume_center[0], volume_center[1], initial_camera_distance)
    camera.SetFocalPoint(volume_center)
    camera.SetViewUp(0, 1, 0)





# # Example usage
# npy_file = 'C:/Users/andre/Desktop/zeis/filtered_volume_s10.npy'
# vtk_file = "C:/Users/andre/Desktop/zeis/filtered_volume_s10.vtk"
# npy_to_vtk(npy_file, vtk_file)

# # # # # Read the TIF stack and convert it to a numpy array
volume = tifffile.imread('C:/Users/andre/Desktop/zeis/maskS9.tif')

# #cleaned_volume = remove_isolated_pixels2(volume, target_class=1)
# # # # cleaned_volume = remove_small_objects(cleaned_volume, target_class=1)

# # # # #Set the values of class 2 to 0
# # # # #volume[volume == 0] = 0

# left_volume,right_volume = separate_volume(volume)

# #left_volume = clean_volume(left_volume)
# # #Label the bubbles
# labeled_volume, num_features = label_bubbles(left_volume)

# # # # #Now you can visualize the labeled_volume

# print(num_features)
# # # # # #Visualize the volume

# # #visualize_volume(left_volume)

filtered_volume = np.load("C:/Users/andre/Desktop/zeis/filtered_volume_S9.npy")

# left_volume, right_volume = separate_volume(volume)

# visualize_volume(right_volume, True)

csv_file = "C:/Users/andre/Desktop/zeis/output_S9.csv"

#blocking_voxel = membrane_block_visualization(volume, filtered_volume, capture=True)

visualize_property("orientation", filtered_volume, csv_file, side ="whole", capture = True)
#visualize_property("elongation", filtered_volume, csv_file, side ="whole", log=True,capture = True)
#visualize_property("volume", filtered_volume, csv_file, side ="whole", log=True,capture = True)
#visualize_property("sphericity", filtered_volume, csv_file, side ="whole",capture = True)
#visualize_property("flatness", filtered_volume, csv_file, side ="whole", log=True, capture = True)

# labeled_volume, num_features = label_bubbles(volume)
# print(num_features)
# visualize_labeled_volume(labeled_volume, num_features, True)
#visualize_volume_realistic(volume)
#visualize_labeled_volume(volume, True)