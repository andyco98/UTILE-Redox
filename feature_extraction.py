 #### Different functions to extract information from the 3D volumes #####
import tifffile
import pandas as pd
from visualization import *
import numpy as np
from skimage.measure import regionprops, marching_cubes, mesh_surface_area
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor
import csv


### TO DO ###
# 1 Check membrane blockade, some bubbles are still displayed even if the are not touching the membrane or just touching the gasket X
# 2 Enhance the visualization with ParaView, or just wrtie some scripts to automate the tranfer to paraview?
# 3 Check the cleaning functions with manual annotated volumes to see if too much (or too less) is removed

def bubble_ratio(volume, total_bubble_volume, voxel_dimensions=(1,1,1), bubble_class=1, background_class=0): #needs to be revised, I think it takes the values of the whole cell and not only the right side
    # Count the number of bubble voxels
    num_bubble_voxels = total_bubble_volume
    num_background_voxels = np.sum(volume == background_class)
    # Calculate the volume of a single voxel
    voxel_volume = voxel_dimensions[0] * voxel_dimensions[1] * voxel_dimensions[2]

    # Calculate the total volume of the bubbles
    total_bubble_volume = num_bubble_voxels * voxel_volume
    total_background_volume = num_background_voxels * voxel_volume

    bubble_background_ratio = total_bubble_volume / total_background_volume
    print("Bubble background ratio:", bubble_background_ratio, "voxels") # Replace 'cubic units' with appropriate unit (e.g., cubic millimeters)
    return bubble_background_ratio

def wall_proximity(args): #first we map the boundaries of the membrane, then we know the volume limits and we compare the centroid coordinates with the plane coordinates of the boundaries
    #map the mebrane surface in a matrix and then compare it to the centroid to accelerate the process

    centroid, volume, membrane_coordinates, membrane_class, tree = args
    # Dimensions of the volume
    depth, height, width = volume.shape
    
    # Membrane distance
    # Query the tree for the closest point to the centroid
    min_dist, index = tree.query(centroid)

    # Dictionary to store distances for each bubble
    distances = {}
    # Calculate distances to each boundary
    distance_to_top = height - centroid[1]
    distance_to_bottom = centroid[1]
    distance_to_left = centroid[2]
    distance_to_right = width - centroid[2]
    distance_to_front = centroid[0]
    distance_to_back = depth - centroid[0]
    distance_to_membrane = min_dist

    # Store distances in a dictionary
    distances = {
        'top': distance_to_top,
        'bottom': distance_to_bottom,
        'left': distance_to_left,
        'right': distance_to_right,
        'front': distance_to_front,
        'back': distance_to_back,
        'membrane': distance_to_membrane
    }

    # Find the closest boundary and its distance
    closest_boundary = min(distances, key=distances.get)
    closest_distance = distances[closest_boundary]
    #print('closest distance', closest_distance, closest_boundary)

    return closest_distance, closest_boundary

def membrane_block_visualization(volume, filtered_volume, csv_file, membrane_coords, bubble_class=1):
  # Read the CSV file
    csv_data = pd.read_csv(csv_file)

    # Iterate over each row and check if the bubble blocks the membrane, if yes keep it, if not remove it
    # Iterate over each row in the CSV file

    # for index, row in csv_data.iterrows():
    #     if row["closest_wall"] == "membrane" and row["closest_distance"]:
    #         filtered_volume[filtered_volume == row["label"]] = 1           #It just plots the bubble with nearest_wall = membrane, not the ones touching. We can save the position of the touching voxels and then compare it to the labeled volume  and just show the ones toucheing the membrane
    #     else: filtered_volume[filtered_volume == row["label"]] = 0

    # # Adjust each each class in order to have the same classes as the original volume and visualize it
    # # Adjust class assignments to match the original volume
    # for class_label in np.unique(volume):
    #     if class_label not in [0, 1]:  # Assuming 0 is background, 1 is bubble
    #         filtered_volume[volume == class_label] = class_label

    # # Check each pixel of the membrane if they have a neighbor of class bubble
    # print("Blocking bubble visualization")
    # visualize_volume(filtered_volume)

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
    visualize_volume(new_volume)

    return blocking_voxel

def individual_analysis(volume, membrane_class=2):

    # Label the volume into individual bubbles
    labeled_volume, num_features = label_bubbles(volume)
    print("Starting number of bubbbles:", num_features)
    filtered_volume = np.zeros_like(labeled_volume)

    ### FOR WALL PROXIMITY ###
    #Find the membrane coordinates
    membrane_voxels = np.where(volume == membrane_class)

    membrane_coordinates = list(zip(membrane_voxels[0], membrane_voxels[1], membrane_voxels[2]))

    # Create a k-D tree from membrane coordinates
    tree = cKDTree(membrane_coordinates)

    #########################################

    # Calculate region properties for each bubble
    props = regionprops(labeled_volume)

    print("Number of bubbles", len(props))

    label_list = []
    volume_list = []
    theta_list = []
    phi_list = []    
    sphericity_list = []
    solidity_list = []
    elongation_list = []
    major_axis_list = []
    minor_axis_list =[]
    centroid_list = []
    membrane_block_list = []

    passed_objects = 1

    # Iterate over each bubble and calculate desired properties
    for prop in props:
        labelid = prop.label
        print(labelid)
        # Volume (assuming 1 voxel = 1 unit volume)
        bubble_volume = prop.area  # For 3D images, 'area' gives the volume
        print("bubble volume ", bubble_volume)
        
        if bubble_volume < 60:
            continue
        

        #### Sphericity calculation ####
        # Create a binary mask of the current region
        binary_mask = labeled_volume == prop.label
        

        ## MARCHING CUBES
        #Apply the Marching Cubes algorithm
        verts, faces, _, _ = marching_cubes(binary_mask, level=0)

        # # Calculate surface area
        surface_area = mesh_surface_area(verts, faces)

        # Calculate sphericity
        if surface_area > 0:
            sphericity = (np.pi ** (1/3)) * ((6 * bubble_volume) ** (2/3)) / surface_area
        else:
            sphericity = 0  # To handle the case where surface_area is 0

        if sphericity > 1:
            continue

        print("Sphericity: ", sphericity)


        ##### Orientation calculation #####
        # Get the coordinates of all voxels in the region
        coords = prop.coords # remove if we keep solidity

        # Calculate the covariance matrix
        cov_matrix = np.cov(coords, rowvar=False)

        # Eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # The eigenvector corresponding to the largest eigenvalue
        principal_orientation = eigenvectors[:, np.argmax(eigenvalues)] 

        x, y, z = principal_orientation
        r = np.linalg.norm(principal_orientation)
        theta = np.arctan2(y, x)  # Azimuthal angle
        phi = np.arccos(z / r) if r != 0 else 0  # Polar angle

        theta = np.degrees(theta)
        phi = np.degrees(phi)

        print(principal_orientation)


        ##### Elongation calculation #####
        # Sort eigenvalues in descending order
        sorted_eigenvalues = sorted(eigenvalues, reverse=True)

        # Calculate elongation as the ratio of the largest to smallest eigenvalue
        if sorted_eigenvalues[-1] != 0:
            elongation = sorted_eigenvalues[0] / sorted_eigenvalues[-1]
        else:
            elongation = float(0)  # Handle division by zero
        print("elongation: ", elongation)

        ##### Centroid calculation #####

        centroid = prop.centroid

        ###############################
        
        ###Append the values to the corresponding list if the ROI is suitible for analysis

        centroid_list.append(centroid)
        volume_list.append(bubble_volume)
        #solidity_list.append(solidity)
        sphericity_list.append(sphericity)
        theta_list.append(theta)
        phi_list.append(phi)
        elongation_list.append(elongation)
        
        ### Add the ROI to the filtered visaulization ###
        filtered_volume[labeled_volume == prop.label] = passed_objects
        label_list.append(passed_objects)
        passed_objects += 1


    ### Mulithreading approach to calculate the wall proximity of the bubbles an to which wall ###
    #Prepare arguments for each call to wall_proximity
    args_list = [(centroid, volume, membrane_coordinates, membrane_class, tree) for centroid in centroid_list]

    with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust the number of workers as needed
        wall_proximity_list = list(executor.map(wall_proximity, args_list))
    print(wall_proximity_list)

    closest_distance = []
    closest_wall = []

    for distance, wall in wall_proximity_list:
        closest_distance.append(distance)
        closest_wall.append(wall)

        # Printing the properties (or store them in a list/dict for further analysis) orientation removed for testing
    #print(f"Volume: {volume_list}, Sphericity: {sphericity_list}, ")
            #f"Solidity: {solidity}, Elongation: {elongation}, Centroid: {centroid}")

    #### Membrane blocking value extraction ####
    for item in wall_proximity_list:
        if item[1] == "membrane" and item[0]<15:
            membrane_block_list.append(True)
        else:membrane_block_list.append(False)



    ### Total volume calculation and bubble/background ratio
    total_bubble_volume = sum(volume_list)

    print("Total volume from the individual measurements:", total_bubble_volume)

    bubble_ratio(volume, total_bubble_volume)

    ### Visualization of the filtered volume
    print(len(np.unique(filtered_volume)- 1))
    visualize_labeled_volume(filtered_volume, len(np.unique(filtered_volume) - 1))

    ### Write the CSV file with the result lists

    headers = ["label","volume","sphericity","theta", "phi","elongation", "centroid","closest_distance", "closest_wall"]


    data = list(zip(label_list, volume_list, sphericity_list, theta_list, phi_list, elongation_list, centroid_list, closest_distance, closest_wall))


    # Specify the filename
    filename = f"./output.csv"

    # Write to CSV
    with open(filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write the headers
        csvwriter.writerow(headers)
        
        # Write the data
        csvwriter.writerows(data)

    return filtered_volume, membrane_coordinates


volume = tifffile.imread('C:/Users/a.colliard/Desktop/zeis_imgs/mask2_reordered.tif')
print("Volume load done!")

left_volume, right_volume = separate_volume(volume)
print("Separation done!")

# clean the volume
cleaned_volume = clean_volume(left_volume)
print("Cleaning volume done!")


#bubble_ratio(cleaned_volume)
filtered_volume, membrane_coords = individual_analysis(cleaned_volume)

csv_file = "./output.csv"

visualize_property("closest_distance", filtered_volume, csv_file)
visualize_property("volume", filtered_volume, csv_file)
visualize_property("sphericity", filtered_volume, csv_file)

#visualize_labeled_volume()
blocking_voxel = membrane_block_visualization(volume, filtered_volume, csv_file, membrane_coords)
