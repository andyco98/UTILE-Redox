import imageio
import os

def create_gif(input_folder, output_filename, frame_duration=0.1):
    images = []
    # Ensure the files are sorted correctly
    file_names = sorted([img for img in os.listdir(input_folder) if img.endswith(".png")])

    for filename in file_names:
        file_path = os.path.join(input_folder, filename)
        images.append(imageio.imread(file_path))

    imageio.mimsave(output_filename, images, duration=frame_duration)

# Usage
input_folder = f'C:/Users/andre/Desktop/zeis/gifs/right_frames'  # Replace with your frames folder
output_filename = f'C:/Users/andre/Desktop/zeis/gifs/right_output.gif'  # The output file
frame_duration = 0.1  # Duration of each frame in the GIF in seconds
create_gif(input_folder, output_filename, frame_duration)

directory_folders = "C:/Users/andre/Desktop/zeis/gifs/"

# directory_list = os.listdir(directory_folders)

# for i in directory_list:
#     input_folder = f'C:/Users/andre/Desktop/zeis/gifs/' + i  # Replace with your frames folder
#     output_filename = f'C:/Users/andre/Desktop/zeis/{i}_output.gif'  # The output file
#     frame_duration = 0.1  # Duration of each frame in the GIF in seconds
#     create_gif(input_folder, output_filename, frame_duration)