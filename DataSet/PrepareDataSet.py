import os
import shutil

def get_starting_index(folder):
    # Initialize the maximum index to 0
    max_index = 0
    
    # Iterate through each file in the folder
    for filename in os.listdir(folder):
        # Split the filename by the dot to extract the base part
        base_name = os.path.splitext(filename)[0]
        
        # Check if this base name is numeric
        if base_name.isdigit():
            # Update max_index with the maximum value
            max_index = max(max_index, int(base_name))
    
    # Return the next index to be used
    return max_index + 1

def rename_files(images_folder, annotations_folder, output_folder, data_file):
    # Copy over the data file
    new_data_path = os.path.join(output_folder, 'data.yaml')
    shutil.copyfile(data_file, new_data_path)
    
    # Create the necessary directories
    output_folder = os.path.join(output_folder, 'train')
    output_images_folder = os.path.join(output_folder, 'images')
    output_labels_folder = os.path.join(output_folder, 'labels')
    
    # Ensure the output directories exist
    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)
    
    # List and sort the image and annotation files
    image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.png') or f.endswith('.jpg')])
    annotation_files = sorted([f for f in os.listdir(annotations_folder) if f.endswith('.xml') or f.endswith('.txt')])
    
    # Check if the number of image and annotation files are equal
    if len(image_files) != len(annotation_files):
        print("Error: The number of images and annotations files are not the same.")
        return
    
    # Get the starting index for new files based on existing files in the output folders
    start_index = max(get_starting_index(output_images_folder), get_starting_index(output_labels_folder))
    
    # Rename and move the files
    for idx, (image_file, annotation_file) in enumerate(zip(image_files, annotation_files), start=start_index):
        # Format the new base name with leading zeros
        new_name = f"{idx:05d}"
        
        # Extract the file extensions
        image_ext = os.path.splitext(image_file)[1]
        annotation_ext = os.path.splitext(annotation_file)[1]
        
        # Define the new file paths in the output folders
        new_image_path = os.path.join(output_images_folder, f"{new_name}{image_ext}")
        new_annotation_path = os.path.join(output_labels_folder, f"{new_name}{annotation_ext}")
        
        # Copy the files from the source to the destination
        shutil.copyfile(os.path.join(images_folder, image_file), new_image_path)
        shutil.copyfile(os.path.join(annotations_folder, annotation_file), new_annotation_path)
        
        # Output the changes to the console
        print(f"Copied {image_file} to {new_image_path}")
        print(f"Copied {annotation_file} to {new_annotation_path}")

# Example usage
images_folder = './RobotFlow_TrainSet/V5/train/images'  # Adjust with your images folder path
annotations_folder = './RobotFlow_TrainSet/V5/train/labels'  # Adjust with your annotations folder path
data_file = './RobotFlow_TrainSet/V5/data.yaml'  # Adjust with your yaml folder path
output_folder = './RobotFlow_TrainSet_ProperName/V5'  # Adjust with your desired output folder path

rename_files(images_folder, annotations_folder, output_folder, data_file)
