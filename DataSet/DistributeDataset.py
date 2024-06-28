"""Distribute the dataset into train, test, and validation sets.
It needs a folder of images and annotations.
Where the images are in a folder named 'images' and the annotations are in a folder named 'labels'.
The function will create a new folder structure with the following format:
OUTPUT_FOLDER -> train -> images
                        -> labels
                    test -> images
                        -> labels
                    valid -> images
                        -> labels
"""


import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
import zipfile
import yaml


def distribute_dataset(source_folder, destination_folder, split_ratio=(0.7, 0.2, 0.1)):
    
    # Check if the split ratio is valid
    assert round(sum(split_ratio),10) == 1, "The split ratio must sum up to 1"
    
    # Copy over the yaml file with the correct directory
    update_yaml_file(os.path.join(source_folder, 'data.yaml'), os.path.join(destination_folder, 'data.yaml'), destination_folder)
    
    # Copy over the yaml file with the correct directory
    # Define the subfolders for images and labels within each dataset split
    subfolders = ['images', 'labels']

    # Create the structure for train, test, and validation sets within the destination folder
    dataset_splits = ['train', 'test', 'valid']
    folders_structure = {split: {sub: os.path.join(destination_folder, split, sub) for sub in subfolders} for split in dataset_splits}

    # Initialize counters for the number of images in each split
    image_count = {'train': 0, 'test': 0, 'valid': 0}

    # Create the directories if they don't exist and clear them if they do
    for split, paths in folders_structure.items():
        for sub, path in paths.items():
            os.makedirs(path, exist_ok=True)
            # Clear existing files to prevent mix-ups
            for file in os.listdir(path):
                os.remove(os.path.join(path, file))

    # Prepare to distribute images and labels
    images_folder = os.path.join(source_folder, 'images')
    labels_folder = os.path.join(source_folder, 'labels')

    # Dictionary to keep track of image data by class
    class_images = {}

    # Group images by class
    for image_name in os.listdir(images_folder):
        if not image_name.endswith(('.png', '.jpg', '.jpeg')):
            continue
        # Parse class from the filename (as float)
        class_id = float(image_name.split('_')[1])
        if class_id not in class_images:
            class_images[class_id] = []
        class_images[class_id].append(image_name)

    # Function to distribute files into train, test, val
    def distribute_files(files, class_id):
        # Split the files based on the given ratio
        train_files, test_files = train_test_split(files, test_size=(1 - split_ratio[0]), random_state=42)
        test_files, valid_files = train_test_split(test_files, test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]), random_state=42)
        
        # Helper to copy files to target folder
        def copy_files(file_list, target_folder_images, target_folder_labels, split_name):
            for file in file_list:
                # Copy image
                shutil.copy2(os.path.join(images_folder, file), os.path.join(target_folder_images, file))
                # Copy corresponding label
                label_file = file.rsplit('.', 1)[0] + '.txt'
                shutil.copy2(os.path.join(labels_folder, label_file), os.path.join(target_folder_labels, label_file))
                # Increment the counter
                image_count[split_name] += 1
                print("image count: ", sum(image_count.values()))

        # Distribute files
        copy_files(train_files, folders_structure['train']['images'], folders_structure['train']['labels'], 'train')
        copy_files(test_files, folders_structure['test']['images'], folders_structure['test']['labels'], 'test')
        copy_files(valid_files, folders_structure['valid']['images'], folders_structure['valid']['labels'], 'valid')

    # Distribute files for each class
    for class_id, files in class_images.items():
        distribute_files(files, class_id)

    # Print the distribution count
    print(f"Distributed files into {destination_folder} with the following counts:")
    print(f"Train: {image_count['train']} images")
    print(f"Test: {image_count['test']} images")
    print(f"Valid: {image_count['valid']} images")

    # # Create a zip archive of the distributed dataset
    # zip_filename = f"{destination_folder}.zip"
    # with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    #     for root, dirs, files in os.walk(destination_folder):
    #         for file in files:
    #             abs_path = os.path.join(root, file)
    #             rel_path = os.path.relpath(abs_path, os.path.join(destination_folder, '..'))
    #             zipf.write(abs_path, rel_path)
    
    # print(f"Created zip archive: {zip_filename}")

def update_yaml_file(input_yaml_path, output_yaml_path, base_path):
    with open(input_yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    # Update the val path
    data['val'] = base_path + '/valid/images'
    
    # Update the train path
    data['train'] = base_path + '/train/images'
    
    # Update the test path
    data['test'] = base_path + '/test/images'

    # Write the updated data to the new yaml file
    with open(output_yaml_path, 'w') as file:
        yaml.dump(data, file)

# Example Usage
# Define the source and destination folders
source_folder = './sdc-challenge/DataSet/AugmentedDataset'
destination_folder = './sdc-challenge/DataSet/DistributedDataset'

# Define the split ratio for train, test, and validation sets
TRAIN_SPLIT = 0.7 # 70% of the data for training
TEST_SPLIT = 0.2 # 20% of the data for testing
VALID_SPLIT = 0.1 # 10% of the data for validation
# These have to sum up to 1

split_ratio = (TRAIN_SPLIT, TEST_SPLIT, VALID_SPLIT)
distribute_dataset(source_folder, destination_folder, split_ratio)
