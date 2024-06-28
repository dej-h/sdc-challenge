""" This file contains functions to generate a dataset from a given set of images and labels.

    Returns:
        A dataset with augmented images and labels.
"""

import random
from dataclasses import dataclass
import numpy as np
import cv2
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import queue
from threading import Thread, Lock
import time
import yaml

@dataclass
class ImageData:
    image_name: str
    image: np.ndarray
    bboxes: list  # List of lists for bounding boxes



"""This function crops the image and adjusts the bounding boxes based on the annotations.
"""

def crop_and_downsample(image, bboxes, target_size=(442, 442)):
    img_height, img_width = image.shape[:2]
    half_width = img_width // 2

    # Check if all labels are in the right 50% of the image (Including the edge of the annotation)
    all_in_right_half = all((bbox[1] - (bbox[3] / 2)) > 0.5 for bbox in bboxes)
    cropped_img = image.copy()

    if all_in_right_half:
        # Crop the right half of the image
        cropped_img = image[:, half_width:]
        offset = (half_width, 0, cropped_img.shape[1], cropped_img.shape[0])
    else:
        # Find the minimum x_center and corresponding left edge of the bounding box
        min_x_center = min(bbox[1] * img_width for bbox in bboxes)
        min_x_left = min((bbox[1] - (bbox[3] / 2)) * img_width for bbox in bboxes)
        
        # Ensure the crop_start includes the left side of the corresponding bounding box
        crop_start = max(int(min_x_left), 0)
        cropped_img = image[:, crop_start:crop_start + half_width]
        offset = (crop_start, 0, cropped_img.shape[1], cropped_img.shape[0])
        #if crop_start == 0:
            #print("cropping from start of the image")

    #print(f"Offset: {offset}")

    # Downsample the cropped image
    downsampled_img = cv2.resize(cropped_img, target_size)
    
    # Calculate the scale factors
    scale_x = target_size[0] / cropped_img.shape[0]
    scale_y = target_size[1] / cropped_img.shape[1]

    # Adjust the bounding boxes
    adjusted_bboxes = []
    for bbox in bboxes:
        class_id, x_center, y_center, width, height = bbox
        # Normalize the coordinates based on the cropped image
        new_x_center = (x_center * img_width - offset[0]) / cropped_img.shape[1]
        new_y_center = (y_center * img_height - offset[1]) / cropped_img.shape[0]
        new_width = width * scale_x
        new_height = height * scale_y
        adjusted_bboxes.append([class_id, new_x_center, new_y_center, new_width, new_height])
    
    return downsampled_img, offset, cropped_img.shape[:2], cropped_img.copy(), adjusted_bboxes

    """This function zooms in on a specific area of the image based on the bounding box coordinates.
    """
def zoom_image(image, bbox, zoom_level):
    x_center, y_center, width, height = bbox
    img_height, img_width = image.shape[:2]
    image_ratio = img_width / img_height
    
    x_center = x_center * img_width
    y_center = y_center * img_height
    box_width = int(width * img_width)
    box_height = int(height * img_height)
    
    # Calculate the zoom factor
    zoom_factor = 1
    if zoom_level == 0:
        zoom_factor = 1
    else:
        zoom_factor = 1 + (zoom_level - 1) * 1.5

    new_width = int(img_width / zoom_factor)
    new_height = int(img_height / zoom_factor)
    
    # Ensure new dimensions maintain aspect ratio and don't cut out the bounding box
    if new_width < box_width or new_height < box_height or new_width > img_width or new_height > img_height:
        new_width = max(new_width, box_width)
        new_height = max(new_height, box_height)
        if img_width >= img_height:
            new_width = min(new_width, img_width)
            new_height = int(new_width / image_ratio)
        else:
            new_height = min(new_height, img_height)
            new_width = int(new_height * image_ratio)
    else:
        if new_width / new_height > image_ratio:
            new_width = int(new_height * image_ratio)
        else:
            new_height = int(new_width / image_ratio)
    
    # Calculate the top-left corner of the zoomed area
    x1 = int(x_center - new_width / 2)
    y1 = int(y_center - new_height / 2)
    
    # Ensure the zoomed area is within the image bounds
    x1 = max(0, min(x1, img_width - new_width))
    y1 = max(0, min(y1, img_height - new_height))
    
    # Ensure the bounding box is fully within the zoomed area
    if x_center - box_width / 2 < x1:
        x1 = int(x_center - box_width / 2)
    if x_center + box_width / 2 > x1 + new_width:
        x1 = int(x_center + box_width / 2 - new_width)
    if y_center - box_height / 2 < y1:
        y1 = int(y_center - box_height / 2)
    if y_center + box_height / 2 > y1 + new_height:
        y1 = int(y_center + box_height / 2 - new_height)
    
    # Ensure the zoomed area is within the image bounds after adjustments
    x1 = max(0, min(x1, img_width - new_width))
    y1 = max(0, min(y1, img_height - new_height))

    zoomed_img = image[y1:y1 + new_height, x1:x1 + new_width]
    
    # Final check if the zoomed area is cropped correctly
    if (img_height *0.9 < box_height) or (img_width * 0.9 < box_width):
        return image, (0, 0, img_width, img_height)
    
    return zoomed_img, (x1, y1, new_width, new_height)


    """This function adjusts the bounding boxes based on the new dimensions of the image.
    """
def adjust_annotations(bboxes, old_dimensions, offset):
    x_offset, y_offset, new_width, new_height = offset
    #print (f"x_offset: {x_offset}, y_offset: {y_offset}, new_width: {new_width}, new_height: {new_height}")
    
    original_width, original_height = old_dimensions
    #print(f"Original width: {original_width}, Original height: {original_height}")
    
    adjusted_bboxes = []
    if new_width == 0 or new_height == 0:
        print(" width or height is 0")
    for bbox in bboxes:
        class_id, x_center, y_center, width, height = bbox
        abs_x_center = x_center * original_width
        abs_y_center = y_center * original_height
        abs_width = width * original_width
        abs_height = height * original_height

        x1 = abs_x_center - abs_width / 2
        y1 = abs_y_center - abs_height / 2
        x2 = x1 + abs_width
        y2 = y1 + abs_height

        # Calculate the area of the bounding box
        bbox_area = abs_width * abs_height

        # Calculate the area of the bounding box within the zoomed area
        x1_clipped = max(x1, x_offset)
        y1_clipped = max(y1, y_offset)
        x2_clipped = min(x2, x_offset + new_width)
        y2_clipped = min(y2, y_offset + new_height)

        clipped_width = x2_clipped - x1_clipped
        clipped_height = y2_clipped - y1_clipped
        clipped_area = clipped_width * clipped_height

        # Check if more than 80% of the bounding box is out of the frame
        if (clipped_area / bbox_area >= 0.6) and (clipped_width > 5) and (clipped_height > 5):
            new_x2 = max(x2_clipped - x_offset, 0)
            new_y2 = max(y2_clipped - y_offset, 0)
            new_x1 = min(x1_clipped - x_offset, new_width)
            new_y1 = min(y1_clipped - y_offset, new_height)

            new_x_center = (new_x1 + new_x2) / 2 / new_width
            new_y_center = (new_y1 + new_y2) / 2 / new_height
            adjusted_new_width = (new_x2 - new_x1) / new_width
            adjusted_new_height = (new_y2 - new_y1) / new_height

            if adjusted_new_width < 0 or adjusted_new_height < 0 or new_x_center < 0 or new_y_center < 0:
                print("negative lengths or centers")
                
            adjusted_bboxes.append([class_id, new_x_center, new_y_center, adjusted_new_width, adjusted_new_height])
            
    #print("Adjusted bboxes: ", adjusted_bboxes)

    return adjusted_bboxes

    """This function draws bounding boxes on the image.
    """
def draw_bounding_boxes(image, bboxes, img_dimensions):
    img_width, img_height = img_dimensions
    for bbox in bboxes:
        class_id, x_center, y_center, width, height = bbox
        x_center = int(x_center * img_width)
        y_center = int(y_center * img_height)
        width = int(width * img_width)
        height = int(height * img_height)

        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = x1 + width
        y2 = y1 + height

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(int(class_id)), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

    """This function applies augmentations to the image.
    """
def apply_augmentations(image):
    # Apply blur
    blur_levels = [3,7]
    blurred_images = [cv2.GaussianBlur(image, (k, k), 0) for k in blur_levels]

    # Apply brightness/contrast adjustments
    # generate the brightness and contrast levels randomly between 30 and 60
    rN1 = np.random.randint(30, 60)
    rN2 = np.random.randint(30, 60)
    rN3 = np.random.randint(-60, -30)
    rN4 = np.random.randint(-60, -30)
    
    brightness_contrast_levels = [(30, 30),(-30,-30),(0,0)]
    brightness_contrast_images = []
    for (brightness, contrast) in brightness_contrast_levels:
        new_image = np.int16(image)
        new_image = new_image * (contrast / 127 + 1) - contrast + brightness
        new_image = np.clip(new_image, 0, 255)
        new_image = np.uint8(new_image)
        brightness_contrast_images.append(new_image)

    augmented_images = blurred_images + brightness_contrast_images

    return augmented_images

    """This function prints a progress bar in the terminal.
    """
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()

    """This function clears and creates the output folders.
    """
def clear_and_create_output_folders(output_folder):
    output_images_folder = os.path.join(output_folder, 'images')
    output_labels_folder = os.path.join(output_folder, 'labels')
    output_boxed_images_folder = os.path.join(output_folder, 'boxed_images')
    
    for folder in [output_images_folder, output_labels_folder, output_boxed_images_folder]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                os.remove(os.path.join(folder, file))
        os.makedirs(folder, exist_ok=True)
    print("Cleansed and created output folders")

    """This function preloads the data from the images and labels folders.
    """
def preload_data(images_folder, labels_folder):
    data = []
    class_counts = defaultdict(int)

    for image_name in os.listdir(images_folder):
        if not image_name.endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(images_folder, image_name)
        label_path = os.path.join(labels_folder, image_name.rsplit('.', 1)[0] + '.txt')

        image = cv2.imread(image_path)
        with open(label_path, 'r') as file:
            bboxes = [list(map(float, line.split())) for line in file.readlines()]

        data.append(ImageData(image_name=image_name, image=image, bboxes=bboxes))

        for bbox in bboxes:
            class_counts[bbox[0]] += 1

    return data, class_counts

    """This function balances the classes in the dataset by duplicating images.
    """
def balance_classes(data, class_counts, tolerance=0.1, max_duplicates=3, debug=True):
    max_class_count = max(class_counts.values())
    target_count = int(max_class_count * (1 - tolerance))

    # Find the highest current index for image naming
    existing_indices = [int(img.image_name.split('.')[0]) for img in data]
    next_index = max(existing_indices) + 1

    # Track the number of times each image has been duplicated
    duplication_counts = defaultdict(int)
    original_data = data.copy()

    while True:
        underrepresented_classes = {cls: count for cls, count in class_counts.items() if count < target_count}
        
        if not underrepresented_classes:
            break

        # Find overrepresented classes
        overrepresented_classes = {cls: count for cls, count in class_counts.items() if count > target_count}

        # Select images containing underrepresented classes and filter out those with overrepresented classes
        selectable_images = [img for img in original_data if any(bbox[0] in underrepresented_classes for bbox in img.bboxes) and not any(bbox[0] in overrepresented_classes for bbox in img.bboxes)]
        
        if not selectable_images:
            print("Too many duplicates or no valid images available. Cannot balance classes further.")
            break
        
        # Remove over-duplicated images from the selectable pool
        selectable_images = [img for img in selectable_images if duplication_counts[img.image_name] < max_duplicates]

        if not selectable_images:
            print("All images have reached the maximum duplication limit. Cannot balance classes further.")
            break

        # Select a random image from the filtered list
        selected_image = random.choice(selectable_images)
        
        # Generate the next highest index for the new image name
        new_image_name = f"{next_index:05d}.png"
        new_image = ImageData(image_name=new_image_name, image=selected_image.image.copy(), bboxes=selected_image.bboxes.copy())

        data.append(new_image)

        for bbox in new_image.bboxes:
            class_counts[bbox[0]] += 1

        # Increment the duplication count for the selected image
        duplication_counts[selected_image.image_name] += 1

        next_index += 1

    if debug:
        # Plot the number of duplicates for each original image
        plt.figure(figsize=(10, 6))
        plt.bar(duplication_counts.keys(), duplication_counts.values())
        plt.xlabel('Image Names')
        plt.ylabel('Number of Duplicates')
        plt.title('Number of Duplicates for Each Original Image')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    return data, class_counts

"""This function converts a list of data to a queue.
"""
def data_list_to_queue(data_list):
    data_queue = queue.Queue()
    for item in data_list:
        data_queue.put(item)
    return data_queue

    """This function processes images and labels in the input folders
    and saves the augmented images and labels in the output folders.
    """
# def process_images(images_folder, labels_folder, output_folder):
    #Opening the path to the output folders
    output_images_folder = os.path.join(output_folder, 'images')
    output_labels_folder = os.path.join(output_folder, 'labels')
    output_boxed_images_folder = os.path.join(output_folder, 'boxed_images')
    
    # calculate image folder size
    image_folder_size = len([name for name in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, name))])
    print(f"Amount of pictures in the input directory: {image_folder_size}")
    k = 0
    
    #Go through all the images in the folder
    for image_name in os.listdir(images_folder):
        k = k + 1
        # Check for correct file type
        if not image_name.endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Get the image and label paths
        image_path = os.path.join(images_folder, image_name)
        label_path = os.path.join(labels_folder, image_name.rsplit('.', 1)[0] + '.txt')

        #Read the image
        image = cv2.imread(image_path)
        # Split the image through the middle into 2 images in the x-axis
        
        # Read the labels
        with open(label_path, 'r') as file:
            bboxes = [list(map(float, line.split())) for line in file.readlines()]
        
        #Down sample the image and adjust the inital bounding boxes
        downsampled_image,offset_crop,new_dimensions,cropped_image,cropped_bboxes = crop_and_downsample(image, bboxes)
        
        #Test functions
        """
        #cropped_bboxes = adjust_annotations(bboxes, (img_width, img_height),  offset_crop)
            #show the normal cropped image with bboxes
            # boxed_image = draw_bounding_boxes(cropped_image.copy(), cropped_bboxes, (new_dimensions_crop[0], new_dimensions_crop[1]))
            # cv2.imshow("Cropped Image", boxed_image)
            # while True:
            #     if cv2.waitKey(0) & 0xFF == ord('q'):
            #         break
            
            #show the downsampled image with bboxes
            # boxed_image = draw_bounding_boxes(downsampled_image.copy(), cropped_bboxes, (downsampled_image.shape[0], downsampled_image.shape[1]))
            # cv2.imshow("Downsampled Image", boxed_image)
            # while True:
            #     if cv2.waitKey(0) & 0xFF == ord('q'):
            #         break
        """
       # Initlaize a list of already handled classes
        handled_classes = []
       
        # Go through every bbox in the images and apply the zoom function
        for i, bbox in enumerate(cropped_bboxes):
            class_id, x_center, y_center, width, height = bbox
            
            #Generate zoomed images for each class
            for zoom_level in range(0, 3):
                zoomed_img, offset = zoom_image(downsampled_image, [x_center, y_center, width, height], zoom_level)
                adjusted_bboxes = adjust_annotations(cropped_bboxes, (downsampled_image.shape[0], downsampled_image.shape[1]), offset)

                augmented_images = apply_augmentations(zoomed_img)

                for idx, aug_img in enumerate([zoomed_img] + augmented_images):
                    # Check if the class_id is in handled_classes and adjust it
                    if class_id in handled_classes:
                        class_id = round(class_id + 0.1, 1)

                    # Format the class_id as a float with one decimal place in the file names
                    new_image_name = f"{image_name.rsplit('.', 1)[0]}_{class_id:.1f}_{zoom_level}_aug{idx}.png"
                    new_label_name = f"{image_name.rsplit('.', 1)[0]}_{class_id:.1f}_{zoom_level}_aug{idx}.txt"


                    new_image_path = os.path.join(output_images_folder, new_image_name)
                    new_label_path = os.path.join(output_labels_folder, new_label_name)
                    boxed_image_path = os.path.join(output_boxed_images_folder, new_image_name)

                    cv2.imwrite(new_image_path, aug_img)

                    boxed_image = draw_bounding_boxes(aug_img.copy(), adjusted_bboxes, (zoomed_img.shape[0], zoomed_img.shape[1]))
                    cv2.imwrite(boxed_image_path, boxed_image)

                    with open(new_label_path, 'w') as file:
                        for new_bbox in adjusted_bboxes:
                            file.write(' '.join(map(str, new_bbox)) + '\n')

                    # print(f"Saved augmented image {new_image_path} and label {new_label_path}")
                    # print(f"Saved boxed image {boxed_image_path}")
            
            #Append the handled class to the list        
            handled_classes.append(class_id)
        # print the progress bar
        print_progress_bar(k,image_folder_size,prefix='Creating enhanced Database:',suffix='Complete',length=50)
                    
    print(f"Amount of pictures in the output directory: {len(os.listdir(output_images_folder))}")

""" This function processes images using multithreading.
"""
def worker(data_queue, output_folder):
    output_images_folder = os.path.join(output_folder, 'images')
    output_labels_folder = os.path.join(output_folder, 'labels')
    output_boxed_images_folder = os.path.join(output_folder, 'boxed_images')

    while not data_queue.empty():
        try:
            item = data_queue.get_nowait()
        except queue.Empty:
            break

        image_name = item.image_name
        image = item.image
        bboxes = item.bboxes

        downsampled_image, offset_crop, new_dimensions, cropped_image, cropped_bboxes = crop_and_downsample(image, bboxes)

        handled_classes = []

        for i, bbox in enumerate(cropped_bboxes):
            class_id, x_center, y_center, width, height = bbox

            for zoom_level in range(1, 3):
                zoomed_img, offset = zoom_image(downsampled_image, [x_center, y_center, width, height], zoom_level)
                adjusted_bboxes = adjust_annotations(cropped_bboxes, (downsampled_image.shape[0], downsampled_image.shape[1]), offset)

                augmented_images = apply_augmentations(zoomed_img)

                for idx, aug_img in enumerate([zoomed_img] + augmented_images):
                    if class_id in handled_classes:
                        class_id = round(class_id + 0.1, 1)

                    new_image_name = f"{image_name.rsplit('.', 1)[0]}_{class_id:.1f}_{zoom_level}_aug{idx}.png"
                    new_label_name = f"{image_name.rsplit('.', 1)[0]}_{class_id:.1f}_{zoom_level}_aug{idx}.txt"

                    new_image_path = os.path.join(output_images_folder, new_image_name)
                    new_label_path = os.path.join(output_labels_folder, new_label_name)
                    boxed_image_path = os.path.join(output_boxed_images_folder, new_image_name)

                    cv2.imwrite(new_image_path, aug_img)

                    boxed_image = draw_bounding_boxes(aug_img.copy(), adjusted_bboxes, (zoomed_img.shape[0], zoomed_img.shape[1]))
                    cv2.imwrite(boxed_image_path, boxed_image)

                    with open(new_label_path, 'w') as file:
                        for new_bbox in adjusted_bboxes:
                            file.write(' '.join(map(str, new_bbox)) + '\n')

            handled_classes.append(class_id)

        data_queue.task_done()

    """This function prints the class counts.
    """
def print_class_counts(class_counts, target_count, tolerance=0.1):
    print("Class count evaluation:")
    for class_id, count in class_counts.items():
        class_name = CLASS_NAMES.get(class_id, f'Unknown-{class_id}')
        status = []
        if count < target_count*(1-tolerance):
            status.append("under-represented")
        if count > target_count*(1+tolerance):
            status.append("over-represented")
        status_str = ', '.join(status) if status else "balanced"
        print(f"{class_name}: {count} ({status_str})")
    print()

    """This function processes images using multithreading.
    """
def process_images_multithreaded(data_list, output_folder, num_threads=4):
    # Create necessary output subfolders
    output_images_folder = os.path.join(output_folder, 'images')
    output_labels_folder = os.path.join(output_folder, 'labels')
    output_boxed_images_folder = os.path.join(output_folder, 'boxed_images')

    for folder in [output_images_folder, output_labels_folder, output_boxed_images_folder]:
        os.makedirs(folder, exist_ok=True)

    # Convert data list to queue
    data_queue = data_list_to_queue(data_list)

    # Progress tracking
    total_items = data_queue.qsize()

    threads = []
    for _ in range(num_threads):
        thread = Thread(target=worker, args=(data_queue, output_folder))
        thread.start()
        threads.append(thread)

    while any(thread.is_alive() for thread in threads):
        remaining_items = data_queue.qsize()
        processed_items = total_items - remaining_items
        print_progress_bar(processed_items, total_items, prefix='Progress:', suffix='Complete', length=50)
        time.sleep(1)  # Update the progress bar every second

    for thread in threads:
        thread.join()

    
    # print the total amount of labeled images
    print(f"Amount of pictures in the output directory: {len(os.listdir(output_images_folder))}")

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
        
# Example usage
data_file = './sdc-challenge/DataSet/PreparedDataset/data.yaml'
images_folder = './sdc-challenge/DataSet/PreparedDataset/train/images'
labels_folder = './sdc-challenge/DataSet/PreparedDataset/train/labels'
output_folder = './sdc-challenge/DataSet/AugmentedDataset'
clear_and_create_output_folders(output_folder)
debug = False

# Load the data.yaml file
with open(data_file, 'r') as file:
    data_yaml = yaml.safe_load(file)

# Extract class names from the data.yaml file
class_names_from_yaml = data_yaml.get('names', [])

# Create a new dictionary to update CLASS_NAMES
CLASS_NAMES = {i: name for i, name in enumerate(class_names_from_yaml)}


data, class_counts = preload_data(images_folder, labels_folder)
print("Data preloaded")
target_count = max(class_counts.values())
if debug:
    print_class_counts(class_counts,target_count, tolerance=0.1)


rebalanced_data, rebalanced_class_counts = balance_classes(data, class_counts, tolerance=0.1, max_duplicates=4, debug=debug)
print("Data rebalanced")
if debug:
    print_class_counts(rebalanced_class_counts,target_count, tolerance=0.1)

# Copy over the yaml data file
new_data_path = os.path.join(output_folder, 'data.yaml')
update_yaml_file(data_file, new_data_path, output_folder)


# Process images using multithreading
process_images_multithreaded(rebalanced_data, output_folder, num_threads=8)
