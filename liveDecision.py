
"""
Created on Mon Jun 10 11:35:35 2024

@author: sdc
"""
import os
import shutil
import cv2
import numpy as np
from pathlib import Path
import torch
import time
import threading
import queue
import math
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from collections import deque
import  collections
from datetime import datetime
import json
import csv
import psutil
#Calculate distance based of detection and real life widths and heights
def estimate_distance(x1, y1, x2, y2, real_width, real_height, x_offset=424, y_offset=240, image_width=848, image_height=480,fov_based = False):
    # Constants for Logitech StreamCam
    camera_fov_h = 67.5  # Horizontal field of view in degrees
    camera_fov_v = 41.2  # Vertical field of view in degrees
    focal_length = 540   # focal length of the camera
    box_width = abs(x2 - x1)
    box_height = abs(y2 - y1)
    
    #Standard distance offset (distance from front of car to camera)
    camera_offset  = 0.3
    #Assure there is no division by 0
    if box_height <= 0:
        box_height = 1 
    if box_width <=0:
        box_width = 1
    #FOV based calculations
    if fov_based:
        # Translate bounding box coordinates to full image coordinates
        x1 += x_offset
        y1 += y_offset
        x2 += x_offset
        y2 += y_offset
    
        # Calculate the center and dimensions of the bounding box
        box_x_center = (x1 + x2) / 2
        box_y_center = (y1 + y2) / 2
        
    
        # Calculate the horizontal and vertical angles relative to the center of the image
        angle_h = (box_x_center - image_width / 2) * (camera_fov_h / image_width)
        angle_v = (box_y_center - image_height / 2) * (camera_fov_v / image_height)
    
        # Calculate the estimated distances using width and height
        est_w_d = (real_width * focal_length) / box_width
        est_h_d = (real_height * focal_length) / box_height
    
        # Adjust the estimated distances based on the angles
        adjusted_est_w_d = est_w_d / math.cos(math.radians(angle_h))
        adjusted_est_h_d = est_h_d / math.cos(math.radians(angle_v))
    
        # Average the distances if the entire object is within the FOV
        if box_width < box_height:
            distance = (adjusted_est_w_d + adjusted_est_h_d) / 2
        else:
            distance = adjusted_est_w_d
    
        return max(distance -camera_offset,0)
    #focal based calculations
    else:
        
        est_w_d = (focal_length * real_width)/box_width
        est_h_d = (focal_length * real_height)/box_height
        
        return max((est_w_d+ est_h_d)/2 - camera_offset,0)
    
    
    
# DETECTION PROCESSING FUNCTIONS

# This function makes a lot of predictions on where the objects could be
def run_inference(model, frame, device, stride=32):
    # Measure time for preprocessing
    start_preprocessing = time.time()
    
    # Resize the image to ensure its dimensions are multiples of the model's stride
    img_size = check_img_size(frame.shape[:2], s=stride)  # Ensure multiple of stride
    img = cv2.resize(frame, (img_size[1], img_size[0]))

    # Convert image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))  # Convert to [3, height, width]
    img = np.expand_dims(img, axis=0)  # Add batch dimension [1, 3, height, width]
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # Normalize to 0.0 - 1.0
    
    end_preprocessing = time.time()
    preprocessing_time = end_preprocessing - start_preprocessing
    #print(f"Preprocessing Time: {preprocessing_time:.4f} seconds /n")
    
    # Measure time for prediction
    start_prediction = time.time()
    
    with torch.no_grad():
        pred = model(img)
    
    end_prediction = time.time()
    prediction_time = end_prediction - start_prediction
    #print(f"Prediction Time: {prediction_time:.4f} seconds")
    
    return pred, img.shape

# This function takes the prediction and turns it processes it to object detection
def process_detections(pred, frame, img_shape, conf_thres=0.80, iou_thres=0.45, max_det=1000):
    
    det = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det)[0]
    detections = []

    if det is not None and len(det):
        det[:, :4] = scale_boxes(img_shape[2:], det[:, :4], frame.shape).round()
        for *xyxy, conf, cls in reversed(det):
            x1, y1, x2, y2 = map(int, xyxy)
            confidence = conf.item()
            class_id = int(cls.item())
            if class_id == 0:
                obj_class = 'Car'
            elif class_id == 1:
                obj_class = 'Person'
            elif class_id == 2:
                obj_class = 'Speed-limit-10km-h'
            elif class_id == 3:
                obj_class = 'Speed-limit-15km-h'
            elif class_id == 4:
                obj_class = 'Speed-limit-20km-h'
            elif class_id == 5:
                obj_class = 'Traffic Light Green'
            elif class_id == 6:
                obj_class = 'Traffic Light Red'
            else:
                obj_class = 'Unknown'

            detections.append({
                'class': obj_class,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            })

    return detections

 # This function takes the image and detections and returns a visual representation of the detections that it does, it also returns the detection that it does
   
def draw_bounding_boxes(image, detections, down_sample_factor,offset, focal_length):
    real_widths = {
        'Speed-limit-10km-h': 0.6,
        'Speed-limit-15km-h': 0.6,
        'Speed-limit-20km-h': 0.6,
        'Traffic Light Green': 0.07,
        'Traffic Light Red': 0.07,
        'Car': 2.5,
        'Person': 0.5,
        'Unknown': 0.5
    }
    real_heights = {
        'Speed-limit-10km-h': 0.6,
        'Speed-limit-15km-h': 0.6,
        'Speed-limit-20km-h': 0.6,
        'Traffic Light Green': 0.3,
        'Traffic Light Red': 0.3,
        'Car': 1.8,
        'Person': 2.0,
        'Unknown': 0.5
        }
    font_scale = 0.5
    thickness = 2
    
    detection_info = []
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        
        
        obj_class = det['class']
        confidence = det['confidence']
        label = f"{obj_class} {confidence*100:.1f}%"
        
        #Box width calculation
        real_width_m = real_widths.get(obj_class, 0.5)
        
        #Box height caclulation
        real_height_m = real_heights.get(obj_class,0.5)
        
        #Calculating the distance
        distance = estimate_distance(x1,y1,x2,y2,real_width_m,real_height_m)
        distance_label = f"{distance:.2f}m"
        t_x1 = x1 - offset[1]
        t_y1 = y1 - offset[0]
        t_x2 = x2 - offset[1]
        t_y2 = y2 - offset[0]
        
        #Draw it back on properly
        t_x1 = t_x1 // down_sample_factor
        t_y1 = t_y1 // down_sample_factor
        t_x2 = t_x2 // down_sample_factor
        t_y2 = t_y2 // down_sample_factor
        
        cv2.rectangle(image, (t_x1, t_y1), (t_x2, t_y2), (0, 255, 0), 2)
        cv2.putText(image, label, (t_x1, t_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
        cv2.putText(image, distance_label, (t_x1, t_y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (139, 0, 0), thickness)
        
        detection_info.append({
            'class': det['class'],
            'bbox': det['bbox'],
            'distance': distance
        })

    return image,detection_info
    

# MODEL INITIALIZATION FUCTIONS

#Loads up the base model
def load_model(weights, device):
    model = DetectMultiBackend(weights, device=device, fp16=False)  # Use fp16=False for CPU
    model.warmup()
    return model


#quantizes the model it has as input
def quantize_model(model):
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

#initializes the model  and returns the model, device and according directories
def initialize(weights_path, output_dir_base, no_gui_save_dir):
    # Base directory for saving outputs
    os.makedirs(output_dir_base, exist_ok=True)
    print("Output Directory Created")

    # Determine the next available directory for saving detections
    detections_dir = find_next_available_dir(output_dir_base, 'detections')
    os.makedirs(detections_dir, exist_ok=True)
    print(f"Detections Directory Created: {detections_dir}")

    # Directory to save images if GUI is not available
    os.makedirs(no_gui_save_dir, exist_ok=True)
    clear_directory(no_gui_save_dir)
    print("Output directory cleared")

    # Load model once
    device = select_device('CPU')
    model = load_model(weights_path, device)
    
   
    quantized_model = quantize_model(model)
    print("Done optimizing model")
    # Check if GUI is available
    gui_available = False
    try:
        print("still alive")
        cv2.imshow('Test', np.zeros((100, 100), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        gui_available = True
        print("staying aliveee")
    except:
        gui_available = True

    return quantized_model, device, detections_dir, no_gui_save_dir, gui_available


def process_image(model, device, input_image, gui_available):
    if isinstance(input_image, str):
        # If input is a string, treat it as a file path
        frame = cv2.imread(input_image)
        if frame is None:
            print(f"Failed to read the image at {input_image}. Skipping.")
            return None, None
    elif isinstance(input_image, np.ndarray):
        # If input is an image (numpy array), use it directly
        frame = input_image
    else:
        print("Invalid input type. Input should be either a file path (str) or an image (numpy array).")
        return None, None

    # Crop the top right portion of the image
    height, width, _ = frame.shape
    top_right_frame = frame[:height // 2, width // 2:]
    
    #Crop the top left portion of the image
    top_left_frame = frame[:height // 2, :width // 2]
    
    #Top middle portion of the iamge
    top_middle_frame = frame[:height // 2, width // 4: 3 * width // 4]
    
    #Down sample
    
    down_sample_factor = 2
    # Ensure the cropped image is resized to the desired dimensions (multiples of model's stride)
    stride = 32
    original_height, original_width = top_right_frame.shape[:2]
    desired_height = (((original_height // stride) + 1) * stride) // down_sample_factor
    desired_width = (((original_width // stride) + 1) * stride) // down_sample_factor
    
    #Resizing
    resized_top_right_frame = cv2.resize(top_right_frame, (desired_width, desired_height))
    
    #now for the left side 
    resized_top_left_frame = cv2.resize(top_left_frame, (desired_width, desired_height))
    
    #Resized middle frame
    resized_top_middle_frame = cv2.resize(top_middle_frame, (desired_width, desired_height))
    
    
    
    
    
    
    
    #Decision making from the self driving car
    detection_info = []
    right_detection_info = []
    left_detection_info = []
    middle_detection_info = []
    
    def apply_offsets(detection_info, offset):
            for det in detection_info:
                x1, y1, x2, y2 = det['bbox']
                x1 = x1 * down_sample_factor
                y1 = y1 * down_sample_factor
                x2 = x2 * down_sample_factor
                y2 = y2 * down_sample_factor
                
                x1 += offset[1]
                y1 += offset[0]
                x2 += offset[1]
                y2 += offset[0]
                det['bbox'] = (x1, y1, x2, y2)
            return detection_info
        
    def filter_edges(detection_info, x_value, p_treshold, width):
        max_x = min(x_value + (p_treshold*width),width)
        min_x = max(x_value - (p_treshold*width),0)
        
        for r_dets in detection_info:
            x1, y1, x2, y2 = r_dets['bbox']
            center_x = (x1+x2)/2
            if (center_x < max_x) and (center_x > min_x):
                detection_info.remove(r_dets)
        return detection_info
    
    
    
    if gui_available:
        
        #RIGHT
        # Process the resized cropped frame with YOLOv5 detection
        pred, img_shape = run_inference(model, resized_top_right_frame, device)
        detections_right = process_detections(pred, resized_top_right_frame, img_shape)
        
        #Apply offsets to the image coordinates for top left and top right
        top_right_offset = (0, width // 2)
        detections_right = apply_offsets(detections_right, top_right_offset)
        
        processed_top_right_frame, temp_detection_info = draw_bounding_boxes(resized_top_right_frame, detections_right, down_sample_factor,top_right_offset, focal_length=540)
        right_detection_info = temp_detection_info
        right_detection_info = filter_edges(right_detection_info, width//2, 0.15, width)
        # Resize the processed top right frame back to original dimensions
        processed_top_right_frame = cv2.resize(processed_top_right_frame, (original_width, original_height))
        
       
        
        
        #LEFT
        # Left
        pred, img_shape = run_inference(model, resized_top_left_frame, device)
        detections_left = process_detections(pred, resized_top_left_frame, img_shape)
        
        top_left_offset = (0,0)
        detections_left = apply_offsets(detections_left, top_left_offset)
        
        processed_top_left_frame,temp_detection_info = draw_bounding_boxes(resized_top_left_frame, detections_left, down_sample_factor,top_left_offset, focal_length=540)
        left_detection_info = temp_detection_info
        left_detection_info = filter_edges(left_detection_info, width//2, 0.15, width)
        
        # Resize the processed top right frame back to original dimensions
        processed_top_left_frame = cv2.resize(processed_top_left_frame, (original_width, original_height))
        
        
        #MIDDLE
        #Middle detection filtering
        pred, img_shape = run_inference(model, resized_top_middle_frame, device)
        detections_middle = process_detections(pred, resized_top_middle_frame, img_shape)
        # Middle
        #Apply offsets to the image coordinates for top left and top right
        top_middle_offset = (0, width // 4)
        detections_middle = apply_offsets(detections_middle, top_middle_offset)
        
        
        # middle processing
        processed_top_middle_frame, temp_detection_info = draw_bounding_boxes(resized_top_middle_frame, detections_middle, down_sample_factor, top_middle_offset, focal_length=540)
        
        
        # Resize the processed top right frame back to original dimensions
        processed_top_middle_frame = cv2.resize(processed_top_middle_frame, (original_width, original_height))
        
        
        middle_detection_info = temp_detection_info
        #Filter out edges of middle part of the image
        middle_detection_info = filter_edges(middle_detection_info, (width*3//10), 0.05, width)
        middle_detection_info = filter_edges(middle_detection_info, (width*7//10), 0.05, width)
        #Filter the middle
        
        middle_detection_info = middle_detection_info
        
        
        #END
        # Place the processed top right frame back onto the original frame
        frame[:height // 2, width // 2:] = processed_top_right_frame
        frame[:height //2, :width // 2] = processed_top_left_frame
        
        
        # Draw cut-off lines
        cut_off_color_middle = (0, 255, 0)  # Green color for the cut-off lines
        cut_off_color_sides = (50,255,50)
        line_thickness = 2
        
        # Vertical lines separating left, middle, and right sections
        #cv2.line(frame, (int(width // 2), 0), (int(width // 2), height // 2), cut_off_color_middle, line_thickness)
        #cv2.line(frame, (int(width * 0.3), 0), (int(width * 0.3), height // 2), cut_off_color_sides, line_thickness)
        #cv2.line(frame, (int(width * 0.7), 0), (int(width * 0.7), height // 2), cut_off_color_sides, line_thickness)
        
        detection_info = right_detection_info + left_detection_info + middle_detection_info
       
    

    return detection_info, frame

# DECISION MAKING THREADS

# This thread reads the frame queue and processes an image based on the thread
def traffic_object_detection(frame_queue, state_queue, model, device,processed_frame_queue):
    # Set distance threshold for red light/traffic sign detection
    red_light_distance_threshold = 5  # meters
    speed_sign_distance_threshold = 10  # meters
    person_distance_threshold = 10 # meters
    car_distance_threshold = 30 # meters

    # Memory buffers for red lights and speed signs
    red_light_memory = collections.deque(maxlen=5)
    speed_sign_memory = collections.deque(maxlen=5)
    person_memory = collections.deque(maxlen=10)
    car_memory = collections.deque(maxlen=5)
    
    saw_red_light = False
    last_speed_limit = 10 #  this sets the initial speed of the car
    
    initial_person_position = "None"
    current_person_position = "None"
    
    # Car
    car_spotted = False
    
    # Data Logger
       # Define the headers for the CSV file
    headers = ["detections", "detect_processing_time", "traffic_processing_time", 
               "year", "month", "day", "hour", "minute", "second", "state", "cpu_usage"]
    
    
        # Create a new folder named with the current date and time
    now = datetime.now()
    folder_name = now.strftime("%m%d_%H")
    os.makedirs(folder_name, exist_ok=True)
    
    # Create a new CSV file within the new folder, also named with the current date and time
    file_name = now.strftime("%Y%m%d_%H%M%S") + "_detection_log.csv"
    file_path = os.path.join(folder_name, file_name)
    
    # Define the headers for the CSV file
    headers = ["detections", "detect_processing_time", "traffic_processing_time", 
               "year", "month", "day", "hour", "minute", "second", "state", "cpu_usage"]
    
    # Create the CSV file and write the headers
    with open(file_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
    print("Data loggers ready")       
    while True:
        if frame_queue:
            # Get the newest frame from the queue
            frame = frame_queue.popleft()
            if frame is not None:
                # Record the start time for processing
                start_time = time.time()
    
                # Process the frame and create a new state
                dets,frame = process_image(model, device, frame, True)
                
                
                # Record the end time for processing and calculate the duration
                end_time = time.time()
                detect_processing_time = end_time - start_time
    
                # Local loop variables for checking closest traffic lights and signs
                # Speed signs
                bool_saw_speed_sign = False
                closest_seen_speed_limit = 0
                closest_speed_sign_distance = np.inf
    
                # Traffic lights
                closest_traffic_light_distance = np.inf
                bool_saw_red_light = False
                
                #People
                closest_person_distance = np.inf
                
                temp_initial_person_position = "None"
                temp_current_person_position = "None"
                
                # Cars
                temp_car_spotted = False
                closest_car_distance = np.inf
                #TODO: Implement logic to detect cars
                if dets:
                    for det in dets:
                        # First check for traffic lights
                        if det['class'] == 'Traffic Light Red':
                            closest_traffic_light_distance = min(closest_traffic_light_distance, det['distance'])
                            bool_saw_red_light = True
                            
                        elif det['class'] in ['Speed-limit-10km-h', 'Speed-limit-15km-h', 'Speed-limit-20km-h']:
                            if det['distance'] < closest_speed_sign_distance:
                                closest_speed_sign_distance = det['distance']
                                closest_seen_speed_limit = int(det['class'].split('-')[2].replace('km', ''))
                                bool_saw_speed_sign = True
                                
                        elif det['class'] == 'Person':
                            x_center = (det['bbox'][0] + det['bbox'][2])/2
                            middle_bound_left = 0.40
                            middle_bound_right = 0.60
                            closest_person_distance = min(closest_person_distance, det['distance'])
                            #Initialize person position
                            if initial_person_position == "None":
                                temp_initial_person_position = "Right" if x_center > width // 2 else "Left"
                                temp_current_person_position = "Right" if x_center > width // 2 else "Left"
                                
                            #Update person position if it was right
                            elif initial_person_position == "Right":
                                if (x_center > width * middle_bound_left and  x_center < width * middle_bound_right):
                                    temp_current_person_position = "Middle"
                                elif x_center < width * middle_bound_left:
                                    temp_current_person_position = "Left"
                                else:
                                    temp_current_person_position = "Right"
                            
                            #Update person position if it was left
                            elif initial_person_position == "Left":
                                if (x_center > width * middle_bound_left and  x_center < width * middle_bound_right):
                                    temp_current_person_position = "Middle"
                                elif x_center > width * middle_bound_right:
                                    temp_current_person_position = "Right"
                                else:
                                    temp_current_person_position = "Left"
                        elif det['class'] =='Car':
                            closest_car_distance = min(closest_car_distance, det['distance'])
                            temp_car_spotted = True
                        
                        
    
                # Update memory buffers
                red_light_memory.append(bool_saw_red_light and closest_traffic_light_distance < red_light_distance_threshold)
                speed_sign_memory.append((bool_saw_speed_sign, closest_seen_speed_limit) if closest_speed_sign_distance < speed_sign_distance_threshold else (False, 0))
                person_memory.append((closest_person_distance < person_distance_threshold, temp_current_person_position) if closest_person_distance < person_distance_threshold else (False, "None"))
                car_memory.append(temp_car_spotted and closest_car_distance < car_distance_threshold)
                
                # Check the memory buffers
                saw_red_light = all(red_light_memory)
                # Update last speed limit if all detected speed signs agree
                if all(sign[0] for sign in speed_sign_memory) and len(set(sign[1] for sign in speed_sign_memory)) == 1:
                    last_speed_limit = closest_seen_speed_limit
                
               # Update person positions if all detections agree
                if all(person[0] for person in person_memory):
                    positions = [person[1] for person in person_memory]
                    if len(set(positions)) == 1:
                        current_person_position = positions[0]
                        if initial_person_position == "None":
                            initial_person_position = positions[0]
                
                # Check if person memory is all false
                if not any(person[0] for person in person_memory):
                    initial_person_position = "None"
                    current_person_position = "None"
                    
                # Memory for car spotting
                car_spotted = all(car_memory)
                
                new_state = {
                    "spotted_red_light": saw_red_light,
                    "Speed limit": last_speed_limit,
                    "Initial Person Position": initial_person_position,
                    "Current Person Position": current_person_position,
                    "Car Spotted": car_spotted
                }
                
    
                state_queue.append(new_state)
                #print(f"Updated shared_state: {new_state}")
                end_time = time.time()
                
                traffic_detect_processing_time = end_time - start_time
                
                # Measure CPU usage
                cpu_usage = psutil.cpu_percent(interval=None)
                
                # Display the current state on the frame
                state_text = (
                    f"Red Light: {saw_red_light}\n"
                    f"Speed Limit: {last_speed_limit}\n"
                    f"Initial Person Position: {initial_person_position}\n"
                    f"Current Person Position: {current_person_position}\n"
                    f"Car Within Treshold: {car_spotted}\n"
                    f"CPU Usage: {cpu_usage}\n"
                    f"FPS: {1/traffic_detect_processing_time}"
                )
                
                # Define the starting position
                x, y = 10, frame.shape[0] - 150  # Adjust y to fit all lines within the frame
                
                # Split the state text into lines
                for i, line in enumerate(state_text.split('\n')):
                    cv2.putText(frame, line, (x, y + (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                
                # Save the frame to an image file for testing
                #cv2.imwrite('test2.png', frame)
                print("sending frames")
                processed_frame_queue.append(frame)
                
               # Show the frame with the current state
                
                    
                                 # Extract the current timestamp and break it into components
                now = datetime.now()
                year, month, day = now.year, now.month, now.day
                hour, minute, second = now.hour, now.minute, now.second
                
                
                
                detection_info = {
                    "detections": dets,
                    "detect_processing_time": detect_processing_time,
                    "traffic_processing_time": traffic_detect_processing_time,
                    "year": year,
                    "month": month,
                    "day": day,
                    "hour": hour,
                    "minute": minute,
                    "second": second,
                    "state": new_state,
                    "cpu_usage": cpu_usage
                }
                
                with open(file_path, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        detection_info["detections"],
                        detection_info["detect_processing_time"],
                        detection_info["traffic_processing_time"],
                        detection_info["year"],
                        detection_info["month"],
                        detection_info["day"],
                        detection_info["hour"],
                        detection_info["minute"],
                        detection_info["second"],
                        detection_info["state"],
                        detection_info["cpu_usage"]
                    ])

def adjust_throttle(state_queue, throttle_queue, max_car_speed=20):
    while True:
        if state_queue:
            # Get the newest state from the queue
            new_state = state_queue.pop()
            # Adjust the throttle based on the shared state
            throttle_speed = calculate_throttle_based_on_state(new_state, max_car_speed)
            # Update the throttle queue with the most recent throttle speed
            if len(throttle_queue) >= 1:
                throttle_queue.pop()
            throttle_queue.append(throttle_speed)

            # Prepare the data to be dumped into JSON
            throttle_info = {
                "speed": new_state["Speed limit"],
                "throttle": throttle_speed,
                "saw_red_light": new_state["spotted_red_light"],
                "timestamp": datetime.now().isoformat()
            }

            # Append the data to a JSON log file
            #with open("throttle_log.json", "a") as f:
               # f.write(json.dumps(throttle_info) + "\n")

        time.sleep(0.03)

def calculate_throttle_based_on_state(state,max_car_speed=20):
    # Dummy function to calculate throttle speed based on state
    # Replace with your actual throttle calculation logic
    if state["spotted_red_light"]:
        return 0  # Stop if red light is spotted
    
    elif state["Car Spotted"]:
        #TODO: Implement logic to slow down if a car is spotted
        return 12
    
    elif (state["Current Person Position"] == "Right" or state["Current Person Position"] =="Middle") and state["Initial Person Position"] == "Right":
        return 0  # Stop if person is on the right side or on the road and started on the right side
    
    elif(state["Current Person Position"] == "Left" or state["Current Person Position"] =="Middle") and state["Initial Person Position"] == "Left":
        return 0 # Stop if person is on the left side or on the road and started on the left side
    
    else:
        car_speed_km_h = min(state["Speed limit"], max_car_speed)
        return int(car_speed_km_h/max_car_speed *100)  # Throttle speed is calculated as a percentage of max speed of the car(NOT SIGN)




height = 480
width = 848

# Image folder functions
def find_next_available_dir(base_path, base_name):
    counter = 0
    while True:
        new_path = os.path.join(base_path, f"{base_name}_{counter}")
        if not os.path.exists(new_path):
            return new_path
        counter += 1

def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

    
    
    
    
    
def main():
    print("Starting...")
    # Initialization of the model
    weights_path = 'v5_model.pt'
    output_directory_base = 'detection_frames'
    no_gui_save_dir= '/home/sdc/Documents/ANNOTATED_IMAGES'
    input_type = 'image_directory'
    input_source = "/home/sdc/Documents/latest_recording"
    
    print("Initilazing model...")
    model, device, detections_dir, no_gui_save_dir, gui_available = initialize(weights_path, output_directory_base, no_gui_save_dir)
    print("GUI available: ", gui_available)
    gui_available = True
    
    # Start up of decision making threads
    # Shared state and queues
    global shared_state
    shared_state = {
        "spotted_red_light": False,
        "Speed limit": 10,
        "Initial Person Position": "None", # This can be "Left" or "Right" or "Middle" or "None"
        "Current Person Position": "None", # This can be "Left" or "Right" or "Middle" or "None"
        "Car Spotted": False
    }
    
    MAX_CAR_SPEED = 20
    
    
    # Event to signal threads to stop
    global stop_event
    stop_event = threading.Event()
    
    # Deques for state and frame queues with a maximum length
    queue_maxsize = 1
    state_queue = deque(maxlen=queue_maxsize)
    frame_queue = deque(maxlen=queue_maxsize)
    
    processed_frame_queue = deque(maxlen=1)
    
    # Shared queue for throttle speed
    throttle_queue = deque(maxlen=1)

    # Initialize the threads for frame processing and throttle adjustment
    frame_processing_thread = threading.Thread(target=traffic_object_detection, args=(frame_queue, state_queue,model,device,processed_frame_queue))
    throttle_adjustment_thread = threading.Thread(target=adjust_throttle, args=(state_queue, throttle_queue,MAX_CAR_SPEED))

    # Start the threads
    frame_processing_thread.start()
    throttle_adjustment_thread.start()
    print("Object detection threads started...")
    
    # LOOP Part
    frame_index = 0
    
    image_paths = sorted(Path(input_source).glob('*.png'))
    print(f"Found {len(image_paths)} images in the directory")
    paused = False
    speed = 1
    total_frames = len(image_paths)
    total_processing_time = 0
    processed_frames = 0
    
    try:
        while frame_index < total_frames:
            image_path = image_paths[frame_index]
            start_time = time.time()
            frame = cv2.imread(str(image_path))
            frame_queue.append(frame)
            
            end_time = time.time()
            processing_time = end_time - start_time
            total_processing_time += processing_time
            processed_frames += 1
    
            average_processing_time = total_processing_time / processed_frames
    
            print(f"Frame {frame_index}/{total_frames - 1} | \n Speed: {'Paused' if paused else f'{speed}x'} | \n Avg. Processing Time: {average_processing_time:.4f}s", end='\n')
    
            
    
            if not paused:
                frame_index += speed
    
            if gui_available:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    paused = not paused
                elif key == ord('a'):
                    speed = max(1, speed // 2)
                elif key == ord('d'):
                    speed = min(64, speed * 2)
                elif key == ord('w'):
                    frame_index = min(total_frames - 1, frame_index + int(10 * speed))
                elif key == ord('s'):
                    frame_index = max(0, frame_index - int(10 * speed))
                elif key == ord('e'):
                    print(f"\nCurrent Frame: {frame_index} | Avg. Processing Time: {average_processing_time:.4f}s")
                elif key == ord('r'):
                    save_path = os.path.join(detections_dir, f"{frame_index:04d}.jpg")
                    cv2.imwrite(save_path, frame)
                    print(f"\nSaved Frame: {save_path}")
                
                #Simulate 30 FPS
                time.sleep(1/30)
                
                #Show the processed frame
                if processed_frame_queue:
                    #print("showing frame")
                    altered_frame = processed_frame_queue.pop()
                    cv2.imshow("Processed frame", altered_frame)
            
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        print("Stopping threads...")
        stop_event.set()
        frame_processing_thread.join()
        throttle_adjustment_thread.join()
        cv2.destroyAllWindows()
        print("Threads stopped and resources cleaned up.")

    
main()






