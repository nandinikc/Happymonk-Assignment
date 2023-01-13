import cv2
from models import *  # import the models

# Define the model and load the weights
yolov5 = YOLOv5('yolov5s.pt')

# Load the image
img = cv2.imread('image.jpg')

# Perform object detection
outputs = yolov5(img)

# Extract the bounding boxes for the Person and vehicle classes
person_boxes = outputs[0]['boxes'][outputs[0]['scores'] > 0.5]
vehicle_boxes = outputs[1]['boxes'][outputs[1]['scores'] > 0.5]

# Initialize list to store mean distance between Person and vehicle
mean_distance = []

# Iterate through all the images
for i in range(len(person_boxes)):
    for j in range(len(vehicle_boxes)):
        # Calculate the distance between Person and vehicle
        distance = ((person_boxes[i][0] - vehicle_boxes[j][0]) ** 2 + 
                    (person_boxes[i][1] - vehicle_boxes[j][1]) ** 2) ** 0.5

        mean_distance.append(distance)

# Print the mean distance between Person and vehicle
print(np.mean(mean_distance))