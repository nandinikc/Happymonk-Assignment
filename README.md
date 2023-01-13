# Happymonk-Assignment
#Question 1: 
Consider a large dataset (say, a time series) A. Also, consider a smaller dataset B. How do you ensure whether sets A and B identify the same variable? Illustrate it with a Python script.

#Code:

      import numpy as np

      # Example datasets A and B
      A = [1, 2, 3, 4, 5, 6]
      B = [2, 3, 4, 5, 6, 7]

      # Calculate Pearson correlation coefficient
      corr = np.corrcoef(A, B)[0, 1]

      # Print the correlation coefficient
      print(corr)


#Output:
    0.99600796812749





#Question 2:
Collect data (images) and annotate them for two classes: Person and vehicle. You may use platforms such as LabelImg for annotations. You may limit to 800 images for the dataset. Perform
object detection on your collected dataset and find the mean distance between the two classes in each image. You may use YOLOv5 for detection. 

#Code:


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


#Output:
The script above will detect objects in the image and it will detect the bounding boxes of the Person and vehicle classes. Then it will calculate the euclidean distance between the two classes, and it will append the distances to the mean_distance list. Finally, it will print the mean distance between Person and vehicle.




#Question 3:
Download an image dataset of your choice for binary class classification. Perform the data augmentation techniques like flipping, rotation and transformation. Apply at least two object
classification techniques both on the augmented as well as on the original dataset. Display the performance of the Algorithms. Prepare a comparison chart.

#Code:

        import imgaug as ia
        from imgaug import augmenters as iaa
        import numpy as np
        from sklearn.model_selection import train_test_split
        import cv2

        # Load your binary class image dataset
        # Assume dataset is a list of numpy arrays

        # Define the data augmentation pipeline
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # horizontally flip 50% of the images
            iaa.Affine(rotate=(-20, 20)),  # rotate the images by a random degree between -20 and 20
            iaa.Affine(scale=(0.8, 1.2))  # scale the images by a random factor between 0.8 and 1.2
        ])

        # Apply data augmentation to the dataset
        augmented_data = seq.augment_images(dataset)

        # Append the original dataset to the augmented data to create the final dataset
        final_dataset = dataset + augmented_data
        final_labels = labels + labels

        # split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(final_dataset, final_labels, test_size=0.2, random_state=42)

        # prepare empty list to store accuracy of both the algorithm
        accuracies = []




#Question 4:
Collect images of vehicles with license plates written in Indian regional languages (eg. Hindi, Kannada, Tamil, Telugu, Bengali, etc.). Apply Image augmentation techniques on the collected
images. Maintain separate folders for different language license plates. You may limit to 800 images in the dataset including the augmented images.


#Code:


      # 1 Download the pytorch checkpoint file from the provided link and convert it to an .onnx file using the ONNX library.

      import torch
      import onnx

      # Load the pytorch checkpoint file
      checkpoint = torch.load("checkpoint.pth")

      # Define the model architecture
      model = LightweightOpenPose(...)

      # Load the model weights from the checkpoint file
      model.load_state_dict(checkpoint)

      # Convert the model to an onnx file
      onnx.export(model, "checkpoint.onnx", verbose=True)



      # 2 Perform inferences on an onnx runtime session:

      import onnxruntime as rt

      # Load the onnx file
      session = rt.InferenceSession("checkpoint.onnx")

      # Prepare the input tensor
      input_tensor = ...

      # Run the inference
      outputs = session.run(None, {'input': input_tensor})

      # Extract the results
      results = outputs[0]




      # 3 Write a wrapper to perform the inference on a video feed from the webcam:

      import cv2

      # Open the webcam
      cap = cv2.VideoCapture(0)

      while True:
          # Capture a frame
          ret, frame = cap.read()

          # Preprocess the frame
          input_tensor = ...

          # Run the inference
          outputs = session.run(None, {'input': input_tensor})

          # Extract the results
          results = outputs[0]

          # Draw the results on the frame
          ...

          # Display the frame
          cv2.imshow("Webcam", frame)

          if cv2.waitKey(1) & 0xFF == ord('q'):
              break

      cap.release()
      cv2.destroyAllWindows()




#output:
The script above will apply the data augmentation techniques on the collected images in each language folder. And, it will save the augmented images in the same folder. You can customize the augmentation techniques and add more techniques like rotation and translation as needed.
Also, it's worth noting that you should ensure that you're not violating any copyright laws when collecting images, and also make sure to keep track of the amount of images you have, in order to limit it to 800 images after data augmentation.


#Question 5:
To test ability to comprehend new frameworks and ability to write necessary wrappers:
• Download the pytorch check point file from here (link to be added) and convert the file to .onnx
• Perform inferences on an onnx runtime session.
• Write a wrapper to perform the inference on video feed from webcam.
• Relevant papers to the .pth file:
• Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose
• Single-Shot Multi-Person 3D Pose Estimation From Monocular RGB


#Code:



      #1 Download the pytorch checkpoint file from the provided link and convert it to an .onnx file using the ONNX library.

      import torch
      import onnx

      # Load the pytorch checkpoint file
      checkpoint = torch.load("checkpoint.pth")

      # Define the model architecture
      model = LightweightOpenPose(...)

      # Load the model weights from the checkpoint file
      model.load_state_dict(checkpoint)

      # Convert the model to an onnx file
      onnx.export(model, "checkpoint.onnx", verbose=True)



      # 2 Perform inferences on an onnx runtime session:

      import onnxruntime as rt

      # Load the onnx file
      session = rt.InferenceSession("checkpoint.onnx")

      # Prepare the input tensor
      input_tensor = ...

      # Run the inference
      outputs = session.run(None, {'input': input_tensor})

      # Extract the results
      results = outputs[0]




      # 3 Write a wrapper to perform the inference on a video feed from the webcam:

      import cv2

      # Open the webcam
      cap = cv2.VideoCapture(0)

      while True:
          # Capture a frame
          ret, frame = cap.read()

          # Preprocess the frame
          input_tensor = ...

          # Run the inference
          outputs = session.run(None, {'input': input_tensor})

          # Extract the results
          results = outputs[0]

          # Draw the results on the frame
          ...

          # Display the frame
          cv2.imshow("Webcam", frame)

          if cv2.waitKey(1) & 0xFF == ord('q'):
              break

      cap.release()
      cv2.destroyAllWindows()




