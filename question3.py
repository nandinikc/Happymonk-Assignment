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