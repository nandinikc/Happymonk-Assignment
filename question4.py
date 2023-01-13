import os
import imgaug as ia
from imgaug import augmenters as iaa
from PIL import Image

# Define the data augmentation pipeline
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    iaa.Affine(rotate=(-20, 20)),  # rotate the images by a random degree between -20 and 20
    iaa.Affine(scale=(0.8, 1.2))  # scale the images by a random factor between 0.8 and 1.2
])

# Define the path of the root folder which contains different language folders
root_folder = 'path/to/root/folder'

# Iterate through all the folders which contains the license plates of different languages
for language_folder in os.listdir(root_folder):
    language_folder_path = os.path.join(root_folder, language_folder)
    for image_file in os.listdir(language_folder_path):
        image_file_path = os.path.join(language_folder_path, image_file)
        # Open the image
        img = Image.open(image_file_path)
        # Apply data augmentation
        augmented_img = seq(images=np.array(img))[0]
        # Save the augmented image with the same name in the same folder
        cv2.imwrite(image_file_path, augmented_img)