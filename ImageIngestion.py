import os
import cv2
import pandas as pd

# Define the desired image size (e.g., 64x64 pixels)
IMAGE_SIZE = (64, 64)

dataset_path = '/Users/aishuvinod/Desktop/breakthrough/Google Project/Google-Aftershoot-BTTAI-Project/EuroSAT dataset'
data = []
image_names = []  # List to store image names for the index. We may not need this at all/can just have default 0,1,2 index

# loop through each folder 
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    
    # check if the path is a directory
    if os.path.isdir(folder_path):
        # Using the folder name as the label for now later we can change it numbers?? 0, 1, 2 etc
        label = folder
        
        # loop through each image 
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                # Resize the image to 64x64
                resized_image = cv2.resize(image, IMAGE_SIZE)
                
                # Convert the image to RGB format
                image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                
                # Flatten the image and add the label
                flattened_image = image_rgb.flatten()
                print(flattened_image)
                labeled_image = list(flattened_image) + [label]  # Add the label for the image
                
                # Add to list
                data.append(labeled_image)
                image_names.append(image_name)

# Ensure all rows have the same length by verifying against the resized image dimensions
num_pixels = IMAGE_SIZE[0] * IMAGE_SIZE[1] * 3  # Number of pixels in the flattened image (3 for RGB channels)

# Create column names for the DataFrame
columns = [f'pixel_{i}' for i in range(num_pixels)] + ['label']

# Convert list to DataFrame with rows and columns
df = pd.DataFrame(data, columns=columns, index=image_names)

# Save the DataFrame as a CSV file
df.to_csv('sizedimage_data.csv', index_label='image_name')

print("Completed!!")
