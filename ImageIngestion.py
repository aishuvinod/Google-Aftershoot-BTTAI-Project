import os
import cv2
import pandas as pd

"""
Sources

1. https://medium.com/analytics-vidhya/read-image-using-cv2-imread-opencv-python-idiot-developer-4b7401e76c20
2. https://www.geeksforgeeks.org/impact-of-image-flattening/

"""


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
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Using this because we want to have RBG values and the default cv2.IMREAD_COLOR flag reads as BGR
                flattened_image = image_rgb.flatten()
                labeled_image = list(flattened_image) + [label] #add the label for the image
                
                # add to list
                data.append(labeled_image)
                image_names.append(image_name)

# convert list to DataFrame with rows and columns. each column represents one pixel in the flattened image array.
columns = [f'pixel_{i}' for i in range(len(flattened_image))] + ['label']
df = pd.DataFrame(data, columns=columns, index=image_names)

# save the DataFrame as a CSV file
df.to_csv('image_data.csv', index_label = 'image_name')

print("completed!!")
