import os
import numpy as np
import cv2
import pandas as pd

# using OpenCV, creating the matrix, looping over each image, creating csv

# define path to the dataset
dataset_path = '/content/2750'

# intialize empty lists to store the features and labels
features = []
labels = []

# assign numeric labels
numlabels = {
    'AnnualCrop': 0,
    'Forest': 1,
    'HerbaceousVegetation': 2,
    'Highway': 3,
    'Industrial': 4,
    'Pasture': 5,
    'PermanentCrop': 6,
    'Residential': 7,
    'River': 8,
    'SeaLake': 9
}

# loop through each folder, get list of files and directories in dataset
for folder in os.listdir(dataset_path):
    # get concatenated path to the folder
    folder_path = os.path.join(dataset_path, folder)
    if not os.path.isdir(folder_path):  ####
        continue

    # loop through each image in the folder
    for img_name in os.listdir(folder_path):
        # get concatenated path to the image
        img_path = os.path.join(folder_path, img_name)

        # load image, return np array with pixel values
        img = cv2.imread(img_path)

        # make sure image size is 64x64
        img_resized = cv2.resize(img, (64, 64))

        # flatten image to 1D array and normalize
        img_flattened = img_resized.flatten() / 255.0

        features.append(img_flattened)
        labels.append(numlabels[folder])

# convert features and labels to numpy arrays so we can append
features = np.array(features)
labels = np.array(labels)
data = np.column_stack((features, labels))

df = pd.DataFrame(data)

df.to_csv('eurosatimage_data.csv')
        ####
print("data prep done")

