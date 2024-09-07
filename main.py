import os
import cv2
import pandas as pd

#time out methods bc something isn't working ;-;
import signal

# Define a handler for the timeout
def handler():
    raise Exception("image processing timeout")

# Set the signal handler for the timeout
signal.signal(signal.SIGALRM, handler)

# Function to safely load an image with timeout
def load_image_with_timeout(image_path, timeout=5):
    try:
        # Start the timer
        signal.alarm(timeout)
        image = cv2.imread(image_path)
        signal.alarm(0)  # Disable the alarm
        return image
    except Exception as e:
        print(f"Error reading {image_path}: {e}")
        return None
    

# path of the current directory (where main.py is located)
current_directory = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(current_directory, '2750')
data = []
image_names = []  # stores image names 

# loop through each folder 
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    
    # check if the path is a directory
    if os.path.isdir(folder_path):
        label = folder
        print(f"processing folder: {folder}")
        
        # loop through each image 
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            print(f"Processing image: {image_name}")
            image = load_image_with_timeout(image_path)
            #print(f"Image {image_name} read successfully. Shape: {image.shape}")
                  
            #trying to fix the issue of the process stopping 
            if image is None:
                print(f"Skipping {image_name}, could not read image.")
                continue

            if image is not None:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Using this because we want to have RBG values and the default cv2.IMREAD_COLOR flag reads as BGR
                #sees if the image got successfully converted from bgr to rbg 
                # Check the shape of the original and converted image
                print(f"Original image shape: {image.shape}")
                print(f"Converted image shape: {image_rgb.shape}")

                # Check if the number of channels is correct and if it isn't then it skips the image 
                # figure out a way to add the bad images to another folder maybe and then you can fix them so you 
                # add them to the data set? 
                if image.shape[2] == 3 and image_rgb.shape[2] == 3:
                    print("Both images have 3 channels.")
                else: 
                    continue 
                
                flattened_image = image_rgb.flatten()
                if flattened_image.ndim == 1: 
                    print("image flattened")
                else: 
                    print("image not flattened") 
                    continue 

                # df wants images of the same size so if they aren't the right size then skip
                # later: figure out a way to just add them to another list and fix the size and read them  
                # bc idk how many images are getting lost bc of this check 
                if (flattened_image.size != (64, 64, 3)): 
                    continue 
                
                labeled_image = list(flattened_image) + [label] #add the label for the image
                
                # adds to list
                data.append(labeled_image)
                image_names.append(image_name)


# convert list to DataFrame with rows and columns. each column represents one pixel in the flattened image array.
print(f"Number of images processed: {len(data)}")
columns = [f'pixel_{i}' for i in range(len(flattened_image))] + ['label']
df = pd.DataFrame(data, columns=columns, index=image_names)
print(df.head()) 

# save the DataFrame as a CSV file
output_csv_path = os.path.join(current_directory, 'image_data.csv')
df.to_csv(output_csv_path, index_label='image_name')
print(f"CSV file saved to: {output_csv_path}")

print("done!!")