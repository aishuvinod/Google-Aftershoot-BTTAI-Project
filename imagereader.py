import numpy 
import os
import pandas as pd
import glob
import cv2

labels = [] #list of labels 0 - 9 for the images (10 categories, 0 -9) 
images = [] #image pixels list



def read_image(data_dir, folder, label) :
  '''
  Takes in data_dir, individual folder, and a count (for the label) and flattens each image data and stores it into image list. 
  '''
  # retrieving folder path by joining the data directory and folder
  folder_path = os.path.join(data_dir, folder)

  # Retreiving a list of images in the folder
  image_list = os.listdir(folder_path)

   # looping through the list of images in the folder
  for img_jpg in image_list:
    image_path = os.path.join(folder_path, img_jpg)

    #read the images
    read_img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    #resize the images 
    resized_image = cv2.resize(read_img, (64,64))
    #resized_img_flatten = resized_image.reshape(-1)
    #print(resized_img_flatten.shape)
    
    #append the index of the label into the labels list
    labels.append(label)

    if read_img is not None:
      #flatten the image
      flatten_img = resized_image.flatten()
      #print(flatten_img.shape); 

      #append the flattened image to the images list
      images.append(flatten_img)
    
    else:
      print('None')



def main(): 
    print("Reading data")

    label_index = 0; #counter variable for the category 
    data = "/Users/isabellewang/Downloads/Google-Aftershoot-BTTAI-Project/EuroSAT dataset"
    folders = os.listdir(data)
    folders.remove(".DS_Store")

    #folder_paths = [os.path.join(data, folder) for folder in folders]

    #looping into each folder/category to read the image 
    for folder in folders:
        read_image(data, folder, label_index)
        label_index+=1

    # makes the dataframe for the images with pixels as column names
    df = pd.DataFrame(data = images, columns = [i for i in range(0, images[0].shape[0])])

    #append the label to dataframe
    df['label'] = labels

    #convert the dataframe to a csv file
    df.to_csv("Eurodataset_preprocessing.csv", index = "image_name"); 

    print("Data stored in a csv complete!")

    
if __name__ == '__main__': 
    main()
    

    

