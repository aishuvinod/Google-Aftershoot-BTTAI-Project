import numpy 
import os
import pandas as pd
import glob
import cv2

labels = []
images = []


def read_image(data_dir, folder) :
  # retrieving folder path by joining the data directory and folder
  folder_path = os.path.join(data_dir, folder)

  # Retreiving a list of images in the folder
  image_list = os.listdir(folder_path)

   # looping through the list of images in the folder
  for img_jpg in image_list:
    image_path = os.path.join(folder_path, img_jpg)
    read_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    resized_image = cv2.resize(read_img, (64,64))
    #resized_img_flatten = resized_image.reshape(-1)
    #print(resized_img_flatten.shape)
    
    labels.append(folder)

    if read_img is not None:
      img_data = resized_image
      flatten_img = img_data.flatten()
      print(flatten_img.shape); 
      #print(flatten_img.shape)
      images.append(flatten_img)
    else:
      print('None')


def main(): 
    data = "/Users/isabellewang/Downloads/BTTAI Google:Aftershoot Isabelle/Google-Aftershoot-BTTAI-Project/EuroSAT dataset"
    folders = os.listdir(data)
    #folder_paths = [os.path.join(data, folder) for folder in folders]
    for folder in folders:
       if folder != ".DS_Store": 
          read_image(data, folder)

    


if __name__ == '__main__': 
    print("Reading data")
    main()
    
    df = pd.DataFrame(data = images, columns = [i for i in range(0, images[0].shape[0])])
    df['label'] = labels

    #data = df.iloc[:50]
    df.to_csv("Eurodataset_preprocessing.csv", index = "image_name"); 

    print("Data stored in a csv complete!")

    

