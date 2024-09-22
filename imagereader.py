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
    labels.append(folder)

    if read_img is not None:
      img_data = read_img
      flatten_img = img_data.flatten()
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
    df = pd.DataFrame(data = images)
    df['label'] = labels
    df.to_csv('Eurodataset_preprocessing.csv', index = False)
    print("Data stored in a csv complete!")

    

