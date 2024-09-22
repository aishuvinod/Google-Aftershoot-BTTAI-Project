import os
from PIL import Image

dataset_path = '/Users/aishuvinod/Desktop/breakthrough/Google Project/Google-Aftershoot-BTTAI-Project/EuroSAT dataset'

total_images = 0  # Variable to store the count of total images
category_image_count = {}  # Dictionary to store image counts for each category

# Loop through each folder (each category)
for folder in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder)
    
    # Check if the path is a directory (i.e., a category folder)
    if os.path.isdir(folder_path):
        category_image_count[folder] = 0  # Initialize count for this category
        
        # Loop through each image in the category folder
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if os.path.isfile(image_path):  # Ensure it's a file
                total_images += 1
                category_image_count[folder] += 1  # Increment the category count
                
                # Open the image and get its dimensions
                # with Image.open(image_path) as img:
                #     width, height = img.size
                #     print(f"Image: {image_name} | Category: {folder} | Dimensions: {width}x{height}")

# # Print the total number of images
print(f"Total number of images: {total_images}")

# Print the number of images in each category
for category, count in category_image_count.items():
    print(f"Category '{category}' has {count} images.")
