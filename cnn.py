import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#import tensorflow.keras as keras
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

csv = "image_data.csv" 
df = pd.read_csv(csv) 

pixel_data = df.drop(columns=['image_name']) 
pixel_values = pixel_data.to_numpy  # converts to numpy array

#reshape pixel values into a 3D format 
image_dim = 64
num_channels = 3
X = pixel_values

#getting labels 
df['label'] = df['image_name'].str.split('_').str[0] #label is like the image_name prefix 
y = pd.factorize(df['label'])[0]  # converts labels to integers
print(y) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123, stratify=y)

#normalizing the pixel values in X to be between 0 and 1 by dividing by 255 
X_train = X_train/255.0
X_test = X_test/255.0 

image_dim = 64
num_channels = 3
X = pixel_values.reshape((-1, image_dim, image_dim, num_channels))

#building cnn model (from the ml class over the summer)
# Build the CNN model
cnn_model = keras.Sequential()

# 1. Input layer
input_layer = keras.layers.InputLayer(input_shape=(64, 64, 3))  # Adjusted for RGB
cnn_model.add(input_layer)

# 2. First convolutional block
cnn_model.add(keras.layers.Conv2D(16, (3, 3), padding="same"))
cnn_model.add(keras.layers.BatchNormalization())
cnn_model.add(keras.layers.ReLU())

# 3. Second convolutional block
cnn_model.add(keras.layers.Conv2D(32, (3, 3), padding="same"))
cnn_model.add(keras.layers.BatchNormalization())
cnn_model.add(keras.layers.ReLU())

# 4. Third convolutional block
cnn_model.add(keras.layers.Conv2D(64, (3, 3), padding="same"))
cnn_model.add(keras.layers.BatchNormalization())
cnn_model.add(keras.layers.ReLU())

# 5. Fourth convolutional block
cnn_model.add(keras.layers.Conv2D(128, (3, 3), padding="same"))
cnn_model.add(keras.layers.BatchNormalization())
cnn_model.add(keras.layers.ReLU())

# 6. Global average pooling
cnn_model.add(keras.layers.GlobalAveragePooling2D())

cnn_model.add(keras.layers.Dense(units=len(np.unique(y)), activation="softmax")) 

# Compile the model
cnn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Print the model summary
cnn_model.summary()
