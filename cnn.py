import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import time 

print("reading file")
fileName= "/Users/isabellewang/Downloads/Google-Aftershoot-BTTAI-Project/Eurodataset.csv"
df = pd.read_csv(fileName) 

X = df.drop(columns= "label") 


#reshape pixel values into a 3D format 
image_dim = 64
num_channels = 3

#getting labels 
y = df["label"] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#normalizing the pixel values in X to be between 0 and 1 by dividing by 255 
X_train = X_train/255.0
X_test = X_test/255.0 


image_dim = 64
num_channels = 3
X_train = np.reshape(X_train, (-1, image_dim, image_dim, num_channels))
X_test = np.reshape(X_test, (-1, image_dim, image_dim, num_channels))

print (X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
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

#7. Output Layer
cnn_model.add(keras.layers.Dense(10, activation="softmax")) 

# Print the model summary
cnn_model.summary()

print("Model completed.")

# Optimization function 
sgd_optimizer = keras.optimizers.SGD(learning_rate = 0.1)

# loss function 
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits = True)

# Compile model 
cnn_model.compile(optimizer = sgd_optimizer, loss = loss_fn, metrics = ['accuracy'])

# Fit the model 
num_epochs = 1 # Number of epochs
t0 = time.time() # start time
cnn_model.fit(X_train, y_train, epochs = num_epochs, validation_split = 0.1) # fit model
t1 = time.time() # stop time
print('Elapsed time: %.2fs' % (t1-t0))

# Evaluating the model 
loss, accuracy = cnn_model.evaluate(X_test, y_test)
print('Loss: ', str(loss) , 'Accuracy: ', str(accuracy))
