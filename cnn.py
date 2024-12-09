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
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve

print("reading file")
fileName= "/Users/isabellewang/Downloads/Google-Aftershoot-BTTAI-Project/Eurodataset.csv"
df = pd.read_csv(fileName) 

# get X 
X = df.drop(columns= "label") 

#reshape pixel values into a 3D format 
image_dim = 64
num_channels = 3 # RBG

#getting labels 
y = df["label"] 

# ensuring all labels are included/represented in the labels column 
print(df['label'].nunique())

# Split by class. 80/20 per class 
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

# 7. Global average pooling
cnn_model.add(keras.layers.GlobalAveragePooling2D())

#8. Output Layer
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
num_epochs = 6 # Number of epochs
t0 = time.time() # start time
cnn_model.fit(X_train, y_train, epochs = num_epochs, validation_split = 0.1) # fit model
t1 = time.time() # stop time
print('Elapsed time: %.2fs' % (t1-t0))

# Evaluating the model 
loss, accuracy = cnn_model.evaluate(X_test, y_test)
print('Loss: ', str(loss) , 'Accuracy: ', str(accuracy))

from sklearn.metrics import classification_report

# Get logits (raw model outputs)
logits = cnn_model.predict(X_test)

# Convert logits to class predictions
predictions = logits.argmax(axis=1)

# Generate and print classification report
print("Classification Report:")
print(classification_report(y_test, predictions, target_names=[str(x) for x in range(0, 10)]))

# Generate confusion matrix 
cm = confusion_matrix(y_test, predictions)
print(cm) 
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot = True, fmt = '0.3f', linewidth = 0.5, square = True, cbar = False)
plt.ylabel('Actual Values')
plt.xlabel('Predicted values')
plt.show()


from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve

# Binarize the true labels for AUC-ROC computation
class_labels = sorted(y.unique())  # Assuming y contains the original labels
y_test_binarized = label_binarize(y_test, classes=class_labels)

# Get probabilities from the model for each class
y_pred_prob = cnn_model.predict(X_test)

# Compute AUC for each class
auc_scores = []
plt.figure(figsize=(10, 6))
for i, class_label in enumerate(class_labels):
    fpr, tpr, _ = roc_curve(y_test_binarized[:, i], y_pred_prob[:, i])
    auc_score = roc_auc_score(y_test_binarized[:, i], y_pred_prob[:, i])
    auc_scores.append(auc_score)
    plt.plot(fpr, tpr, label=f"Class {class_label} (AUC = {auc_score:.2f})")

# Plot Random Chance Line
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')

# Plot settings
plt.title("AUC-ROC Curve (Multi-class)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()

# Compute overall AUC
overall_auc = roc_auc_score(y_test_binarized, y_pred_prob, average="weighted", multi_class="ovr")
print(f"\nWeighted AUC-ROC Score: {overall_auc}")

# save models into different formats. 
cnn_model.save('cnnmodel1.h5')
cnn_model.save('cnn_model1.keras')
