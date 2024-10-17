import os
import numpy as np
import cv2
import pandas as pd

df = pd.read_csv("/Users/Mina/VSCodeProjects/googlebttai_self/eurosat_imagedata.csv")

y = df.iloc[:,-1:].values.ravel()
#print(y)

X = df.iloc[:,:-1]  #features
#print(X)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=123, stratify = y)

from sklearn.svm import SVC

# Create and train the SVC model
svc_model = SVC()
svc_model.fit(X_train, y_train)

from sklearn.metrics import classification_report, accuracy_score

# Predict on the test set
y_pred = svc_model.predict(X_test)

# Print evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

import pickle 

saved_model = pickle.dumps(svc_model) 

