import imagereader

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import numpy as np
from sklearn import svm
import os
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score
#https://www.freecodecamp.org/news/svm-machine-learning-tutorial-what-is-the-support-vector-machine-algorithm-explained-with-code-examples/

filename = "/Users/isabellewang/Downloads/Google-Aftershoot-BTTAI-Project/Eurodataset_preprocessing.csv"
df = pd.read_csv(filename)
ds = np.array(df)


def train_test(X_train, X_test, y_train, y_test): 
   model = svm.SVC(kernel='linear', C=1.0)

   model.fit(X_train, y_train)

   class_label_prediction = model.predict(X_test)

   acc_score = accuracy_score(y_test, class_label_prediction)

   return acc_score


def main(): 
    data = "/Users/isabellewang/Downloads/Google-Aftershoot-BTTAI-Project/EuroSAT dataset"
    folders = os.listdir(data)
    folders.remove(".DS_Store")

    # image_count = 0; 
    # y = df['label'][df['label'] == image_count]
    # X = df[df["label"] == image_count].drop(columns = "label")

    y = df['label']
    X = df.drop(columns = "label")

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123, test_size = 0.10, stratify = y)
    print(X_train.shape)
    print(y_train.shape)

    acc = train_test(X_train, X_test, y_train, y_test)

    print("The accuracy score is: " , str(acc))

if __name__ == "__main__": 
    main() 



