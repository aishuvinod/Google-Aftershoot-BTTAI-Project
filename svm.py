import imagereader

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import numpy as np
from sklearn import svm
import os
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score

#access file
filename = "/Users/isabellewang/Downloads/Google-Aftershoot-BTTAI-Project/Eurodataset_preprocessing.csv"

#convert csv to a dataframe
df = pd.read_csv(filename)


def train_test(X_train, X_test, y_train, y_test): 
   '''
   fit and predict an SVM model and returns the accuracy score
   '''
   model = svm.SVC(kernel='linear', C=1.0)

   model.fit(X_train, y_train) #fit model

    #predict 
   class_label_prediction = model.predict(X_test)

   #determine accuracy score
   acc_score = accuracy_score(y_test, class_label_prediction)

   return acc_score


def main(): 
    #accessing folders
    data = "/Users/isabellewang/Downloads/Google-Aftershoot-BTTAI-Project/EuroSAT dataset"
    folders = os.listdir(data)
    folders.remove(".DS_Store")

    y = df['label'] #label
    X = df.drop(columns = "label") #features

    #split each category into a 90/10 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 123, test_size = 0.10, stratify = y)

    acc = train_test(X_train, X_test, y_train, y_test)

    #prints the accuracy score
    print("The accuracy score is: " , str(acc))

if __name__ == "__main__": 
    main()



