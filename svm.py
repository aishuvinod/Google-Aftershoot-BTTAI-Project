import imagereader

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
import numpy as np
from sklearn import svm
import os
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import seaborn as sns
import pickle 
from sklearn.preprocessing import label_binarize


#access file
filename = "/Users/isabellewang/Downloads/Google-Aftershoot-BTTAI-Project/Eurodataset.csv"

#convert csv to a dataframe
df = pd.read_csv(filename)

def train_test(X_train, X_test, y_train, y_test): 
   '''
   fit and predict an SVM model and returns the accuracy score
   '''
   print("here")
   model = svm.SVC(kernel='linear', C=1)
   
   print("fit")
   model.fit(X_train, y_train) #fit model

   # predict 
   print("class")
   class_label_prediction = model.predict(X_test)

   print("acc")
   #determine accuracy score
   acc_score = accuracy_score(y_test, class_label_prediction)

   #c_m = confusion_matrix(y_test, class_label_prediction,labels = [True, False])
   #print(c_m)

   return acc_score

def random_grid_search_best_param(X_train, y_train, X_test, y_test): 
    model = svm.SVC(kernel = "linear")
    param_random = {'C': [0.001, 0.01, 0.1, 1], 'gamma': [0.01]}
    grid_search = GridSearchCV(model, param_random, cv = 5, random_state = 42)
    grid_search.fit(X_train, y_train) 
    best = grid_search.best_params_
    print(best)
    return best

def random_search_best_param(X_train, y_train, X_test, y_test): 
    print("in")
    model = svm.SVC(kernel = "linear")
    param_random = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.01]}

    search = RandomizedSearchCV(model, param_random, cv = 5, random_state=42)
    search.fit(X_train, y_train)

    best = search.best_params_
    print(best)

    return best


def compute_precision_recall(X_train, y_train, X_test, y_test): 
    print("computing")
    print(X_train.shape)
    print(y_train.shape)
    model_best = svm.SVC(kernel='linear', C=1e-06, gamma = 0.01)

    model_best.fit(X_train, y_train)
    class_label_prediction = model_best.predict(X_test)

    cm = confusion_matrix(y_test, class_label_prediction)
    print(cm)
    print("Confusion Matrix")
    print("Reporting")
    print(metrics.classification_report(y_test, class_label_prediction))
    plt.figure(figsize=(9, 9))
    sns.heatmap(cm, annot = True, fmt = '0.3f', linewidth = 0.5, square = True, cbar = False)
    plt.ylabel('Actual Values')
    plt.xlabel('Predicted values')
    plt.show()
    

def train_test_best_model(best_C, best_gamma, best_kernel, X_train, y_train, X_test, y_test): 
    model = svm.SVC(kernel = best_kernel, C = best_C, gamma = best_gamma)
    model.fit(X_train, y_train)
    class_label_predictions = model.predict(X_test)
    acc_score = accuracy_score(y_test, class_label_predictions)
    
    return acc_score

def graph_ROC(X_train, y_train, X_test, y_test): 
    pass

def main(): 
    print("starting")
    
    y = df['label'] #label
    print(df['label'].unique())
    X = df.drop(columns = "label") #features
    print(y.shape)
    print(X.shape)
    #y = label_binarize(y, classes = [i for i in range(0, 10)])
    #print(y.shape)
    #split each category into a 90/10 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, test_size = 0.10, stratify = y)
    
    X_train = X_train/255.0
    X_test = X_test/255.0 


    print("ahe")
    acc = train_test(X_train, X_test, y_train, y_test)

    #prints the accuracy score
    #print("The accuracy score is: " , str(acc))
    print(random_search_best_param(X_train, y_train, X_test, y_test))
    #print(random_grid_search_best_param(X_train, y_train, X_test, y_test))
    #compute_precision_recall(X_train, y_train, X_test, y_test)


if __name__ == "__main__": 
    main()


#notes
#param_random = {'C': [1, 10, 100], 'gamma': [0.01, 0.1]}
#best: {'gamma': 0.01, 'C': 1} for cv = 5
#best: {gamma = 0.01, C = 0.001} cv = 5 random search 
#{'gamma': 1, 'C': 1e-06} cv = 5 The accuracy score is:  0.9883008356545961
#The accuracy score is:  0.9821727019498607
#rbf score: The accuracy score is:  0.24846796657381615
#poly : The accuracy score is:  0.9793871866295265