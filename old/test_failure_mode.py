#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:40:53 2024

@author: miguelgomez
"""

# This is a test of a classifier for the failure mode
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# Import the ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


def run_multinomial_lr(data, features, predval):
    '''
    Runs a multinomial logistic regression with DATA, using FEATURES as the 
    predictors and PREDVAL as the predicted value
    
    '''
    
    # Define the independent variables and the dependent variable
    X = data[features]
    y = data[predval]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the multinomial logistic regression model
    log_reg = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    log_reg.fit(X_train, y_train)

    # Predict on the test set
    y_pred = log_reg.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    
    # Compute ROC curve and ROC area for each class (binarized)
    y_train_bin = label_binarize(y_train, classes=[0, 1, 2])
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    n_classes = y_train_bin.shape[1]
    y_score = log_reg.predict_proba(X_test)
    
    for vals in zip(y_score, y_test):
        print(vals)
    

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot the ROC curves
    # plt.figure(dpi=500)
    #for i in range(n_classes):
    #    plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')


if __name__ == '__main__':
    
    data = pd.read_csv('data_spiral_wnd.csv')
    features = ['ar', 'lrr', 'srr', 'alr', 'sdr']
    predval = 'ft'
    
    run_multinomial_lr(data, features, predval)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    