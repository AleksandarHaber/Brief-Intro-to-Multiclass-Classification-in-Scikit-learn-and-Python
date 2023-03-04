# -*- coding: utf-8 -*-
"""
Demonstration of how to perform multi-class classification by using Scikit-learn and Python 
Note that we are using the support vector machine classifier SVC()
Author: Aleksandar Haber 
Date: March 2023

"""
# here, we import the data set library
from sklearn import datasets 
# here, we import the standard scaler in order to standardize the data
from sklearn.preprocessing import StandardScaler 
# here, we import the function for splitting the data set into training and test data sets
from sklearn.model_selection import train_test_split
# support vector machine classifier
from sklearn.svm import SVC
import numpy as np

# load the data set
dataSet=datasets.load_iris()

# input data for classification
Xtotal=dataSet['data']
Ytotal=dataSet['target']

# split the data set into training and test data sets
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
Xtrain, Xtest, Ytrain, Ytest = train_test_split(Xtotal, Ytotal, test_size=0.3)

# create a standard scaler
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
scaler1=StandardScaler()
# scale the training and test input data
# fit_transform performs both fit and transform at the same time
XtrainScaled=scaler1.fit_transform(Xtrain)
# here we only need to transform
XtestScaled=scaler1.transform(Xtest)

# initialize the classifier
# automatically recognizes that the problem is the multiclass problem 
# and performs one-vs-one classification
classifierSVM=SVC(decision_function_shape='ovo')

# train the classifier
classifierSVM.fit(Xtrain,Ytrain)

#predict classes by using the trained classifier
# complete prediction
predictedY=classifierSVM.predict(Xtest)
# peform basic comparison
Ytest==predictedY

# single sample prediction 
predictedSampleY=classifierSVM.predict(Xtest[5,:].reshape(1,-1))
Ytest[5]

