# importing useful libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV


# read the CSV table

df = pd.read_csv('df.csv')     #read_csv import the file to data frame format

##### MODELLING THE DATA #####
# we are interested in predicting whether a delivery will fail or not
# in the prediction of a boolean value, we use a classification model
# for score function, we'll use a risk-averse one -> recall score function
# to sum up: Random Forest Model with Recall Score Function


# separating the data between test and train

X_train, X_test, y_train, y_test = train_test_split(df.drop(["ineficiencia"], axis=1), df["ineficiencia"],test_size=0.4,   )
# reminder: between 0.4 and 0.6 is good practice

# dealing with the hyperparameters

clf = GridSearchCV(
   RandomForestClassifier(),
   {
       "criterion": ["gini", "entropy"],
       "max_depth": [50 + i * 10 for i in range(10)],
       "oob_score": [True, False],
       "max_features": ["sqrt", None, 6],
       "min_impurity_decrease": [0.00, 0.001],
       "class_weight": ["balanced"],
   },
   scoring="recall",
   n_jobs=-1,
   verbose=3,
)

# fit and tune model

clf.fit(X_train, y_train)

# print the best model parameters among the tested

print( "Recall score of best model:",
        round(recall_score(y_test, clf.predict(X_test)) * 100, 2),
       "%",
)

print("Best model parameters:")
clf.best_params_

