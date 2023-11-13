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

# checking directory to read csv file

current_directory = os.getcwd()
print("Current directory is:")
print(current_directory)
print(".")

# read the CSV table

df = pd.read_csv('base_chamados.csv')     #read_csv import the file to data frame format

if df is None:
    print('\n')
    print('fail in part 1')

##### cleaning data #####

# selecting only contratante_1 and droping a few columns - some of them because scikit-learn doesn't read strings

df = df.drop(index=[row for row in df.index if 'CONTRATANTE_2' in df.loc[row].values])
df = df.drop(index=[row for row in df.index if '-' in df.loc[row].values])
df = df.drop(columns=['contratante', 'operador', 'grupo_servico','numero_os', 'ec_codcliente', 'sistema_abertura', 'dia_referencia'])

if df is None: # checking
    print('\n')
    print('fail in part 2')


# saving modified file for later

df.to_csv('df.csv', index=False)


# checking for missing values in columns or NaNs

df.dropna()
df = df[df['dois_telefones'].notna()]
result1 = df.isna().any()
print('there is nan = ')
print(result1)

# converting all columns to numeric (since scikit-learn doesn't work with strings)

result2 = np.unique(df['ineficiencia_os_anterior'])
print('unique values in ineficiencia_os_anterior')
print(result2)

df['ineficiencia_os_anterior'] = df['ineficiencia_os_anterior'].astype('int')
print(df.dtypes)

r1 = np.unique(df['forneceu_numero_logradouro'])
print(r1)
r2 = np.unique(df['forneceu_complemento'])
print(r2)
r3 = np.unique(df['endereco_igual_cadastro'])
print(r3)
r4 = np.unique(df['forneceu_telefone'])
print(r4)
r5 = np.unique(df['dois_telefones'])
print(r5)
r6 = np.unique(df['taxa_ineficiencia_cliente'])
print(r6)
r7 = np.unique(df['ineficiencia_os_anterior'])
print(r7)
r8 = np.unique(df['ineficiencia'])
print(r8)

df = df[df['dois_telefones'].notna()]
result1 = df.isna().any()
print('there is nan = ')
print(result1)

print(np.unique(df['dois_telefones']))

print(df.dtypes)


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


