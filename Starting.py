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

# selecting only contratante_1 and droping a few useless columns


df = df.drop(index=[row for row in df.index if 'CONTRATANTE_2' in df.loc[row].values])
df = df.drop(columns=['numero_os', 'ec_codcliente', 'sistema_abertura'])

if df is None:
    print('\n')
    print('fail in part 2')

# constructing other data based on the essencially modified one

data1_analysis1 = df.drop(index=[row for row in df.index if 0 == df.loc[row, 'endereco_igual_cadastro']]) # drop the negative ones
data2_analysis1 = df.drop(index=[row for row in df.index if 1 == df.loc[row, 'endereco_igual_cadastro']]) # drop the positive ones

data1_analysis2 = df.drop(index=[row for row in df.index if 0 == df.loc[row, 'dois_telefones']]) # drop the negative ones
data2_analysis2 = df.drop(index=[row for row in df.index if 1 == df.loc[row, 'dois_telefones']]) # drop the positive ones

data1_analysis3 = df.drop(index=[row for row in df.index if 0 == df.loc[row, 'forneceu_numero_logradouro']]) # drop the negative ones
data2_analysis3 = df.drop(index=[row for row in df.index if 1 == df.loc[row, 'forneceu_numero_logradouro']]) # drop the positive ones

data1_analysis4 = df.drop(index=[row for row in df.index if 0 == df.loc[row, 'forneceu_complemento']]) # drop the negative ones
data2_analysis4 = df.drop(index=[row for row in df.index if 1 == df.loc[row, 'forneceu_complemento']])

data1_analysis5 = df.drop(index=[row for row in df.index if 0 == df.loc[row, 'forneceu_telefone']]) # drop the negative ones
data2_analysis5 = df.drop(index=[row for row in df.index if 1 == df.loc[row, 'forneceu_telefone']])



# checking



# saving modified file for later

df.to_csv('df.csv', index=False)

data1_analysis1.to_csv('data1_analysis1.csv', index=False)
data2_analysis1.to_csv('data2_analysis1.csv', index=False)

data1_analysis2.to_csv('data1_analysis2.csv', index=False)
data2_analysis2.to_csv('data2_analysis2.csv', index=False)

data1_analysis3.to_csv('data1_analysis3.csv', index=False)
data2_analysis3.to_csv('data2_analysis3.csv', index=False)

data1_analysis4.to_csv('data1_analysis4.csv', index=False)
data2_analysis4.to_csv('data2_analysis4.csv', index=False)

data1_analysis5.to_csv('data1_analysis5.csv', index=False)
data2_analysis5.to_csv('data2_analysis5.csv', index=False)




##### treating data more carefuly #####

# checking for missing values in columns

np.unique(df['taxa_ineficiencia_cliente']) #OK




