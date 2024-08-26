#Importar las librerias que vamos a usar 
import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

#importamos el dataframe 
data=pd.read_csv("nissan-dataset.csv")
data 
#analizamos y limpiamos los datos 
data.shape
data.info()
#miramos los datos faltantes
data.isna().sum()
#tenemos que rellenar los nulos de algunas columnas 
#visualización de la edad según la frequencia
data['age'].plot(kind='hist', title='Age Column')
data['age'].fillna(data['age'].mean(), inplace=True)
data['performance'].fillna(data['performance'].mean(), inplace=True)
#visualización de la columna km totales de conduccion de cada coche según la frequencia
data['km'].plot(kind='hist', title='km Column')
data['km'].fillna(data['km'].mean(), inplace=True)
data.isna().sum()
data.dropna(inplace=True)
data.isna().sum()#verificamos que el dataset este libre de nulos
data.shape
data.columns
data.drop(['id', 'full_name'], axis=1, inplace=True)
data.shape
#nos centramos solo en unos pocos ejemplos 
data.head()
#sacamos los valores únicos de cada columna 
print("Unique Values in color column: ", data['color'].nunique())
print("Unique Values in gender column: ", data['gender'].nunique())
print("Unique Values in model column: ", data['model'].nunique())
print("Unique Values in condition column: ", data['condition'].nunique())
data['gender'].unique()
#convertimos las columnas categoricas con dummies 
cod_df = pd.get_dummies(data, columns=['color', 'gender', 'model', 'condition'])
cod_df
print(cod_df.columns)
