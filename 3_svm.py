# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 20:36:05 2022

@author: -
"""


import sympy as sp
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sympy import Symbol, solveset, S, erf, log, sqrt, diff,Float
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# linear regression feature importance
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot


datos=pd.read_excel('Data/Pumpkin_Seeds_Dataset.xlsx')
#correlacion=datos.corr()
#checando nulos
datos.isnull().sum()
#checando cuantos hay de cada tipo
datos['Class'].value_counts()
datos['Class'] = [ 1 if each == 'Çerçevelik' else 0 for each in datos['Class']]
#splitting data

x = datos.drop('Class', axis = 1)
y = datos.Class.values


#correlation
plt.figure(figsize  = (10,10))
data = x.corr()
sns.heatmap(data, annot = True)

#normalizar datos

#Normales y sin outliers
scaler=StandardScaler()
scaler.fit(x)
scaled_data=scaler.transform(x)



#%% Algoritmo pca
pca=PCA()
#pca=PCA(n_components=10) # Indicar el número de componentes principales
pca.fit(scaled_data)

# Ponderación de los componentes principales (vectores propios)
pca_score=pd.DataFrame(data    = pca.components_, columns = x.columns,)

# Mapa de calor para visualizar in influencia de las variables
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 7))
componentes = pca.components_
plt.imshow(componentes.T, cmap='plasma', aspect='auto')
plt.yticks(range(len(x.columns)), x.columns)
plt.xticks(range(len(x.columns)), np.arange(pca.n_components_)+ 1)
plt.grid(False)
plt.colorbar();


# Pesos
loading_scores = pd.DataFrame(pca.components_[1])
#Nombre de las columnas
df = pd.DataFrame(datos)
loading_scores.index=x.columns
# Ordena de mayor a menor los pesos
sorted_loading_scores = loading_scores[0].abs().sort_values(ascending=False)

per_var=np.round(pca.explained_variance_ratio_*100, decimals=1)
# Porcentaje de varianza acumulado de los componentes
porcent_acum = np.cumsum(per_var)

#a partir del quinto componente ya no cambias
pca=PCA(n_components=5) # Indicar el número de componentes principales

df=pd.DataFrame( pca.fit_transform(scaled_data))

#extra
#feature importance
#Linear Regression Feature Importance

# define the model
model = LinearRegression()
# fit the model
model.fit(scaled_data, y)
# get importance
importance = model.coef_
# summarize feature importance
for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()


df.to_excel('pca.xlsx')

#empezando modelo dividir train test

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size = 0.2, random_state = 42)



#SVM for classification
svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print("Accuracy = ", accuracy_score(y_test, y_pred)*100)
