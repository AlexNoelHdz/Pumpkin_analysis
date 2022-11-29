import sympy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, solveset, S, erf, log, sqrt, diff,Float
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import svm
from sklearn.metrics import accuracy_score
# linear regression feature importance
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
from functools import wraps
from sklearn.linear_model import LogisticRegression

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


#%% correlation
plt.figure(figsize  = (10,10))
data = x.corr()
sns.heatmap(data, annot = True)

#quitamos columnas correlacionadas (compactness,aspect_ratio,eccentricity, roundness)
#(area,convex area,equis_diameter)

X = x.drop(['Compactness','Roundness','Eccentricity' ,'Convex_Area','Equiv_Diameter'],axis=1)


#%% #Normales y sin outliers
scaler=StandardScaler()
scaler.fit(X)
scaled_data=scaler.transform(X)



scalerori=StandardScaler()
scalerori.fit(x)
scaled_dataori=scalerori.transform(x)



#%% Algoritmo pca
pca=PCA()
#pca=PCA(n_components=10) # Indicar el número de componentes principales
pca.fit(scaled_data)

# Ponderación de los componentes principales (vectores propios)
pca_score=pd.DataFrame(data    = pca.components_, columns = X.columns,)


# Pesos
loading_scores = pd.DataFrame(pca.components_[1])
#Nombre de las columnas

loading_scores.index=X.columns
# Ordena de mayor a menor los pesos
sorted_loading_scores = loading_scores[0].abs().sort_values(ascending=False)

per_var=np.round(pca.explained_variance_ratio_*100, decimals=1)
# Porcentaje de varianza acumulado de los componentes
porcent_acum = np.cumsum(per_var)

#con 4 compenetes es decente
pca=PCA(n_components=4) # Indicar el número de componentes principales

df=pd.DataFrame( pca.fit_transform(scaled_data))
pca_sincorrelacion_n4=df.copy()

#a partir del quinto componente ya no cambias
pca5=PCA(n_components=5) # Indicar el número de componentes principales
df5=pd.DataFrame( pca5.fit_transform(scaled_data))
pca_sincorrelacion_n5=df5.copy()




df.to_excel('pca_sincorrelacion_n4.xlsx')
#%% SVC con PCA n=4
#empezando modelo dividir train test

x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(df, y, test_size = 0.2, random_state = 42)

##modelo final
svc = svm.SVC().fit(x_train_pca, y_train_pca)
y_pred_pca = svc.predict(x_test_pca)
print("Accuracy = ", accuracy_score(y_test_pca, y_pred_pca)*100)
#Accuracy =  87.0


#%% SVC con PCA n=5
#empezando modelo dividir train test

x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(pca_sincorrelacion_n5, y, test_size = 0.2, random_state = 42)

##modelo final
svc = svm.SVC().fit(x_train_pca, y_train_pca)
y_pred_pca = svc.predict(x_test_pca)
print("Accuracy = ", accuracy_score(y_test_pca, y_pred_pca)*100)
#Accuracy =  87.0




#%% SVC sin PCA sin correlacion

x_train, x_test, y_train, y_test = train_test_split(scaled_data, y, test_size = 0.2, random_state = 42)


##modelo final
svc2 = svm.SVC().fit(x_train, y_train)
y_pred = svc2.predict(x_test)
print("Accuracy = ", accuracy_score(y_test, y_pred)*100)
#Accuracy =  87.0


#%% SVC con datos originales

x_train, x_test, y_train, y_test = train_test_split(scaled_dataori, y, test_size = 0.2, random_state = 42)


##modelo final
svc2 = svm.SVC().fit(x_train, y_train)
y_pred = svc2.predict(x_test)
print("Accuracy = ", accuracy_score(y_test, y_pred)*100)
#Accuracy =  86.8


#%% LR con PCA n=4
x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(df, y, test_size = 0.2, random_state = 42)
# define the model
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train_pca, y_train_pca)
y_pred_lr = logisticRegr.predict(x_test_pca)
print("Accuracy = ", accuracy_score(y_test_pca, y_pred_lr)*100)
#Accuracy =  84.2


#%% LR con PCA n=5
x_train_pca, x_test_pca, y_train_pca, y_test_pca = train_test_split(pca_sincorrelacion_n5, y, test_size = 0.2, random_state = 42)
# define the model
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train_pca, y_train_pca)
y_pred_lr = logisticRegr.predict(x_test_pca)
print("Accuracy = ", accuracy_score(y_test_pca, y_pred_lr)*100)
#Accuracy =  84.2



#%% LR sin PCA
# define the model
x_train, x_test, y_train, y_test = train_test_split(scaled_data, y, test_size = 0.2, random_state = 42)
logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
y_pred_lr2 = logisticRegr.predict(x_test)
print("Accuracy = ", accuracy_score(y_test, y_pred_lr2)*100)
#Accuracy =  85.2

#%% LR datos original
# define the model

x_train, x_test, y_train, y_test = train_test_split(scaled_dataori, y, test_size = 0.2, random_state = 42)

logisticRegr = LogisticRegression()
logisticRegr.fit(x_train, y_train)
y_pred_lr2 = logisticRegr.predict(x_test)
print("Accuracy = ", accuracy_score(y_test, y_pred_lr2)*100)
#Accuracy =  85.6
