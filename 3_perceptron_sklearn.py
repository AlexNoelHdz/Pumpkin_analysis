#Modelo Perceptron Multicapa con SKLearn

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

N = 7 # num caracteristicas
features_used = list(range(N)) #indices de las características a incluir. En este caso todas
response_index = [len(features_used)]  # indice de la variable respuesta

#data = np.array(pd.read_excel('Data/Pumpkin_Seeds_Dataset.xlsx', usecols = features_used + response_index  ).replace({'Class' : { 'Çerçevelik' : 0, 'Ürgüp Sivrisi' : 1}}))
#data = np.array(pd.read_excel('Data/pca_sincorrelacion_n4.xlsx', usecols = features_used + response_index ).replace({'Class' : { 'Çerçevelik' : 0, 'Ürgüp Sivrisi' : 1}}))
#data = np.array(pd.read_excel('Data/pca_sincorrelacion_n5.xlsx', usecols = features_used + response_index ).replace({'Class' : { 'Çerçevelik' : 0, 'Ürgüp Sivrisi' : 1}}))
data = np.array(pd.read_excel('Data/Pumpkin_Seeds_Dataset_sin_c.xlsx', usecols = features_used + response_index  ).replace({'Class' : { 'Çerçevelik' : 0, 'Ürgüp Sivrisi' : 1}}))




for i in range(N):
     data[:,i] =  (data[:,i] - data[:,i].min()) / (data[:,i].max() - data[:,i].min()) #scaling 0-1

X = data[:, :N]
Y = data[:, N:]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

# modelo final
clf = MLPClassifier(solver='lbfgs', alpha = 10**-5, hidden_layer_sizes=(2, 2), random_state=1)
clf = clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("Accuracy = ", accuracy_score(y_test, y_pred)*100)


result = permutation_importance(clf,X,Y,scoring='accuracy')
importance = result.importances_mean

for i,v in enumerate(importance):
	print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.title("Importancia de las características, sin corr.")
pyplot.xlabel("Indice de la característica " )
pyplot.show()

MLPClassifier(alpha=1e-05, hidden_layer_sizes=(2, 2), random_state=1, solver='lbfgs')
MLPClassifier