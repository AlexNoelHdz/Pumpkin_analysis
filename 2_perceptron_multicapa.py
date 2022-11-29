import pandas as pd
import numpy as np
import time
from functools import wraps

def print_duration(func):
    """Decorador de Python para imprimir la duracion de un metodo
    """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Función toma {total_time:.10f} seconds')
        return result
    return timeit_wrapper

def forward(x, w_h, w_o):
    a = 1
    net_h = w_h@x.T
    yh = 1/(1 + np.e**(-a*net_h)) #sigmoid function
    net_o = w_o @ yh
    y = 1/(1 + np.e**(-a*net_o)) #sigmoid function
    return yh, y

def backward(x, w_h, w_o, d, alpha):
    yh, y = forward(x, w_h, w_o)
    delta_o = (d.T-y) * y * (1-y)
    delta_h = yh * (1-yh) * (w_o.T@delta_o)
    dwo = alpha*delta_o @ yh.T
    dwh = alpha*delta_h @ x
    return dwo, dwh, np.linalg.norm(delta_o)

@print_duration
def training(N, M, x, d):
    alpha = 0.8
    #print(L)
    w_h = np.random.uniform(-1, 1, (L,N))
    w_o = np.random.uniform(-1, 1, (M,L))
    while (True) :
        for j in range(len(x)) :
            dwo, dwh, norm = backward(x[j].reshape(1,N), w_h, w_o, d[j].reshape(1,M), alpha)
            w_o = w_o + dwo #adjusting output weights
            w_h = w_h + dwh #adjusting hidden weights
        if norm < 10 **-1 :
            #if condition to stop adjusting is sattisfied
            break
    return w_h, w_o

def read_data(features_used, response_index):
    N = len(features_used)
    #data = np.array(pd.read_excel('Data/pca_sincorrelacion_n4.xlsx', usecols = features_used + response_index ).replace({'Class' : { 'Çerçevelik' : 0, 'Ürgüp Sivrisi' : 1}}))
    #data = np.array(pd.read_excel('Data/pca_sincorrelacion_n5.xlsx', usecols = features_used + response_index ).replace({'Class' : { 'Çerçevelik' : 0, 'Ürgüp Sivrisi' : 1}}))

    #data = np.array(pd.read_excel('Data/Pumpkin_Seeds_Dataset.xlsx', usecols = features_used + response_index  ).replace({'Class' : { 'Çerçevelik' : 0, 'Ürgüp Sivrisi' : 1}}))
    data = np.array(pd.read_excel('Data/Pumpkin_Seeds_Dataset_sin_c.xlsx', usecols = features_used + response_index  ).replace({'Class' : { 'Çerçevelik' : 0, 'Ürgüp Sivrisi' : 1}}))

    for i in range(N):
        data[:,i] =  (data[:,i] - data[:,i].mean()) / np.std(data[:,i]) #scaling 0-1

    np.random.shuffle(data)
    return data

def estimate(test, wh, wo):
    #Last forward, with estimated weights
    est = np.zeros(test[:,N:].shape)
    for j in range(len(test[:,:N])):
        _, y = forward(test[:,:N][j].reshape(1,N), wh, wo)
        est[j] = y.reshape(len(y),)
    return np.round(est) #replacing in test array, rounds [0-1] decimal values to 0 or 1



features_used = list(range(7)) #indices de las características a incluir. En este caso todas
response_index = [7]


N = len(features_used) #M number of inputs
M = 1 #M number of outputs
perc_train = 0.7 #porcentaje a usar como train

data = read_data(features_used, response_index)
nrows, ncols = data.shape

simulaciones = []  #lista que contendra las 100 simulaciones (para guardar y promediar)
promedios_simulaciones = []  # Lista que contendra el promedio para cada n diferente usada (3 neuronas, 4, 5, etc)

for n in range ( 3, 16 ): #numero de neuronas usadas
    L = n   #L number  of neurons
    for i in range( 100 ) : #ajusta para n neuronas
        train, test = data[:int(nrows*perc_train),:], data[int(nrows*perc_train):,:] #separacion data en test/train
        x_train, d_train = train[:,:N], train[:,N:]   #separa la columna de respuesta de las features
        wh, wo = training(N, M, x_train, d_train) #estima los pesos de la red
        estimations = estimate(test, wh, wo)   # predice
        evaluation = np.reshape(test[:,N] , [int(nrows*(1-perc_train)),1] ) == estimations # compara (estimo correcto o no?)
        simulaciones.append( np.sum(evaluation) / int(nrows*(1-perc_train)) )  #calcula accuracy
    promedios_simulaciones.append( np.mean(simulaciones) )  #  promedios de accuracy de las 100 sims para cada numero de neuronas se guarda en arreglo
    simulaciones = [] # reinicio simulaciones para que el promedio de la proxima neurona sea independiente

print(promedios_simulaciones)
