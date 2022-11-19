import pandas as pd
import numpy as np
import time
from functools import wraps
#np.random.seed(100)

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
    alpha = .88
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
    #data = np.array(pd.read_excel('pca.xlsx', usecols = features_used + response_index ).replace({'Class' : { 'Çerçevelik' : 0, 'Ürgüp Sivrisi' : 1}}))
    data = np.array(pd.read_excel('Data/Pumpkin_Seeds_Dataset.xlsx', usecols = features_used + response_index  ).replace({'Class' : { 'Çerçevelik' : 0, 'Ürgüp Sivrisi' : 1}}))
    for i in range(N):
        data[:,i] =  (data[:,i] - data[:,i].min()) / (data[:,i].max() - data[:,i].min()) #scaling 0-1
    np.random.shuffle(data)
    return data

def estimate(test, wh, wo):
    #Last forward, with estimated weights
    est = np.zeros(test[:,N:].shape)
    for j in range(len(test[:,:N])):
        _, y = forward(test[:,:N][j].reshape(1,N), wh, wo)
        est[j] = y.reshape(len(y),)
    return np.round(est) #replacing in test array, rounds [0-1] decimal values to 0 or 1



features_used = list(range(12)) #indices de las características a incluir. En este caso todas
response_index = [len(features_used)]


N = len(features_used) #M number of inputs
M = 1 #M number of outputs
L = 5  #L number  of layers
perc_train = 0.7 #ratio para separar en train y test

data = read_data(features_used, response_index)
nrows, ncols = data.shape


train, test = data[:int(nrows*perc_train),:], data[int(nrows*perc_train):,:]

x_train, d_train = train[:,:N], train[:,N:]
wh, wo = training(N, M, x_train, d_train)

estimations = estimate(test, wh, wo)
evaluation = np.reshape(test[:,N] , [int(nrows*(1-perc_train)),1] ) == estimations

print('Listo. Accuracy:')
print(np.sum(evaluation) / int(nrows*(1-perc_train)))
