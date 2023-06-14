# My Utility : auxiliars functions

import pandas as pd
import numpy  as np


# Configuration of the DAEs
def load_cnf_dae(ruta_archivo='cnf_dae.csv'):    
    # Línea 1: Número de Clases : 5
    # Línea 2: Número de Frame : 100
    # Línea 3: Tamaño de Frame : 1024
    # Línea 4: Porcentaje Training : 0.8
    # Línea 5: Func. Activación Encoder : 1
    # Línea 6: Max. Iteraciones : 60
    # Línea 7: Tamaño miniBatch : 32
    # Línea 8: Tasa Aprendizaje : 0.001
    # Línea 9: Nodos Encoder1. : 192
    # Línea 10: Nodos Encoder2. : 128
    # Línea 11: Nodos Encoder3. : 64
    
    with open(ruta_archivo, 'r') as archivo_csv:

        conf = [int(i) if '.' not in i else float(i)
              for i in archivo_csv if i != '\n']

    return conf
# Initialize one-wieght    
def iniW(next,prev):
    r = np.sqrt(6/(next+ prev))
    w = np.random.rand(next,prev)
    w = w*2*r-r    
    return(w)
    
# STEP 1: Feed-forward of AE
def dae_forward(X, W, Param):
    # cambiar activaciones por config
    act_encoder = Param[1]

    A = []
    z = []
    Act = []

    # data input
    z.append(X)
    A.append(X)
    
    # iter por la cantidad de pesos
    for i in range(len(W)):
        # print(W[i].shape, X.shape)
        X = np.dot(W[i], X)
        z.append(X)
        if i == 0:
            X = act_function(X, act=act_encoder)

        A.append(X)

    Act.append(A)
    Act.append(z)
    
    return Act


# Activation function
def act_function(x, act=1, a_ELU=1, a_SELU=1.6732, lambd=1.0507):
    
    
    

    # Relu

    if act == 1:
        condition = x > 0
        return np.where(condition, x, np.zeros(x.shape))

    # LRelu

    if act == 2:
        condition = x >= 0
        return np.where(condition, x, x * 0.01)

    # ELU

    if act == 3:
        condition = x > 0
        return np.where(condition, x, a_ELU * np.expm1(x))

    # SELU

    if act == 4:
        condition = x > 0
        return lambd * np.where(condition, x, a_SELU * np.expm1(x))

    # Sigmoid

    if act == 5:
        return 1 / (1 + np.exp(-1*x))

    return x

# Derivatives of the activation funciton
def deriva_act(x, act=1, a_ELU=1, a_SELU=1.6732, lambd=1.0507):

    # Relu

    if act == 1:
        condition = x > 0
        return np.where(condition, np.ones(x.shape), np.zeros(x.shape))

    # LRelu

    if act == 2:
        condition = x > 0
        return np.where(condition, np.ones(x.shape), np.ones(x.shape) * 0.01)

    # ELU

    if act == 3:
        condition = x > 0
        return np.where(condition, np.ones(x.shape), a_ELU * np.exp(x))

    # SELU falta

    if act == 4:
        condition = x > 0
        return lambd * np.where(condition, np.ones(x.shape), a_SELU * np.exp(x))

    # Sigmoid

    if act == 5:
        # pasarle la sigmoid
        return np.multiply(act_function(x, act=5), (1 - act_function(x, act=5)))

    return x


# STEP 2: Feed-Backward for DAE
def gradW(Act, ye, W, Param):   
    L = len(Act[0])-1
    gW = []
     
    M = ye.shape[1]
    Cost = np.sum(np.sum(np.square(Act[0][L] - ye), axis=0)/2)/M
    
    # grad salida
    delta = np.multiply(Act[0][L] - ye, deriva_act(Act[1][L], act=5))
     
    gW_l = np.dot(delta, Act[0][L-1].T)
     
    gW.append(gW_l)
     
    # grad capas ocultas
     
    for l in reversed(range(1,L)):
        
        t1 = np.dot(W[l].T, delta)
         
        t2 = deriva_act(Act[1][l], act=Param[6])
         
        delta = np.multiply(t1, t2)
         
        t3 = Act[0][l-1].T
         
        gW_l = np.dot(delta, t3)
        gW.append(gW_l)
     
    gW.reverse()
    
    return gW, Cost       

# Update DAE's weight via mAdam
def updW_madam(W, V, S, gW, t, Param, b1 = 0.9 , b2 = 0.999, e = 10**-6):
    
    u = Param[7]
    for i in range(len(W)):
        V[i] = b1 * V[i] + (1-b1)(gW[i])
        S[i] = b2 * S[i] + (1-b2)(gW[i])**2
        gAdam = (np.sqrt(b2**t)/(1-b1**t)) *  ((V[i]) / (np.sqrt(S[i]+e)))
        W[i] = W[i] - u * gAdam
    
    return W, V
# Update Softmax's weight via mAdam
def updW_sft_madam(w,v,gw,mu):
    ...    
    return (w,v)

# Softmax's gradient
def gradW_softmax(x,y,a):        
   
    M      = y.shape[1]
    Cost   = -(np.sum(np.sum(  np.multiply(y,np.log(a)) , axis=0)/2))/M
    
    gW     = -(np.dot(y-a,x.T))/M  #cambios aca
    
    return(gW,Cost)

# Calculate Softmax
def softmax(z):
    exp_z = np.exp(z-np.max(z))
    return(exp_z/exp_z.sum(axis=0,keepdims=True))


# save weights DL and costo of Softmax
def save_w_dl(W,Ws,Cost):    
    np.savez('wAEs.npz', W[0], W[1])
    np.savez('wSoftmax.npz', Ws)
    
    
    df = pd.DataFrame( Cost )
    df.to_csv('costo.csv',index=False, header = False )
