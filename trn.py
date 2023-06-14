#Training DL via mAdam

import pandas     as pd
import numpy      as np
import utility    as ut


# Training miniBatch 
def train_sft_batch(X, Y, W, V, Param):
    costo = []
    M = Param[2]
    numBatch = np.int16(np.floor(X.shape[1]/M))   
    for i in range(numBatch):   
        Idx = get_Idx_n_Batch(n, M)
        xe, ye = X[:,slice(*Idx)], Y[:,slice(*Idx)]
        
        z = np.dot(W, xe)
        # if np.isnan(z).any():
        #     print(W)
        #     print(a)
        #     print(gW)
        a = ut.softmax(z)
        
        gW, Cost = ut.gradW_softmax(xe, ye, a ,W)
        
        W, V = ut.updWV_RMSprop(W, V, gW, tasa=Param[1])

        costo.append(Cost)     
    return(W,V,costo)

# Softmax's training via mAdam
def train_softmax(x,y,param):
    W,V,S    = ut.iniW(...)    
    ...    
    for Iter in range(1,par1[0]):        
        idx   = np.random.permutation(X.shape[1])
        xe,ye = X[:,idx],Y[:,idx]   
        
        W, V, c = train_sft_batch(xe, ye, W, V, Param)

        Cost.append(np.mean(c))
        
        if Iter % 10 == 0:
            print('\tIterar-SoftMax: ', Iter, ' Cost: ', Cost[Iter-1])
               
    return(W,Costo)    
 
# Training by using miniBatch
def train_dae_batch(x,w1,v,w2,Param):
    
    numBatch = np.int16(np.floor(x.shape[1]/Param[3]))
    cost = []
    W[1] = ut.pinv_ae(X, ut.act_function(np.dot(W[0], x), act=Param[1]), Param[0])  
    
    for i in range(numBatch):                
        Idx = get_Idx_n_Batch(n, Param[3])
        xe= x[:,slice(*Idx)]
        
        Act = ut.forward_ae(xe, W, Param)
        
        gW, Cost = ut.gradW_ae(Act, W, Param)
        
        W_1, v_1 = ut.updWV_RMSprop(W, v, gW, tasa = Param[4])
      
        W[0], v[0] = W_1[0], v_1[0]
        
        cost.append(Cost)             
    return W, v, cost

# DAE's Training 
def train_dae(x,Param):        
    # W,V,S = ut.iniW()     
    
    W, v = ut.iniWs(x.shape[0], Param)
    
    Cost = []
    for Iter in range(1,Param[2]+1):        
        xe     = x[:,np.random.permutation(x.shape[1])]                
        
        W, v, c = train_dae_batch(xe, W, v, Param)
        
        Cost.append(np.mean(c))
        if Iter % 10 == 0:
            print('\tIterar-AE: ', Iter, ' Cost: ', Cost[Iter-1])

    return W, Cost 



#load Data for Training
def load_data_trn():
    ...    
    return(xe,ye)    


# Beginning ...
def main():
    p_dae,p_sft = ut.load_cnf_dae()           
    xe,ye       = load_data_trn()   
    W,Xr        = train_dae(xe,p_sae)         
    Ws, cost    = train_softmax(Xr,ye,...)
    ut.save_w_dl(W,Ws,cost)
       
if __name__ == '__main__':   
	 main()

