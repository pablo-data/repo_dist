import pandas     as pd
import numpy      as np

#gets Index for n-th Frame
def get_Idx_n_Frame(n,l):
    
    Idx = (n*l , (n*l) +l)
    
    return(Idx) #tuple de indices

# Crate new Data : Input and Label 
def create_input_label(Data, Param):
    
    # calcular index de cada frame
    # for i a cant de frames N:
    #     sacar segmento de largo Label
    #     calcular amp. fourier
    #     sacar la mitad de eso (/2)
    #     concat de los frames
    #     concat a la matrix con la anterior
    # calcular etiquetas binarias
    # unir con matrix de X
    # norm data
    # reordenar filas
    # dividir en train y test
    
    nFrame = Param[1]
    lFrame = Param[2]
   
    #optimizar para realizar todas las columnas juntas sin for, 
    F = []
    F_label = []
    for class_n in range(Param[0]):
        for column in range(Data[class_n].shape[1]):
            X = Data[class_n][:,column]
            for n in range(nFrame):
               
               #se puede hacer uso de la misma funcion de los batch para los frames
                Idx = get_Idx_n_Frame(n,lFrame)
                x_frame = X[slice(*Idx)]
        
                x = np.fft.fft(x_frame)
                x = x[:len(x)//2]
                
                F.append(x)
        F_label.append(np.ones(nFrame)*class_n)
        
    F_label = binary_label(F_label)
    
    F = data_norm(F)
    
    # df_x = pd.DataFrame.from_records(F)
    # df_x = df_x.add_prefix('x_')
    
    # df_y = pd.DataFrame.from_records(F_label)
    # df_y = df_y.add_prefix('y_')
    
    # df = pd.concat([df_x, df_y], axis=1)
    
    # p = Param[3]
    
    # df_train = df.sample(frac=p)
    # df_test = df.drop(df_train.index)
    
    # Xe = df_train.filter(regex='x_')
    # Ye = df_train.filter(regex='y_')
    
    # Xv = df_test.filter(regex='x_')
    # Yv = df_test.filter(regex='y_')
    
    return F, F_label

# normalize data
def data_norm(x, a = 0.01, b = 0.99):

    x_max = np.max(x)
    x_min = np.min(x)
    x = ( ( ( x - x_min )/( x_max - x_min ) ) * ( b - a ) ) + a
    
    return x

# Save Data : training and testing
def save_data_csv(X, Y, Param):
     
    df_x = pd.DataFrame.from_records(X)
    df_x = df_x.add_prefix('x_')
    
    df_y = pd.DataFrame.from_records(Y)
    df_y = df_y.add_prefix('y_')
    
    df = pd.concat([df_x, df_y], axis=1)
    
    p = Param[3]
    
    df_train = df.sample(frac=p)
    df_test = df.drop(df_train.index)
    
    df_train.to_csv('train.csv')
    df_test.to_csv('test.csv')
    
    return

# Binary Label
def binary_label(classes):
    n_class = np.max(classes)
    classes = classes -1
    
    label = np.zeros( (classes.shape[0],n_class) )
    label[np.arange(0,len(classes)),classes] = 1
    return label

# Load data csv
def load_class_csv(Param):
    n_class = Param[0] 
    
    path = 'DATA'
    
    Data = []
    
    for n in range(n_class):
        
        path_csv = path + '\class'+str(n+1)+'.csv'
        Data_class = np.genfromtxt(path_csv, delimiter=',')
        Data.append(Data_class)
    
    return Data


# Beginning ...
def main():        
    Param           = ut.load_cnf_dae()	
    Data            = load_class_csv(Param)
    x,y     = create_input_label(Data)
    save_data_csv(x,y,Param)
    

if __name__ == '__main__':   
	 main()


