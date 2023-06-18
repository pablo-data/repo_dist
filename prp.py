import pandas     as pd
import numpy      as np
import utility    as ut

# Crate new Data : Input and Label 
def create_input_label(Data, Param):
    
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
                Idx = ut.get_Idx_n(n,lFrame)
                x_frame = X[slice(*Idx)]
        
                x = np.fft.fft(x_frame)
                
                x = x[:len(x)//2]
                                
                x = np.abs(x)
                
                F.append(x)
                        
        F_label.extend(np.ones(nFrame*Data[class_n].shape[1])*class_n)
      
    F_label = binary_label(F_label)
    
    F = data_norm(F)
    
    return F, F_label

def shuffle_data(X, Y):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return np.array(X)[indices], np.array(Y)[indices]

# normalize data
def data_norm(x, a = 0.01, b = 0.99):

    x_max = np.max(x)
    x_min = np.min(x)
    x = ( ( ( x - x_min )/( x_max - x_min ) ) * ( b - a ) ) + a
    
    return x

# Save Data : training and testing
def save_data_csv(X, Y, Param):
    df_x = pd.DataFrame(X).add_prefix('x_')
    df_y = pd.DataFrame(Y).add_prefix('y_')
    
    df = pd.concat([df_x, df_y], axis=1)
    
    p = Param[3]
    
    df_train = df.sample(frac=p)
    df_test = df.drop(df_train.index)
    
    df_train.to_csv('train.csv', index=False)
    df_test.to_csv('test.csv', index=False)
    
    return


# Binary Label
def binary_label(classes):
    
    n_class = int(np.max(classes))+1
    classes = np.asarray(classes).astype(int)
    # classes = classes - 1
    
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
    Param           = ut.load_cnf_dae()[0]
    Data            = load_class_csv(Param)
    x,y     = create_input_label(Data, Param)
    x, y = shuffle_data(x, y)
    save_data_csv(x,y,Param)
    

if __name__ == '__main__':   
	 main()


