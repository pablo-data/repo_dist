import pandas as pd
import numpy as np
import utility as ut


def save_measure(cm, Fsc):

    df_cm = pd.DataFrame(cm)
    df_cm.to_csv('cmatriz.csv', index=False, header=False)

    df_Fsc = pd.DataFrame(Fsc)
    df_Fsc.to_csv('fscores.csv', index=False, header=False)

    return

# load data for testing


def load_data_tst(ruta_archivo='test.csv'):

    df = pd.read_csv(ruta_archivo, converters={'COLUMN_NAME': pd.eval})

    X = df.filter(regex='x_')
    Y = df.filter(regex='y_')

    return np.asarray(X).T, np.asarray(Y).T


# load weight of the DL in numpy format
def load_w_dl():
    ws_ae = np.load('wdae.npz')

    ws_soft = np.load('wSoftmax.npz')

    ws = [ws_ae[i] for i in ws_ae.files]

    ws.extend([ws_soft[i] for i in ws_soft.files])

    return ws


# Feed-forward of the DL
def forward_dl(x, W):

    for i in range(len(W)):
        x = np.dot(W[i], x)
        if i == len(W)-1:
            x = ut.softmax(x)
        else:
            # X = ut.act_function(X, act=2)
            x = ut.act_function(x, act=1)

    return x


# Métrica
def metricas(y, z):
    cm, cm_m = confusion_matrix(y, z)

    Fsc = []
    for i in range(len(cm_m)):
        TP = cm_m[i, 0, 0]
        FP = cm_m[i, 0, 1]
        FN = cm_m[i, 1, 0]
        TN = cm_m[i, 1, 1]

        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        Fsc.append((2 * Precision * Recall) / (Precision + Recall))

    Fsc.append(sum(Fsc)/len(Fsc))

    return cm, np.asarray(Fsc)

# Confusuon matrix


def confusion_matrix(y, z):
    y, z = y.T, z.T
    m = y.shape[0]
    c = y.shape[1]

    y = np.argmax(y, axis=1)

    z = np.argmax(z, axis=1)

    cm = np.zeros((c, c))

    for i in range(m):
        cm[z[i], y[i]] += 1

    cm_m = np.zeros((cm.shape[0], 2, 2))  # matriz confusion por clase

    for i in range(cm.shape[0]):
        cm_m[i, 0, 0] = cm[i, i]  # TP
        cm_m[i, 0, 1] = np.sum(np.delete(cm[i, :], i, axis=0))  # FP
        cm_m[i, 1, 0] = np.sum(np.delete(cm[:, i], i, axis=0))  # FN
        cm_m[i, 1, 1] = np.sum(
            np.delete(np.delete(cm, i, axis=1), i, axis=0))  # TN

    return cm, cm_m


# Beginning ...
def main():
    xv, yv = load_data_tst()
    W = load_w_dl()
    zv = forward_dl(xv, W)
    cm, Fsc = metricas(yv, zv)
    print(Fsc*100)
    print('Fsc-mean {:.5f}'.format(Fsc.mean()*100))
    save_measure(cm,Fsc)


    
if __name__ == '__main__':   
	 main()

