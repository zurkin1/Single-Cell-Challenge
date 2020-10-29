#Calculate locations of 1297 cells
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import confusion_matrix
import pickle
import math
import pandas as pd
import numpy as np
import sys

def my_mcc(list1, list2):
    tn, fp, fn, tp = confusion_matrix(list1, list2).ravel()
    my_mcc = (tp*tn - fp*fn) / math.sqrt( (tp + fp)*(tp + fn)*(tn + fp)*(tn + fn) )
    return(my_mcc)

num_situ = 40
glist = ['danr','CG14427','dan','CG43394','ImpL2','Nek2','CG8147','Ama','Btk29A','trn','numb','prd','brk','tsh','pxb','dpn','ftz','Kr','h','eve','Traf4','run','Blimp-1','lok','kni','tkv','MESR3','odd','noc','nub','Ilp4','aay','twi','bmm','hb','toc','rho','CG10479','gt','gk']

bdtnp = pd.read_csv('data/binarized_bdtnp.csv')[glist] #(3039, 84)
#DGE (RNA_seq data) contains cell values for the in-citu genes only.
dge = pd.read_csv('data/dge_binarized_distMap.csv').T[glist] #(84, 1297)
dge.reset_index(drop=True, inplace=True)

ind_first = {}
ind_second = {}
i = 0
ind = {}
for index, row in dge.iterrows():
    print(i, ' ', end="")
    sys.stdout.flush()
    my_list = list(mcc(bdtnp.iloc[i], row) for i in range(3039))
    my_list_2 = my_list.copy()
    my_list_2.sort()
    #ind[i] = np.ndarray.flatten(np.argwhere(my_list == (np.amax(my_list) or np.amax(my_list.remove(np.amax(my_list)))))) #Index in bdtnp is zero based.
    #fisrt = np.ndarray.flatten(np.argwhere(my_list==my_list_2[-1]))
    #second = np.ndarray.flatten(np.argwhere(my_list==my_list_2[-2]))
    #ind[i] = np.concatenate([first, second])
    lis_len = len(np.argwhere(my_list >= my_list_2[-10]))
    #if(lis_len < 10):
    #    ind[i] = np.concatenate((np.argwhere(my_list > my_list_2[-10]),
    #                             np.argwhere(my_list == my_list_2[-10])[0:(10-lis_len)]))
    #else:
    ind[i] = np.argwhere(my_list >= my_list_2[-11])
    #if(i>50):
    #    break
    i = i + 1

output = open(f'data/maxcc_{num_situ}_my_list_11.pkl', 'wb')
pickle.dump(ind, output)
output.close()

#Translate to dataframe
data_ind = pd.DataFrame(list(ind.items()))
data_ind.drop([0], axis=1, inplace=True)
data_ind[1] = [np.ndarray.flatten(data_ind[1][i]) for i in range(len(data_ind))]
data_ind.head()
df2 = pd.DataFrame(data_ind[1].values.tolist())
df2.to_csv(f'data/maxmcc_{num_situ}_my_list_11.csv')