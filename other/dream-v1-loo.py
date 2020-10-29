#Create a label.csv file with true locations of 1297 cells.
#Optimization: using pairs to select training and test samples.
#Test different models. Add dropouts. Use Softplus. Use Sigmoid and binary_crossentropy.
#Check convergence using all 84 in situ genes.
#Feature reduction: leave one out and Garson's methods.
#Use correlation test to validate feature reduction results.
#Test with 60 genes.
#Next step: use ensemble.
import numpy as np
import pandas as pd
#import pydot
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, concatenate, Flatten, Dropout, Lambda, Activation, BatchNormalization, LocallyConnected1D, Reshape
from keras.utils import plot_model
import time
import itertools
import random
import sys

print(time.ctime(), 'Read data')
num_situ = 84
num_all = 8924
b = pd.read_csv('b.csv')
#Changes 'na' to 'naa' and 'nan' to 'nana'
d = pd.read_csv('d.csv', index_col=0, header=None, encoding='ISO-8859-1').T
labels = pd.read_csv('labels.csv', index_col=0, header=None).T
#Move in-situ 84 genes to the begining
cols = list(b) + list(set(list(d)) - set(list(b)))
d = d.loc[:,cols]
d = d.div(d.sum(axis=1), axis=0)
d.reset_index(drop=True, inplace=True)
labels.reset_index(drop=True, inplace=True)

#Create the true label pairs. Left: d-array, right: b-array
#Labels start from 1 in the original file. They indicate a specific row in b table.
print(time.ctime(),'Create true list of tuples')
sys.stdout.flush()
d['label'] = labels['label'] - 1
d['one'] = 1
d_true = list(zip(d.index, d.label, d.one))
d = d.drop(['one', 'label'], 1)

#Create the false label pairs
print(time.ctime(), 'Create false list of tuples')
a_list = [i for i in range(0,1297)] # d-array
b_list = [j for j in range(0,3039)] # b-array
d_prod = list(itertools.product(a_list, b_list))
d_false = [x+(0,) for x in d_prod if x not in d_true]

#Merge the two lists. Select 10003 samples from d and 1297 from b for training.
print(time.ctime(), 'Merging lists')
indicies = random.sample(range(len(d_false)), 10003)
d_false1 = [d_false[i] for i in indicies]
d_list = d_true + d_false1
random.shuffle(d_list)
len_list = len(d_list)
print(time.ctime(), f'len(d_list): {len_list}') #11300

print(time.ctime(), 'Create train input arrays')
X1_train = np.empty((len_list, num_situ)) #Can create a test array using X1_test = np.empty((1300, 84))
X2_train = np.empty((len_list, num_situ))
X3_train = np.empty((len_list, num_all - num_situ)) #8840
y_train = np.empty((len_list), dtype=int)
batch=0
for i in d_list[0:len_list]:
    try:
        X1_train[batch] = b.iloc[i[1]]
        X2_train[batch] = d.iloc[i[0]][0:num_situ]
        X3_train[batch] = d.iloc[i[0]][num_situ:]
        y_train[batch] = i[2]
    except:
        print('Exception in train.............', i)
    finally:
        batch = batch + 1

#Model build
print(time.strftime("%H:%M:%S"), ' Model build')
#First input model
input_a = Input(shape=(num_situ,))
dense_a = Dense(200, activation='softplus')(input_a)

#Second input model
input_b = Input(shape=(num_situ-1,))
dense_b = Dense(200, activation='softplus')(input_b)

#Third input model
input_c = Input(shape=(num_all - num_situ,))
dense_c = Dense(50, activation='softplus')(input_c)
drop_c = Dropout(0.2)(dense_c)

concat_a = concatenate([dense_a, dense_b])
dense_d = Dense(num_situ, activation='softplus')(concat_a)
drop_d = Dropout(0.2)(dense_d)

concat_b = concatenate([drop_d, drop_c])
dense_e = Dense(50, activation='softplus')(concat_b)
dense_f = Dense(1, activation='sigmoid')(dense_e)
model = Model(inputs=[input_a, input_b, input_c], outputs=[dense_f])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.save_weights('model.h5')
print(model.summary())
#plot_model(model, to_file='my_model.png')

print(time.strftime("%H:%M:%S"), ' Fit')
sys.stdout.flush()
#Try differnet training ommitting one gene at a time.
val_acc=np.empty((num_situ,4))
for i in range(num_situ):
    model.load_weights('model.h5')
    X2_temp = np.delete(X2_train, i, axis=1)
    #tbCallBack = keras.callbacks.TensorBoard(log_dir='.', histogram_freq=0, write_graph=True, write_images=True)
    history = model.fit(x=[X1_train, X2_temp, X3_train],
                        y=y_train,
                        batch_size=50,
                        epochs=100,
                        verbose=0,
                        validation_split=0.3,
                        class_weight={0:1, 1:10}) #, use_multiprocessing=True, workers=8) #, callbacks=[tbCallBack])
    val_acc[i,0] = np.average(history.history['val_acc'])
    val_acc[i,1] = np.max(history.history['val_acc'])
    val_acc[i,2] = np.average(history.history['acc'])
    val_acc[i,3] = np.max(history.history['acc'])
    print(time.ctime(), f'i: {i}, val_acc average: {val_acc[i,0]}, max: {val_acc[i,1]}')
    sys.stdout.flush()

np.save('val_acc.npy', val_acc)