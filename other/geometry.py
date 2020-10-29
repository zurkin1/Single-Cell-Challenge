"""
Add to .bashrc
function f(){
nohup python $1.py > $1.log &
}
"""
import zipfile
import boto3
import io
import os.path

if not os.path.isfile('./data/b.csv'):
   print('Reading files from AWS')
   session = boto3.session.Session(
        aws_access_key_id="AKIAJIQ7JIXG4KRICKRA",
        aws_secret_access_key="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"
   )
   s3 = session.resource("s3")
   bucket = s3.Bucket('daniglassbox')
   obj = bucket.Object('d.zip')

   with io.BytesIO(obj.get()["Body"].read()) as tf:
      # rewind the file
      tf.seek(0)
      # Read the file as a zipfile and process the members
      with zipfile.ZipFile(tf, mode='r') as zipf:
          for subfile in zipf.namelist():
              print(subfile)
          zipf.extractall('./data/')
else:
   print('Files exist')


import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, concatenate, Flatten, Dropout, Lambda, Activation, BatchNormalization, LocallyConnected1D, Reshape, AlphaDropout
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import time
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle

num_all = 8924
num_situ = 84
glist_60 = [3,16,80,77,19,52,53,57,78,68,62,0,75,21,66,26,81,51,63,7,8,56,35,18,83,6,1,61,65,55,74,22,64,20,59,23,79,48,58,31,69,73,76,24,33,17,47,14,25,15,67,42,54,46,50,28,27,49,43,13]

#Create the true label pairs. Left: d-array, right: b-array
#Labels start from 0 in the original file. They indicate a specific row in b table.
print(time.ctime(),'Create true list of tuples')
b = pd.read_csv('data/b.csv')
d = pd.read_csv('data/magic_dge.csv')
#labels.pkl contains a dictionary mapping of all 1270 cells to (possibly few) locations in [0,3038].
pkl_file = open('data/labels.pkl', 'rb')
ind_load = pickle.load(pkl_file)
pkl_file.close()
data_ind = pd.DataFrame(list(ind_load.items()))
data_ind.drop([0], axis=1, inplace=True)
data_ind[1] = [np.ndarray.flatten(data_ind[1][i]) for i in range(len(data_ind))]
d_true = []
for i in range(len(data_ind)):
    for j in range(len(data_ind.iloc[i].iloc[0])):
        d_true.append((i,data_ind.iloc[i].iloc[0][j],1))
print(time.ctime(), 'Length of dataset: ', len(d_true)) #1693

print(time.ctime(), 'Create train input array')
#Create the labels
geom = pd.read_csv('data/geometry.csv')
scaler = MinMaxScaler()
#geom = scaler.fit_transform(geom)
data = pd.read_csv('data/g2f.csv')
data.drop(['NCBI  Gene ID'], inplace=True, axis=1)
data.fillna(0, inplace=True)
#Remove columns with only zeros or NAs.
data = data.loc[:, ~(data==0).all(axis=0)]
#data.shape # 1693 columns (functions), 66 rows (only 66 genese had data in Gene2Function)

#Model build
print(time.ctime(), 'Model build')
b_in = Input(shape=(len(data.columns)-1,)) #Input for each cell is now in the function dimension (i.e. not in the gene dimension).
b1 = Dense(100)(b_in)
b1 = LeakyReLU()(b1)
b1 = AlphaDropout(0.2)(b1)
"""
b2 = concatenate([b1, b_in])
b2 = Dense(100)(b2)
b2 = LeakyReLU()(b2)
b2 = AlphaDropout(0.2)(b2)

b3 = concatenate([b2,b_in])
b3 = Dense(100)(b3)
b3 = LeakyReLU()(b3)
b3 = AlphaDropout(0.2)(b3)

x = Flatten()(b1)
x = Dense(200)(x)
x = Activation('softplus')(x)
x = Dense(100)(x)
x = Activation('softplus')(x)
x = Dense(50)(x)
x = Activation('softplus')(x)
"""
x = Dense(20)(b1)
x = Activation('softplus')(x)
x = Dense(1)(x)
output = Activation('softplus')(x)

model = Model(inputs=[b_in], outputs=[output])
adam = Adam() #clipnorm=1.0)
model.compile(optimizer=adam, loss='mse')
print(model.summary())

print(time.ctime(), 'Create train input array')
X2_train = np.empty((len(d_true), len(data.columns)-1))
#X3_train = np.empty((len(d_true), num_all - num_situ)) #8840
y_train = np.empty((len(d_true), 1))
batch=0
columns = list(data.columns)
columns.remove('Symbol')

for i in d_true:
    try:
        #Copy the values of the in-situ genes for the specific cell in d.
        factor = d.iloc[i[0]][data.Symbol.values]
        #Multiply the gene function matrix by factor
        X2_train[batch] = data[columns].multiply(factor.values, axis='index').sum()
        #data_1 = scaler.fit_transform(data_1)
        #X3_train[batch] = d.iloc[i[0]][num_situ:]
        y_train[batch] = geom.iloc[i[1]][1]
    except:
        print('Exception in train.............', i)
    finally:
        batch = batch + 1

print(time.ctime(), 'Model fit')
history = model.fit(x=[X2_train],
                    y=y_train,
                    batch_size=5,
                    epochs=1000,
                    verbose=2,
                    validation_split=0.3)
model.save('data/geom_model.hm')
with open('data/history_geom.pkl', 'wb') as fl:
    pickle.dump(history, fl)
    fl.close()
