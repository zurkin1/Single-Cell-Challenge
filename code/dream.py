import tensorflow as tf
"""
#Optimizing Tensorflow
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
from keras import backend as K
# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(1234)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
"""
config = tf.ConfigProto()
config.intra_op_parallelism_threads = 10
config.inter_op_parallelism_threads = 10
tf.Session(config=config)

import zipfile
import boto3
import io
import os
import os.path
import numpy as np
import pandas as pd
#import pydot
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, concatenate, Flatten, Dropout, Lambda, Activation, BatchNormalization, LocallyConnected1D, Reshape, AlphaDropout, Conv1D
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import time
import itertools
import random
import sys
import pickle

if not os.path.isfile('./data/bdtnp.csv'):
   print('Reading files from AWS')
   session = boto3.session.Session(
        aws_access_key_id="AKIAJHH3NP6IDC7SRYQQ",
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

print(time.ctime(), 'Read data')
num_situ = 84
num_all = 8924
#glist_60 = [3,16,80,77,19,52,53,57,78,68,62,0,75,21,66,26,81,51,63,7,8,56,35,18,83,6,1,61,65,55,74,22,64,20,59,23,79,48,58,31,69,73,76,24,33,17,47,14,25,15,67,42,54,46,50,28,27,49,43,13]
#glist_20 = [3,16,80,77,19,52,53,57,78,68,62,0,75,21,66,26,81,51,63,7]
#glist_20_knn = [35,41,70,24,14,56,3,64,58,79,27,30,67,44,73,59,49,83,57,16]

#glist = ['kni','Ance','brk','cad','eve','fkh','hb','hkb','ImpE2','oc','sna','srp','twi','zen','zfh1','Blimp-1','croc','D','Dfd','Doc3','dpn','fj','ftz','gt','h','ken','knrl','Kr','odd','peb','run','tkv','tll','tsh','zen2','Antp','apt','bowl','CG14427','CG17724','CG17786','CG8147','Cyp310a1','dan','disco','Doc2','E(spl)m5-HLH','Ilp4','ImpL2','Mes2','NetA','prd','rau','rho','toc','trn','aay','gk','ems','numb']
#glist = ['kni','Ance','brk','cad','eve','fkh','hb','hkb','ImpE2','oc','sna','srp','twi','zen','zfh1','Blimp-1','croc','D','Dfd','Doc3','dpn','fj','ftz','gt','h','ken','knrl','Kr','odd','peb','run','tkv','tll','tsh','zen2','trn','E(spl)m5-HLH','CG17724','disco','dan']
#glist = ['kni','Ance','brk','cad','eve','fkh','hb','hkb','ImpE2','oc','sna','srp','twi','zen','zfh1','gt','Kr','ftz','tkv','croc']

#glist = ['danr','CG14427','dan','CG43394','ImpL2','Nek2','CG8147','Ama','Btk29A','trn','numb','prd','brk','tsh','pxb','dpn','ftz','Kr','h','eve','Traf4','run','Blimp-1','lok','kni','tkv','MESR3','odd','noc','nub','Ilp4','aay','twi','bmm','hb','toc','rho','CG10479','gt','gk','apt','D','sna','NetA','Mdr49','fj','Mes2','CG11208','Doc2','bun','tll','Cyp310a1','Doc3','htl','Esp','bowl','oc','ImpE2','CG17724','fkh']
glist = ['danr','CG14427','dan','CG43394','ImpL2','Nek2','CG8147','Ama','Btk29A','trn','numb','prd','brk','tsh','pxb','dpn','ftz','Kr','h','eve','Traf4','run','Blimp-1','lok','kni','tkv','MESR3','odd','noc','nub','Ilp4','aay','twi','bmm','hb','toc','rho','CG10479','gt','gk']
#glist = ['danr','CG14427','dan','CG43394','ImpL2','Nek2','CG8147','Ama','Btk29A','trn','numb','prd','brk','tsh','pxb','dpn','ftz','Kr','h','eve']

bdtnp = pd.read_csv('data/bdtnp.csv')
#Changes 'na' to 'naa' and 'nan' to 'nana'
#d = pd.read_csv('scimpute_dge.csv', index_col=0, header=None, encoding='ISO-8859-1').T
d = pd.read_csv('data/magic_dge.csv')
#Move in-situ 84 genes to the begining (actually not needed anymore)
#cols = list(bdtnp) + list(set(list(d)) - set(list(bdtnp)))
#d = d.loc[:,cols]
cols = list(bdtnp)

#Scale data to [0,1]
#d = d.div(d.sum(axis=1), axis=0)
#d.reset_index(drop=True, inplace=True)

#Create the true label pairs. Left: d-array, right: b-array
#Labels start from 0 in the original file. They indicate a specific row in b table.
print(time.ctime(),'Create true list of tuples')
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

#Create the false label pairs
print(time.ctime(), 'Create false list of tuples')
"""
a_list = [i for i in range(0,1297)] # d-array
b_list = [j for j in range(0,3039)] # b-array
d_prod = list(itertools.product(a_list, b_list))
d_false = [x+(0,) for x in d_prod if x not in d_true] #Need to remove the ,1 in d_true before running this code.
with open('d_false.pkl', 'wb') as f:
    pickle.dump(d_false, f)
"""
with open('data/d_false.pkl', 'rb') as f:
    d_false = pickle.load(f)

#Merge the two lists. Select 16003 samples from d_false and 1693 (not 1297 due to multiple max(mcc) values) from d_true for training.
print(time.ctime(), 'Merging lists')
indicies = random.sample(range(len(d_false)), 16007) #2307
d_false1 = [d_false[i] for i in indicies]
d_list = d_true + d_false1
random.shuffle(d_list)
len_list = len(d_list)
print(time.ctime(), 'len(d_list): ', len_list) #11300 4000

print(time.ctime(), 'Create train input arrays')
X1_train = np.empty((len_list, num_situ)) #Can create a test array using X1_test = np.empty((1300, 84))
X2_train = np.empty((len_list, num_situ))
X_train = np.empty((len_list, 2, num_situ))
X3_train = np.empty((len_list, num_all)) #8840
y_train = np.empty((len_list), dtype=int)
batch=0
for i in d_list[0:len_list]:
    if(batch % 100 == 0):
        print(batch, ' ', end="")
        sys.stdout.flush()

    X1_train[batch] = bdtnp.iloc[i[1]][cols] #0:num_situ
    X2_train[batch] = d.iloc[i[0]][cols]
    X_train[batch] = np.vstack([X1_train[batch], X2_train[batch]])
    X3_train[batch] = d.iloc[i[0]] #[num_situ:] #[list(set(d.columns) - set(glist))]
    y_train[batch] = i[2]
    batch = batch + 1

#Model build
sys.stdout.flush()
print(time.strftime("%H:%M:%S"), ' Model build')

def matthews_correlation(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def matthews_correlation_loss(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos =  y_pred
    y_pred_neg = 1 - y_pred_pos

    y_pos = y_true
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = K.square(tp * tn - fp * fn)
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)

    return 50 - 100 * numerator/(denominator + K.epsilon())

"""
def blockBuild(a, b, c):
    #First input model
    a = Dense(100)(a)
    a = AlphaDropout(0.2)(a)
    a = BatchNormalization()(a)
    a = Activation('softplus')(a)
    #Second input model
    b = Dense(100)(b)
    b = AlphaDropout(0.2)(b)
    b = BatchNormalization()(b)
    b = Activation('softplus')(b)
    #Third input model
    c = Dense(100)(c)
    c = AlphaDropout(0.2)(c)
    c = BatchNormalization()(c)
    c = Activation('softplus')(c)

    #Model body
    y = concatenate([a, b])
    y = Dense(num_situ)(y)
    y = BatchNormalization()(y)
    y = Activation('softplus')(y)
    x = concatenate([y, c])
    x = Dense(num_situ)(x)
    x = BatchNormalization()(x)
    x = Activation('softplus')(x)
    return(x)


a = Input(shape=(num_situ,))
b = Input(shape=(num_situ,))
c = Input(shape=(num_all,))
first_block = blockBuild(a,b,c)
second_block = blockBuild(concatenate([first_block, a]),b,c)
third_block = blockBuild(a, concatenate([second_block,b]), c)
forth_block = blockBuild(a,b,concatenate([c,third_block]))
output = Dense(1, activation='sigmoid')(forth_block)
model = Model(inputs=[a, b, c], outputs=[output])
"""

a1 = Input(shape=(2,num_situ,))
a2 = Conv1D (kernel_size = 2, filters = 8, activation='softplus')(a1)
a3 = Flatten()(a2)
a4 = Dense(2*num_situ)(a3)
a5 = AlphaDropout(0.2)(a4)
a6 = BatchNormalization()(a5)
a7 = Activation('softplus')(a6)
a8 = Dense(num_situ, activation='softplus')(a7)

#Third input model
c1 = Input(shape=(num_all,))
c2 = Dense(num_situ)(c1)
c3 = AlphaDropout(0.2)(c2)
c4 = BatchNormalization()(c3)
c5 = Activation('softplus')(c4)
c6 = concatenate([a8,c1]) #third_block
c7 = Dense(num_situ)(c6)
c8 = BatchNormalization()(c7)
c9 = Activation('softplus')(c8)
c10 = Dense(10)(c9)
c11 = BatchNormalization()(c10)
c12 = Activation('softplus')(c11)
output = Dense(1, activation='sigmoid')(c12)
model = Model(inputs=[a1, c1], outputs=[output])
model.compile(optimizer='adam', loss=matthews_correlation_loss, metrics=[matthews_correlation]) #'binary_crossentropy', binary_accuracy
print(model.summary())
#plot_model(model, to_file='my_model.png')

# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_matthews_correlation:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_matthews_correlation', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

print(time.strftime("%H:%M:%S"), ' Fit')
#tbCallBack = keras.callbacks.TensorBoard(log_dir='.', histogram_freq=0, write_graph=True, write_images=True)
history = model.fit(x=[X_train, X3_train],
                    y=y_train,
                    batch_size=50,
                    epochs=3000,
                    verbose=1,
                    validation_split=0.2,
                    callbacks=callbacks_list)
                    #class_weight={0:1, 1:10}) #, use_multiprocessing=True, workers=8) #, callbacks=[tbCallBack])
#model.save(f'data/model_sav_{num_situ}.h5')
sys.stdout.flush()

"""
#Use the model for prediction. Use batch prediction=50
print(time.ctime(), 'Using model')
c = pd.DataFrame()
#Loop on all cells in dge.csv and provide 10 highest probable locations
for index, row_d in d.iterrows():
    print(index, ' ', end="")
    sys.stdout.flush()
    #Loop over all possible 3039 locations in bdtnp.csv, and search for the line/s in BDTNP providing the higher probability
    #Add row_d to all rows in b
    row_expanded = pd.concat([row_d]*len(bdtnp), ignore_index=True, axis=1).T
    #b_expanded = pd.concat([b, row_expanded], axis=1)
    #pred = model.predict([b_expanded.iloc[glist_60_tom], b_expanded.iloc[:,num_situ:2*num_situ], b_expanded.iloc[:,2*num_situ:]], batch_size=50, verbose=0)
    #pred = model.predict([bdtnp[glist], row_expanded[glist], row_expanded[list(set(d.columns) - set(glist))]], batch_size=50, verbose=0)
    pred = model.predict([bdtnp, row_expanded[cols], row_expanded], batch_size=50, verbose=0) #[list(set(d.columns) - set(glist))]]
    c = pd.concat([c,pd.DataFrame([sorted(range(len(pred)), key=lambda i: pred[i])[-10:]], columns=['i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10'])])

c.to_csv(f'ann_{num_situ}.csv')
"""
print(time.ctime(), 'Done')