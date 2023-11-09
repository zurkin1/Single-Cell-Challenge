import tensorflow as tf
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

if not os.path.isfile('./data/binarized_bdtnp.csv'):
   print('Reading files from AWS')
   session = boto3.session.Session(
        aws_access_key_id="AKIAJHH3NP6IDC7SRYQQ",
        aws_secret_access_key="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"
   )
   s3 = session.resource("s3")
   bucket = s3.Bucket('daniglassbox')
   obj = bucket.Object('ann.zip')

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
num_situ = 20
num_all = 8924
glist_20 = ['danr','CG14427','dan','CG43394','ImpL2','Nek2','CG8147','Ama','Btk29A','trn','numb','prd','brk','tsh','pxb','dpn','ftz','Kr','h','eve']
glist_84 = ['aay','Ama','Ance','Antp','apt','Blimp-1','bmm','bowl','brk','Btk29A','bun','cad','CenG1A','CG10479','CG11208','CG14427','CG17724','CG17786','CG43394','CG8147','cnc','croc','Cyp310a1','D','dan','danr','Dfd','disco','Doc2','Doc3','dpn','edl','ems','erm','Esp','E(spl)m5-HLH','eve','exex','fj','fkh','ftz','gk','gt','h','hb','hkb','htl','Ilp4','ImpE2','ImpL2','ken','kni','knrl','Kr','lok','Mdr49','Mes2','MESR3','mfas','Nek2','NetA','noc','nub','numb','oc','odd','peb','prd','pxb','rau','rho','run','sna','srp','tkv','tll','toc','Traf4','trn','tsh','twi','zen','zen2','zfh1']

bdtnp_bin = pd.read_csv('data/binarized_bdtnp.csv')[glist_20]
d1_bin = pd.read_csv('data/dge_binarized_distMap_T.csv')
d2_bin = pd.read_csv('data/magic_dge_bin.csv')
labels = pd.read_csv('data/labels.csv')

print(time.ctime(), 'Create train input array for dge to bdtnp model')
len_ = len(labels)
Z_ = np.empty((len_, 84))
W_ = np.empty((len_, num_all))
y_ = np.empty((len_, num_situ))

for index, row in labels.iterrows():
    if (index % 100 == 0):
        print(index, ' ', end="")
    Z_[index] = d1_bin.iloc[index]
    W_[index] = d2_bin.iloc[index]
    y_[index] = bdtnp_bin.iloc[int(row[0])]

#Model for dge to bdtnp.
print(time.ctime(), 'Model build')

a1 = Input(shape=(84,))
a2 = Dense(84)(a1)
a3 = BatchNormalization()(a2)
a4 = AlphaDropout(0.5)(a3)

c1 = Input(shape=(num_all,))
c3 = Dense(84)(c1)
c4 = BatchNormalization()(c3)
c5 = AlphaDropout(0.5)(c4)

e = concatenate([a4, c5])
e = Dense(84)(e)
e = BatchNormalization()(e)
e = Activation('softplus')(e)
e = AlphaDropout(0.5)(e)
output = Dense(num_situ, activation='sigmoid')(e)
model = Model(inputs=[a1, c1], outputs=[output])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
print(model.summary())

print(time.strftime("%H:%M:%S"), ' Fit')
filepath="models/weights-improvement-{epoch:02d}-{val_binary_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_binary_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(  x=[Z_, W_], y=y_,
            batch_size=10,
            epochs=100000,
            verbose=2,
            validation_split=0.2,
            callbacks=callbacks_list)

sys.stdout.flush()
print(time.ctime(), 'Done')