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
glist_60 = ['danr','CG14427','dan','CG43394','ImpL2','Nek2','CG8147','Ama','Btk29A','trn','numb','prd','brk','tsh','pxb','dpn','ftz','Kr','h','eve','Traf4','run','Blimp-1','lok','kni','tkv','MESR3','odd','noc','nub','Ilp4','aay','twi','bmm','hb','toc','rho','CG10479','gt','gk','apt','D','sna','NetA','Mdr49','fj','Mes2','CG11208','Doc2','bun','tll','Cyp310a1','Doc3','htl','Esp','bowl','oc','ImpE2','CG17724','fkh']
glist_40 = ['danr','CG14427','dan','CG43394','ImpL2','Nek2','CG8147','Ama','Btk29A','trn','numb','prd','brk','tsh','pxb','dpn','ftz','Kr','h','eve','Traf4','run','Blimp-1','lok','kni','tkv','MESR3','odd','noc','nub','Ilp4','aay','twi','bmm','hb','toc','rho','CG10479','gt','gk']
glist_20_mod = ['danr', 'CG14427', 'dan', 'CG43394', 'ImpL2', 'Nek2', 'CG8147', 'Ama', 'Btk29A', 'trn', 'numb', 'prd', 'brk', 'tsh', 'pxb', 'dpn', 'h', 'Traf4', 'run', 'toc']

bdtnp_bin = pd.read_csv('data/binarized_bdtnp.csv')[glist_20_mod]
d1_bin = pd.read_csv('data/dge_binarized_distMap_T.csv')
labels = pd.read_csv('data/labels.csv')

print(time.ctime(), 'Create train input array for dge to bdtnp model')
len_ = len(labels)
Z_ = np.empty((len_, 84))
y_ = np.empty((len_, num_situ))

for index, row in labels.iterrows():
    if (index % 100 == 0):
        print(index, ' ', end="")
    Z_[index] = d1_bin.iloc[index]
    y_[index] = bdtnp_bin.iloc[int(row[0])]

#Model for dge to bdtnp.
print(time.ctime(), 'Model build')

a1 = Input(shape=(84,))
a = Dense(84)(a1)
a = BatchNormalization()(a)
a = Dropout(0.3)(a)

e = Dense(40)(a)
e = BatchNormalization()(e)
e = Activation('softplus')(e)
e = Dropout(0.2)(e)

output = Dense(num_situ, activation='sigmoid')(e)
model = Model(inputs=[a1], outputs=[output])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
print(model.summary())

print(time.strftime("%H:%M:%S"), ' Fit')
filepath="models20/weights-improvement-{epoch:02d}-{val_binary_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_binary_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(  x=[Z_], y=y_,
            batch_size=10,
            epochs=200000,
            verbose=2,
            validation_split=0.2,
            callbacks=callbacks_list)

sys.stdout.flush()
print(time.ctime(), 'Done')
