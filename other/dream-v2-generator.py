"""
Todo
- Add validation pairs.
- Convergence.
- Run on Linux.
- Compare runs of 80 and 60 in situ genes.
- Add dropouts.
- Select the best 60 genes.
"""
import numpy as np
import pandas as pd
import pydot
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Embedding, concatenate, Flatten, Dropout, Lambda, Activation, BatchNormalization, LocallyConnected1D, Reshape
from keras.utils import plot_model
from keras.callbacks import CSVLogger
import time
import itertools
import random
import logging

logging.basicConfig(filename='keras-logs.csv', level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
logging.debug('Start processing files')
b = pd.read_csv('bdtnp.csv')
#Changes 'na' to 'naa' and 'nan' to 'nana'
d = pd.read_csv('dge_raw.csv', index_col=0, header=None, encoding='ISO-8859-1').T
labels = pd.read_csv('labels.csv', index_col=0, header=None).T
#Move in-situ 84 genes to the begining
cols = list(b) + list(set(list(d)) - set(list(b)))
d = d.loc[:,cols]
d = d.div(d.sum(axis=1), axis=0)
d.reset_index(drop=True, inplace=True)
labels.reset_index(drop=True, inplace=True)

#Create the true label pairs. Left: d-array, right: b-array
#Labels start from 1 in the original file. They indicate a specific row in b table.
d['label'] = labels['label'] - 1
d['one'] = 1
d_true = list(zip(d.index, d.label, d.one))
d = d.drop(['one', 'label'], 1)

#Create the false label pairs
a_list = [i for i in range(0,1297)] # d-array
b_list = [j for j in range(0,3039)] # b-array
d_prod = list(itertools.product(a_list, b_list))
d_false = [x+(0,) for x in d_prod if x not in d_true]

#Merge the two lists. Select 10003 samples from d and 1297 from b for training.
indicies = random.sample(range(len(d_false)), 10003)
d_false1 = [d_false[i] for i in indicies]
d_list = d_true + d_false1
random.shuffle(d_list)
logging.debug(f'Length of d_list of pairs: {len(d_list)}') #11300

# Output of generator should be a tuple `(x, y, sample_weight)` or `(x, y)`.
class generate_array(keras.utils.Sequence):
    def __init__(self, pairs_set, batch_s):
        self.pairs = pairs_set
        self.batch_size = batch_s

    # Return the maximume number of batches
    # Dataset size is 11300. For a batch_size=100 we need 113 iterations of 100.
    def __len__(self):
        return int(np.ceil(len(self.pairs) / float(self.batch_size)))

    # Returns the batch #idx
    def __getitem__(self, idx):
        X1 = np.empty((self.batch_size, 84))
        X2 = np.empty((self.batch_size, 84))
        X3 = np.empty((self.batch_size, 8840))
        y = np.empty((self.batch_size), dtype=int)
        batch = 0
        for i in self.pairs[self.batch_size * idx: self.batch_size * (idx + 1)]:
            try:
                b_row = b.iloc[i[1]]
                d_row_1 = d.iloc[i[0]][0:84]
                d_row_2 = d.iloc[i[0]][84:]
            except:
                logging.warning(f'Exception, generate_array......{i}')
                continue
            X1[batch] = b_row
            X2[batch] = d_row_1
            X3[batch] = d_row_2
            y[batch] = i[2]
            batch = batch + 1
        return ([X1, X2, X3], y)

#myg = generate_array(d_list, 100)
#print(myg.__getitem__(0)[1])

#Model build
logging.debug('Model build')
#First input model
input_a = Input(shape=(84,))
dense_a = Dense(200, activation='softplus')(input_a)
#Second input model
input_b = Input(shape=(84,))
dense_b = Dense(200, activation='softplus')(input_b)
#Third input model
input_c = Input(shape=(8840,))
dense_c = Dense(50, activation='softplus')(input_c)
drop_c = Dropout(0.2)(dense_c)
concat_a = concatenate([dense_a, dense_b])
dense_d = Dense(100, activation='softplus')(concat_a)
drop_a = Dropout(0.2)(dense_d)
concat_b = concatenate([drop_a, drop_c])
dense_e = Dense(50, activation='softplus')(concat_b)
dense_f = Dense(1, activation='softplus')(dense_e)
model = Model(inputs=[input_a, input_b, input_c], outputs=[dense_f])
model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])
with open('keras-logs.csv', 'a') as fh:
    model.summary(print_fn=lambda x: fh.write(x + '\n'))
plot_model(model, to_file='my_model.png')

logging.debug('Fit')
tbCallBack = keras.callbacks.TensorBoard(log_dir='.', histogram_freq=0, write_graph=True, write_images=True)
csv_logger = CSVLogger('keras-logs.csv', append=True, separator=',')
history = model.fit_generator(generate_array(d_list[0:10000], 50),
                              steps_per_epoch=200,
                              epochs=20,
                              verbose=2,
                              validation_data=generate_array(d_list[10000:13000],50),
                              class_weight={0:1, 1:10},
                              callbacks=[csv_logger]) #, use_multiprocessing=True, workers=8) #, callbacks=[tbCallBack])

# serialize model to JSON
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
#    json_file.write(model_json)
#model.save_weights('model_w.h5')
model.save('model_sav.h5')