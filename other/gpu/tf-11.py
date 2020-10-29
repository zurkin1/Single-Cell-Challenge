import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import combinations as cs
import sys
import time

glist = ['danr', 'CG14427', 'dan', 'CG43394', 'ImpL2', 'Nek2', 'CG8147', 'Ama', 'Btk29A', 'trn', 'numb', 'prd', 'brk',
         'tsh', 'pxb', 'dpn', 'ftz', 'Kr', 'h', 'eve', 'Traf4', 'run', 'Blimp-1', 'lok', 'kni', 'tkv', 'MESR3', 'odd',
         'noc', 'nub', 'Ilp4', 'aay', 'twi', 'bmm', 'hb', 'toc', 'rho', 'CG10479', 'gt', 'gk', 'apt', 'D', 'sna',
         'NetA', 'Mdr49', 'fj', 'Mes2', 'CG11208', 'Doc2', 'bun', 'tll', 'Cyp310a1', 'Doc3', 'htl', 'Esp', 'bowl', 'oc',
         'ImpE2', 'CG17724', 'fkh', 'edl', 'ems', 'zen2', 'CG17786', 'zen', 'disco', 'Dfd', 'mfas', 'knrl', 'Ance',
         'croc', 'rau', 'cnc', 'exex', 'cad', 'Antp', 'erm', 'ken', 'peb', 'srp', 'E(spl)m5-HLH', 'CenG1A', 'zfh1',
         'hkb']
bdtnp_bin = pd.read_csv('binarized_bdtnp.csv')
bdtnp = bdtnp_bin.values.astype(np.float32)


def generator():
    for tup in cs(glist, 20):  # sequence:
        yield bdtnp_bin[np.reshape(tup, -1)].values.astype(np.float32)


dataset = tf.data.Dataset().from_generator(generator,
                                           output_types=tf.float32,
                                           output_shapes=(tf.TensorShape([3039, 20]))).batch(10).prefetch(
    4000000)  # 4000000
dataset = dataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))  # tf.data.experimental.prefetch_to_device
iter = dataset.make_initializable_iterator()

with tf.device('/GPU:0'):
    Y = tf.reshape(tf.tile(tf.constant([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288], tf.float32),
                           tf.constant([10])),
                   (10, 20, 1))
    X = iter.get_next()
    # Code binary sequences as integers in range [0,2**20].
    product = tf.matmul(X, Y)  # product.shape == (10, 3039, 1)
    # Move to vector
    un_2 = tf.to_int32(tf.squeeze(product))  # un_2.shape == (10, 3039)

    un_3_00, _ = tf.unique(un_2[0, :])
    un_3_01, _ = tf.unique(un_2[1, :])
    un_3_02, _ = tf.unique(un_2[2, :])
    un_3_03, _ = tf.unique(un_2[3, :])
    un_3_04, _ = tf.unique(un_2[4, :])
    un_3_05, _ = tf.unique(un_2[5, :])
    un_3_06, _ = tf.unique(un_2[6, :])
    un_3_07, _ = tf.unique(un_2[7, :])
    un_3_08, _ = tf.unique(un_2[8, :])
    un_3_09, _ = tf.unique(un_2[9, :])

    all_max_0 = tf.maximum(tf.size(un_3_00), tf.size(un_3_01))

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess: # , allow_soft_placement=True
    sess.run(iter.initializer)
    for i in range (100):
       sess.run(all_max_0)
       if(i % 10 == 0):
          print(time.ctime())

    print(time.ctime())  # f'Best result: {result}')
