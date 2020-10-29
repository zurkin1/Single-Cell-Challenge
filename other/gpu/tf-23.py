import numpy as np
import pandas as pd
import tensorflow as tf
from itertools import combinations as cs
import sys

num_situ = 20
batch_size = 100  # 100
two_list = [[[2 ** x] for x in range(0, num_situ)] for j in range(batch_size)]
glist = ['danr', 'CG14427', 'dan', 'CG43394', 'ImpL2', 'Nek2', 'CG8147', 'Ama', 'Btk29A', 'trn', 'numb', 'prd', 'brk',
         'tsh', 'pxb', 'dpn', 'ftz', 'Kr', 'h', 'eve', 'Traf4', 'run', 'Blimp-1', 'lok', 'kni', 'tkv', 'MESR3', 'odd',
         'noc', 'nub', 'Ilp4', 'aay', 'twi', 'bmm', 'hb', 'toc', 'rho', 'CG10479', 'gt', 'gk', 'apt', 'D', 'sna',
         'NetA', 'Mdr49', 'fj', 'Mes2', 'CG11208', 'Doc2', 'bun', 'tll', 'Cyp310a1', 'Doc3', 'htl', 'Esp', 'bowl', 'oc',
         'ImpE2', 'CG17724', 'fkh', 'edl', 'ems', 'zen2', 'CG17786', 'zen', 'disco', 'Dfd', 'mfas', 'knrl', 'Ance',
         'croc', 'rau', 'cnc', 'exex', 'cad', 'Antp', 'erm', 'ken', 'peb', 'srp', 'E(spl)m5-HLH', 'CenG1A', 'zfh1',
         'hkb']
bdtnp_bin = pd.read_csv('../../data/binarized_bdtnp.csv')
bdtnp = bdtnp_bin.values.astype(np.int32)


def generator():
    for tup in cs(glist, num_situ):  # sequence:
        yield bdtnp_bin[np.reshape(tup, -1)].values.astype(np.int32)


dataset = tf.data.Dataset().from_generator(generator,
                                           output_types=tf.int32,
                                           output_shapes=(tf.TensorShape([3039, num_situ]))).batch(batch_size).prefetch(4000000)
dataset = dataset.apply(tf.contrib.data.prefetch_to_device('/gpu:0'))  # tf.data.experimental.prefetch_to_device
iter = dataset.make_initializable_iterator()

with tf.device('/GPU:0'):
    Y = tf.placeholder(tf.int32, shape=(batch_size, num_situ, 1))
    a = tf.Variable(tf.constant(0))
    init = tf.global_variables_initializer()
    # Track both the loop index and summation in a tuple in the form (index, summation)
    index_max = (tf.constant(1), a)


    # The loop condition
    def condition(index, all_max):
        return tf.less(index, 5)

    # The loop body, this will return a result tuple in the same form (index, summation)
    def body(index, all_max):
        X = iter.get_next()
        # Code binary sequences as integers in range [0,2**20].
        product = tf.matmul(X, Y)  # product.shape == (10, 3039, 1)
        # Move to vector
        un_0 = tf.squeeze(product)  # un_0.shape == (10, 3039)
        un_0_00, _ = tf.unique(un_0[0, :])
        un_0_01, _ = tf.unique(un_0[1, :])
        un_0_02, _ = tf.unique(un_0[2, :])
        un_0_03, _ = tf.unique(un_0[3, :])
        un_0_04, _ = tf.unique(un_0[4, :])
        un_0_05, _ = tf.unique(un_0[5, :])
        un_0_06, _ = tf.unique(un_0[6, :])
        un_0_07, _ = tf.unique(un_0[7, :])
        un_0_08, _ = tf.unique(un_0[8, :])
        un_0_09, _ = tf.unique(un_0[9, :])
        un_0_10, _ = tf.unique(un_0[10, :])
        un_0_11, _ = tf.unique(un_0[11, :])
        un_0_12, _ = tf.unique(un_0[12, :])
        un_0_13, _ = tf.unique(un_0[13, :])
        un_0_14, _ = tf.unique(un_0[14, :])
        un_0_15, _ = tf.unique(un_0[15, :])
        un_0_16, _ = tf.unique(un_0[16, :])
        un_0_17, _ = tf.unique(un_0[17, :])
        un_0_18, _ = tf.unique(un_0[18, :])
        un_0_19, _ = tf.unique(un_0[19, :])
        un_0_20, _ = tf.unique(un_0[20, :])
        un_0_21, _ = tf.unique(un_0[21, :])
        un_0_22, _ = tf.unique(un_0[22, :])
        un_0_23, _ = tf.unique(un_0[23, :])
        un_0_24, _ = tf.unique(un_0[24, :])
        un_0_25, _ = tf.unique(un_0[25, :])
        un_0_26, _ = tf.unique(un_0[26, :])
        un_0_27, _ = tf.unique(un_0[27, :])
        un_0_28, _ = tf.unique(un_0[28, :])
        un_0_29, _ = tf.unique(un_0[29, :])
        un_0_30, _ = tf.unique(un_0[30, :])
        un_0_31, _ = tf.unique(un_0[31, :])
        un_0_32, _ = tf.unique(un_0[32, :])
        un_0_33, _ = tf.unique(un_0[33, :])
        un_0_34, _ = tf.unique(un_0[34, :])
        un_0_35, _ = tf.unique(un_0[35, :])
        un_0_36, _ = tf.unique(un_0[36, :])
        un_0_37, _ = tf.unique(un_0[37, :])
        un_0_38, _ = tf.unique(un_0[38, :])
        un_0_39, _ = tf.unique(un_0[39, :])
        un_0_40, _ = tf.unique(un_0[40, :])
        un_0_41, _ = tf.unique(un_0[41, :])
        un_0_42, _ = tf.unique(un_0[42, :])
        un_0_43, _ = tf.unique(un_0[43, :])
        un_0_44, _ = tf.unique(un_0[44, :])
        un_0_45, _ = tf.unique(un_0[45, :])
        un_0_46, _ = tf.unique(un_0[46, :])
        un_0_47, _ = tf.unique(un_0[47, :])
        un_0_48, _ = tf.unique(un_0[48, :])
        un_0_49, _ = tf.unique(un_0[49, :])
        un_0_50, _ = tf.unique(un_0[50, :])
        un_0_51, _ = tf.unique(un_0[51, :])
        un_0_52, _ = tf.unique(un_0[52, :])
        un_0_53, _ = tf.unique(un_0[53, :])
        un_0_54, _ = tf.unique(un_0[54, :])
        un_0_55, _ = tf.unique(un_0[55, :])
        un_0_56, _ = tf.unique(un_0[56, :])
        un_0_57, _ = tf.unique(un_0[57, :])
        un_0_58, _ = tf.unique(un_0[58, :])
        un_0_59, _ = tf.unique(un_0[59, :])
        un_0_60, _ = tf.unique(un_0[60, :])
        un_0_61, _ = tf.unique(un_0[61, :])
        un_0_62, _ = tf.unique(un_0[62, :])
        un_0_63, _ = tf.unique(un_0[63, :])
        un_0_64, _ = tf.unique(un_0[64, :])
        un_0_65, _ = tf.unique(un_0[65, :])
        un_0_66, _ = tf.unique(un_0[66, :])
        un_0_67, _ = tf.unique(un_0[67, :])
        un_0_68, _ = tf.unique(un_0[68, :])
        un_0_69, _ = tf.unique(un_0[69, :])
        un_0_70, _ = tf.unique(un_0[70, :])
        un_0_71, _ = tf.unique(un_0[71, :])
        un_0_72, _ = tf.unique(un_0[72, :])
        un_0_73, _ = tf.unique(un_0[73, :])
        un_0_74, _ = tf.unique(un_0[74, :])
        un_0_75, _ = tf.unique(un_0[75, :])
        un_0_76, _ = tf.unique(un_0[76, :])
        un_0_77, _ = tf.unique(un_0[77, :])
        un_0_78, _ = tf.unique(un_0[78, :])
        un_0_79, _ = tf.unique(un_0[79, :])
        un_0_80, _ = tf.unique(un_0[80, :])
        un_0_81, _ = tf.unique(un_0[81, :])
        un_0_82, _ = tf.unique(un_0[82, :])
        un_0_83, _ = tf.unique(un_0[83, :])
        un_0_84, _ = tf.unique(un_0[84, :])
        un_0_85, _ = tf.unique(un_0[85, :])
        un_0_86, _ = tf.unique(un_0[86, :])
        un_0_87, _ = tf.unique(un_0[87, :])
        un_0_88, _ = tf.unique(un_0[88, :])
        un_0_89, _ = tf.unique(un_0[89, :])
        un_0_90, _ = tf.unique(un_0[90, :])
        un_0_91, _ = tf.unique(un_0[91, :])
        un_0_92, _ = tf.unique(un_0[92, :])
        un_0_93, _ = tf.unique(un_0[93, :])
        un_0_94, _ = tf.unique(un_0[94, :])
        un_0_95, _ = tf.unique(un_0[95, :])
        un_0_96, _ = tf.unique(un_0[96, :])
        un_0_97, _ = tf.unique(un_0[97, :])
        un_0_98, _ = tf.unique(un_0[98, :])
        un_0_99, _ = tf.unique(un_0[99, :])
        all_max_0 = tf.reduce_max(tf.stack([
        tf.size(un_0_00), tf.size(un_0_01), tf.size(un_0_02), tf.size(un_0_03), tf.size(un_0_04), tf.size(un_0_05), tf.size(un_0_06), tf.size(un_0_07), tf.size(un_0_08), tf.size(un_0_09),\
        tf.size(un_0_10), tf.size(un_0_11), tf.size(un_0_12), tf.size(un_0_13), tf.size(un_0_14), tf.size(un_0_15), tf.size(un_0_16), tf.size(un_0_17), tf.size(un_0_18), tf.size(un_0_19),\
        tf.size(un_0_20), tf.size(un_0_21), tf.size(un_0_22), tf.size(un_0_23), tf.size(un_0_24), tf.size(un_0_25), tf.size(un_0_26), tf.size(un_0_27), tf.size(un_0_28), tf.size(un_0_29),\
        tf.size(un_0_30), tf.size(un_0_31), tf.size(un_0_32), tf.size(un_0_33), tf.size(un_0_34), tf.size(un_0_35), tf.size(un_0_36), tf.size(un_0_37), tf.size(un_0_38), tf.size(un_0_39),\
        tf.size(un_0_40), tf.size(un_0_41), tf.size(un_0_42), tf.size(un_0_43), tf.size(un_0_44), tf.size(un_0_45), tf.size(un_0_46), tf.size(un_0_47), tf.size(un_0_48), tf.size(un_0_49),\
        tf.size(un_0_50), tf.size(un_0_51), tf.size(un_0_52), tf.size(un_0_53), tf.size(un_0_54), tf.size(un_0_55), tf.size(un_0_56), tf.size(un_0_57), tf.size(un_0_58), tf.size(un_0_59),\
        tf.size(un_0_60), tf.size(un_0_61), tf.size(un_0_62), tf.size(un_0_63), tf.size(un_0_64), tf.size(un_0_65), tf.size(un_0_66), tf.size(un_0_67), tf.size(un_0_68), tf.size(un_0_69),\
        tf.size(un_0_70), tf.size(un_0_71), tf.size(un_0_72), tf.size(un_0_73), tf.size(un_0_74), tf.size(un_0_75), tf.size(un_0_76), tf.size(un_0_77), tf.size(un_0_78), tf.size(un_0_79),\
        tf.size(un_0_80), tf.size(un_0_81), tf.size(un_0_82), tf.size(un_0_83), tf.size(un_0_84), tf.size(un_0_85), tf.size(un_0_86), tf.size(un_0_87), tf.size(un_0_88), tf.size(un_0_89),\
        tf.size(un_0_90), tf.size(un_0_91), tf.size(un_0_92), tf.size(un_0_93), tf.size(un_0_94), tf.size(un_0_95), tf.size(un_0_96), tf.size(un_0_97), tf.size(un_0_98), tf.size(un_0_99)]))
        return tf.add(index, 1), all_max_0

    wh_loop = tf.while_loop(condition, body, index_max, parallel_iterations = 50)[1]

with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)) as sess:
    sess.run(init)
    result = sess.run(wh_loop)  # , feed_dict={Y: two_list})

print(f'Best result: {result}')