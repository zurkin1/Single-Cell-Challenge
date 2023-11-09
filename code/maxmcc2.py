import pandas as pd
from itertools import combinations as cs
import sys

list_84 = ['danr', 'CG14427', 'dan', 'CG43394', 'ImpL2', 'Nek2', 'CG8147', 'Ama', 'Btk29A', 'trn', 'numb', 'prd', 'brk', 'tsh', 'pxb', 'dpn', 'ftz', 'Kr', 'h', 'eve', 'Traf4', 'run', 'Blimp-1', 'lok', 'kni', 'tkv', 'MESR3', 'odd', 'noc', 'nub', 'Ilp4', 'aay', 'twi', 'bmm', 'hb', 'toc', 'rho', 'CG10479', 'gt', 'gk', 'apt', 'D', 'sna', 'NetA', 'Mdr49', 'fj', 'Mes2', 'CG11208', 'Doc2', 'bun', 'tll', 'Cyp310a1', 'Doc3', 'htl', 'Esp', 'bowl', 'oc', 'ImpE2', 'CG17724', 'fkh', 'edl', 'ems', 'zen2', 'CG17786', 'zen', 'disco', 'Dfd', 'mfas', 'knrl', 'Ance', 'croc', 'rau', 'cnc', 'exex', 'cad', 'Antp', 'erm', 'ken', 'peb', 'srp', 'E(spl)m5-HLH', 'CenG1A', 'zfh1', 'hkb']
bd = pd.read_csv('data/binarized_bdtnp.csv')[list_84]

inx = 0
previous_size = 0
previous_list = []
for tup in cs(bd.columns, 20):
    temp_bd = bd[list(tup)]
    temp_size = len(temp_bd.groupby(temp_bd.columns.tolist(), as_index=False).size())
    if(temp_size > previous_size):
        print(f'Found new set, size:{temp_size}')
        print(list(tup))
        previous_size = temp_size
        previous_list = list(tup)
        sys.stdout.flush()
    inx += 1
    if(inx % 10000 == 0):
       print(f'inx={inx}')
       print(f'Current tuple: {tup}')
       sys.stdout.flush()
