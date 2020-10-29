#Merge files from Gene2Function (second method from non in-situ genes)
import os
import pandas as pd
import gc

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

data = pd.DataFrame()
for filename in os.listdir('c:\\data\\Dream\\macro-1\\'):
    print(filename)
    if('goslim' in filename):
        filenames = filename.split('_') #go_heatmap__goslim_31028.txt
        filename2 = 'go_heatmap__go_' + filenames[4]
    else:
        continue

    file1 = pd.read_csv(f'c:\\data\\Dream\macro-1\\{filename}', delimiter='\t')
    file2 = pd.read_csv(f'c:\\data\Dream\\macro-1\\{filename2}', delimiter='\t')
    file1 = file1.loc[file1.Species == 'Fly']
    file2 = file2.loc[file2.Species == 'Fly']
    file = pd.concat((file1,file2), axis=1)
    file = file.loc[:, ~file.columns.duplicated()]
    data = pd.concat([data, file], axis=0, sort=False)

data.fillna('0', inplace=True)
data.to_csv(f'g2f_other.csv')