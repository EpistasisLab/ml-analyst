import pandas as pd
import numpy as np

def read_file(filename, sep=None):
    
    if filename.split('.')[-1] == 'gz':
        compression = 'gzip'
    else:
        compression = None

    if sep:
        input_data = pd.read_csv(filename, sep=sep, compression=compression)
    else:
        input_data = pd.read_csv(filename, sep=sep, compression=compression,
                engine='python')
    
    input_data.rename(columns={'Label': 'class','Class':'class', 'target':'class'}, 
                      inplace=True)

    feature_names = np.array([x for x in input_data.columns.values if x != 'class'])

    X = input_data.drop('class', axis=1).values.astype(float)
    y = input_data['class'].values

    return X, y, feature_names
