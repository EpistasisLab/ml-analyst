import pandas as pd
import numpy as np
import pdb

def read_file(filename, label='class', sep=None):
    
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

    feature_names = np.array([x for x in input_data.columns.values if x != label])

    X = input_data.drop(label, axis=1).values.astype(float)
    y = input_data[label].values

    assert(X.shape[1] == feature_names.shape[0])

    return X, y, feature_names
