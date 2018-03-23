import sys
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import RandomForestClassifier
from evaluate_model import evaluate_model
from preprocessors import preprocessor_dict

# inputs
dataset = sys.argv[1]
save_file = sys.argv[2]
num_param_combinations = int(sys.argv[3])
random_seed = int(sys.argv[4])
preps = sys.argv[5]


np.random.seed(random_seed)

# construct pipeline
pipeline_components = []
pipeline_parameters = {}

for p in preps.split(','):
    pipeline_components.append((p, preprocessor_dict[p]))
    # if p is 'SelectFromModel':
    #     pipeline_parameters[p] = [{'estimator': }]
    # elif p is 'RFE':
    #     pipeline_parameters[p] = [{'estimator': ExtraTreesClassifier(n_estimators=100, random_state=324089)}]

pipeline_components.append(('RandomForestClassifier',RandomForestClassifier()))

# parameters for method
n_estimators= list(range(50, 1001, 50))
min_impurity_decrease= np.random.exponential(scale=0.01, size=num_param_combinations)
max_features= list(np.arange(0.01, 1., 0.01)) + ['sqrt', 'log2', None]
criterion= ['gini', 'entropy']
max_depth= list(range(1, 21)) + [None]

pipeline_parameters['RandomForestClassifier'] = \
        {'n_estimators': n_estimators, 'min_impurity_decrease': min_impurity_decrease, 'max_features': max_features, 'criterion': criterion, 'max_depth': max_depth, 'random_state': [324089], 'class_weight':['balanced']}

#evaluate
evaluate_model(dataset, save_file, random_seed, pipeline_components, pipeline_parameters, num_param_combinations)
