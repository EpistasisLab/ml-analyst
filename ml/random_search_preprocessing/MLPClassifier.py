import sys
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neural_network import MLPClassifier
from evaluate_model import evaluate_model
from preprocessors import preprocessor_dict

# inputs
dataset = sys.argv[1]
save_file = sys.argv[2]
num_param_combinations = int(sys.argv[3])
random_seed = int(sys.argv[4])
preps = sys.argv[5]
label = sys.argv[6]


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

pipeline_components.append(('MLPClassifier',MLPClassifier()))

# parameters for method
hidden_layer_sizes = [(n_layers,n_nodes) for n_layers in np.arange(1,10) for n_nodes in np.arange(10,100,10)]
print(hidden_layer_sizes)
activation = ['identity','logistic','tanh','relu']
solver = ['lbfgs', 'sgd', 'adam']

pipeline_parameters['MLPClassifier'] = \
        {'hidden_layer_sizes':hidden_layer_sizes, 'activation':activation, 'solver':solver}

#evaluate
evaluate_model(dataset, save_file, random_seed, pipeline_components, pipeline_parameters, num_param_combinations, label)
