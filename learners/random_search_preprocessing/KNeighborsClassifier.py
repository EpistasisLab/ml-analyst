import sys
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.neighbors import KNeighborsClassifier
from evaluate_model import evaluate_model
from preprocessors import preprocessor_dict

dataset = sys.argv[1]
save_file = sys.argv[2]
num_param_combinations = int(sys.argv[3])
random_seed = int(sys.argv[4])
preps = sys.argv[5]

np.random.seed(random_seed)

# construct pipeline
pipeline_components=[]
pipeline_parameters={}
for p in preps.split(','):
    pipeline_components.append((p, preprocessor_dict[p]))
    # if pipeline_components[-1] is SelectFromModel:
    #     pipeline_parameters[SelectFromModel] = [{'estimator': ExtraTreesClassifier(n_estimators=100, random_state=324089)}]
    # elif pipeline_components[-1] is RFE:
    #     pipeline_parameters[RFE] = [{'estimator': ExtraTreesClassifier(n_estimators=100, random_state=324089)}]


pipeline_components.append('KNeighborsClassifier',KNeighborsClassifier())


n_neighbors_values = np.random.randint(low=1, high=100, size=num_param_combinations)
weights_values = np.random.choice(['uniform', 'distance'], size=num_param_combinations)

pipeline_parameters['KNeighborsClassifier'] = \
   {'n_neighbors': n_neighbors_values, 'weights': weights_values}

evaluate_model(dataset, save_file, random_seed, pipeline_components, pipeline_parameters, num_param_combinations)
