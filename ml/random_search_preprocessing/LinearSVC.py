import sys
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import LinearSVC
from evaluate_model import evaluate_model
from preprocessors import preprocessor_dict

dataset = sys.argv[1]
save_file = sys.argv[2]
num_param_combinations = int(sys.argv[3])
random_seed = int(sys.argv[4])
preps = sys.argv[5]
label = sys.argv[6]

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


pipeline_components.append('LinearSVC', LinearSVC ())


C_values = np.random.uniform(low=1e-10, high=10., size=num_param_combinations)
loss_values = np.random.choice(['hinge', 'squared_hinge'], size=num_param_combinations)
penalty_values = np.random.choice(['l1', 'l2'], size=num_param_combinations)
dual_values = np.random.choice([True, False], size=num_param_combinations)
fit_intercept_values = np.random.choice([True, False], size=num_param_combinations)

pipeline_parameters['LinearSVC'] = \
   {'C': C_values, 'penalty': penalty_values, 'fit_intercept': fit_intercept_values, 'dual': dual_values, 'random_state': 324089}
     


evaluate_model(dataset, save_file, random_seed, pipeline_components, pipeline_parameters, num_param_combinations, label)
