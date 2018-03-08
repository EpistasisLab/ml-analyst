import sys
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.svm import SVC
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
    pipeline_components.append(preprocessor_dict[p])
    if pipeline_components[-1] is SelectFromModel:
        pipeline_parameters[SelectFromModel] = [{'estimator': ExtraTreesClassifier(n_estimators=100, random_state=324089)}]
    elif pipeline_components[-1] is RFE:
        pipeline_parameters[RFE] = [{'estimator': ExtraTreesClassifier(n_estimators=100, random_state=324089)}]


pipeline_components.append(SVC )


C_values = np.random.uniform(low=1e-10, high=500., size=num_param_combinations)
gamma_values = np.random.choice(list(np.arange(0.05, 1.01, 0.05)) + ['auto'], size=num_param_combinations)
kernel_values = np.random.choice(['poly', 'rbf', 'sigmoid'], size=num_param_combinations)
degree_values = np.random.choice([2, 3], size=num_param_combinations)
coef0_values = np.random.uniform(low=0., high=10., size=num_param_combinations)

all_param_combinations = zip(C_values, gamma_values, kernel_values, degree_values, coef0_values)
pipeline_parameters[SVC] = \
   [{'C': C, 'gamma': float(gamma) if gamma != 'auto' else gamma, 'kernel': str(kernel), 'degree': 2 if kernel != 'poly' else degree, 'coef0': 0. if kernel not in ['poly', 'sigmoid'] else coef0, 'random_state': 324089}
     for (C, gamma, kernel, degree, coef0) in all_param_combinations]


evaluate_model(dataset, save_file, random_seed, pipeline_components, pipeline_parameters)
