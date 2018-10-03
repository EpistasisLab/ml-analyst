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


pipeline_components.append(('SVC', SVC(random_state=random_seed)))


C_values = np.random.uniform(low=1e-10, high=500., size=num_param_combinations)
gamma_values = np.random.choice(list(np.arange(0.05, 1.01, 0.05)) + ['auto'], size=num_param_combinations)
kernel_values = ['poly', 'rbf', 'sigmoid']
degree_values = [2, 3]
coef0_values = np.random.uniform(low=0., high=10., size=num_param_combinations)

pipeline_parameters['SVC'] = \
   {'C': C_values, 'gamma': [float(gamma) if gamma != 'auto' else gamma for gamma in gamma_values], 'kernel': kernel_values, 
    'degree': degree_values, 'coef0': coef0_values}


evaluate_model(dataset, save_file, random_seed, pipeline_components, pipeline_parameters, num_param_combinations, label)
