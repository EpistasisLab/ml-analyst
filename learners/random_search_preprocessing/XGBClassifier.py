import sys
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import ExtraTreesClassifier

from xgboost import XGBClassifier # Assumes XGBoost v0.6
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


pipeline_components.append(XGBClassifier )


n_estimators_values = np.random.choice(list(range(50, 1001, 50)), size=num_param_combinations)
learning_rate_values = np.random.uniform(low=1e-10, high=5., size=num_param_combinations)
gamma_values = np.random.uniform(low=0., high=1., size=num_param_combinations)
max_depth_values = np.random.choice(list(range(1, 21)), size=num_param_combinations)
subsample_values = np.random.uniform(low=0., high=1., size=num_param_combinations)

all_param_combinations = zip(n_estimators_values, learning_rate_values, gamma_values, max_depth_values, subsample_values)
pipeline_parameters[XGBClassifier] = \
   [{'n_estimators': n_estimators, 'learning_rate': learning_rate, 'gamma': gamma, 'max_depth': max_depth, 'subsample': subsample, 'seed': 324089, 'nthread': 1}
     for (n_estimators, learning_rate, gamma, max_depth, subsample) in all_param_combinations]


evaluate_model(dataset, save_file, random_seed, pipeline_components, pipeline_parameters)
