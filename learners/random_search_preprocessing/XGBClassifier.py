import sys
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import ExtraTreesClassifier

from xgboost import XGBClassifier # Assumes XGBoost v0.6
from evaluate_model import evaluate_model
from preprocessors import preprocessor_dict
from read_file import read_file

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

# need to load data to set balanced weights..
features, labels, _ = read_file(dataset)
if len(np.unique(labels))==2:
    frac = float(sum(labels==0))/float(sum(labels==1))
else:
    frac = 1

pipeline_components.append(('XGBClassifier', XGBClassifier(random_seed=random_seed, n_thread=1,scale_pos_weight=frac)))


n_estimators_values = list(range(50, 1001, 50))
learning_rate_values = np.random.uniform(low=1e-10, high=5., size=num_param_combinations)
gamma_values = np.random.uniform(low=0., high=1., size=num_param_combinations)
max_depth_values = list(range(1, 21))
subsample_values = np.random.uniform(low=0., high=1., size=num_param_combinations)

pipeline_parameters['XGBClassifier'] = \
   {'n_estimators': n_estimators_values, 'learning_rate': learning_rate_values, 'gamma': gamma_values, 'max_depth': max_depth_values, 'subsample': subsample_values}
     


evaluate_model(dataset, save_file, random_seed, pipeline_components, pipeline_parameters, num_param_combinations)
