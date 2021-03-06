import sys
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import SGDClassifier
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


pipeline_components.append('SGDClassifier', SGDClassifier())


loss_values = np.random.choice(['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'], size=num_param_combinations)
penalty_values = np.random.choice(['l2', 'l1', 'elasticnet'], size=num_param_combinations)
alpha_values = np.random.exponential(scale=0.01, size=num_param_combinations)
learning_rate_values = np.random.choice(['constant', 'optimal', 'invscaling'], size=num_param_combinations)
fit_intercept_values = np.random.choice([True, False], size=num_param_combinations)
l1_ratio_values = np.random.uniform(low=0., high=1., size=num_param_combinations)
eta0_values = np.random.uniform(low=0., high=5., size=num_param_combinations)
power_t_values = np.random.uniform(low=0., high=5., size=num_param_combinations)

pipeline_parameters['SGDClassifier'] = \
   [{'loss': loss_values, 'penalty': penalty, 'alpha': alpha_values, 'learning_rate': learning_rate, 'fit_intercept': fit_intercept_values,
     'l1_ratio': 0.15 if penalty != 'elasticnet' else l1_ratio, 'eta0': 0. if learning_rate not in ['constant', 'invscaling'] else eta0,
     'power_t': 0.5 if learning_rate != 'invscaling' else power_t, 'random_state': 324089}
     for penalty, l1_ratio, learning_rate, eta0, power_t in zip(penalty_values, l1_ratio_values, learning_rate_values, eta0_values, power_t_values)]
     


evaluate_model(dataset, save_file, random_seed, pipeline_components, pipeline_parameters, num_param_combinations, label)
