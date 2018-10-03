import sys
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier
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
pipeline_components=[]
pipeline_parameters={}
for p in preps.split(','):
    label = sys.argv[6]
    pipeline_components.append((p, preprocessor_dict[p]))
    # if pipeline_components[-1] is SelectFromModel:
    #     pipeline_parameters[SelectFromModel] = [{'estimator': ExtraTreesClassifier(n_estimators=100, random_state=324089)}]
    # elif pipeline_components[-1] is RFE:
    #     pipeline_parameters[RFE] = [{'estimator': ExtraTreesClassifier(n_estimators=100, random_state=324089)}]


pipeline_components.append(('AdaBoostClassifier',AdaBoostClassifier()))

# parameters for method
learning_rate_values = np.random.uniform(low=1e-10, high=5., size=num_param_combinations)
n_estimators_values = np.random.choice(list(range(50, 1001, 50)), size=num_param_combinations)

pipeline_parameters['AdaBoostClassifier'] = {'learning_rate': learning_rate_values, 'n_estimators': n_estimators_values, 'random_state': 324089}
                                    ]


#evaluate
evaluate_model(dataset, save_file, random_seed, pipeline_components, pipeline_parameters, num_param_combinations, label, label)
