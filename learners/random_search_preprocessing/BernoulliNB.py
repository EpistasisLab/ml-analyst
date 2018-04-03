import sys
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.naive_bayes import BernoulliNB
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


pipeline_components.append('BernoulliNB', BernoulliNB ())


alpha_values = np.random.uniform(low=0., high=50., size=num_param_combinations)
fit_prior_values = np.random.choice([True, False], size=num_param_combinations)
binarize_values = np.random.uniform(low=0., high=1., size=num_param_combinations)

pipeline_parameters[BernoulliNB] = {'alpha': alpha_values, 'fit_prior': fit_prior_values, 'binarize': binarize_values}
                                    


evaluate_model(dataset, save_file, random_seed, pipeline_components, pipeline_parameters, num_param_combinations)
