import sys
import pandas as pd
import numpy as np

#from sklearn.preprocessing import Binarizer, MaxAbsScaler, MinMaxScaler
#from sklearn.preprocessing import Normalizer, PolynomialFeatures, RobustScaler, StandardScaler
#from sklearn.decomposition import FastICA, PCA
#from sklearn.kernel_approximation import RBFSampler, Nystroem
#from sklearn.cluster import FeatureAgglomeration
#from sklearn.feature_selection import SelectFwe, SelectPercentile, VarianceThreshold
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import AdaBoostClassifier
from evaluate_model import evaluate_model
from preprocessors import preprocessor_dict

dataset = sys.argv[1]
save_file = sys.argv[2]
num_param_combinations = int(sys.argv[3])
random_seed = int(sys.argv[4])
preps = sys.argv[5]
print('arguments:')
for i,arg in enumerate(sys.argv[1:]):
    print(i+1,arg)
np.random.seed(random_seed)

pipeline_components=[]
pipeline_parameters={}
for p in preps.split(','):
    pipeline_components.append(preprocessor_dict[p])
    if pipeline_components[-1] is SelectFromModel:
        pipeline_parameters[SelectFromModel] = [{'estimator': ExtraTreesClassifier(n_estimators=100, random_state=324089)}]
    elif pipeline_components[-1] is RFE:
        pipeline_parameters[RFE] = [{'estimator': ExtraTreesClassifier(n_estimators=100, random_state=324089)}]

pipeline_components.append(AdaBoostClassifier)
#preprocessor_list = [Binarizer, MaxAbsScaler, MinMaxScaler, Normalizer,
#                     PolynomialFeatures, RobustScaler, StandardScaler,
#                     FastICA, PCA, RBFSampler, Nystroem, FeatureAgglomeration,
#                     SelectFwe, SelectPercentile, VarianceThreshold,
#                     SelectFromModel, RFE]

#chosen_preprocessor = preprocessor_list[preprocessor_num]

#pipeline_components = [chosen_preprocessor, AdaBoostClassifier]
#pipeline_parameters = {}

learning_rate_values = np.random.uniform(low=1e-10, high=5., size=num_param_combinations)
n_estimators_values = np.random.choice(list(range(50, 1001, 50)), size=num_param_combinations)

all_param_combinations = zip(learning_rate_values, n_estimators_values)
pipeline_parameters[AdaBoostClassifier] = [{'learning_rate': learning_rate, 'n_estimators': n_estimators, 'random_state': 324089}
                                    for (learning_rate, n_estimators) in all_param_combinations]

#if chosen_preprocessor is SelectFromModel:
#    pipeline_parameters[SelectFromModel] = [{'estimator': ExtraTreesClassifier(n_estimators=100, random_state=324089)}]
#elif chosen_preprocessor is RFE:
#    pipeline_parameters[RFE] = [{'estimator': ExtraTreesClassifier(n_estimators=100, random_state=324089)}]
#
evaluate_model(dataset, save_file, random_seed, pipeline_components, pipeline_parameters)
