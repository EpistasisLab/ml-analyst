import sys
import itertools
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from metrics import balanced_accuracy_score
import warnings

from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory
from read_file import read_file
from utils import feature_importance , roc
import pdb

def evaluate_model(dataset, save_file, random_state, pipeline_components, pipeline_parameters, n_combos, label):

    features, labels, feature_names = read_file(dataset, label)
    # pipelines = [dict(zip(pipeline_parameters.keys(), list(parameter_combination)))
    #              for parameter_combination in itertools.product(*pipeline_parameters.values())]

    # Create a temporary folder to store the transformers of the pipeline
    cachedir = mkdtemp()
    memory = Memory(cachedir=cachedir, verbose=0)

    # print ( pipeline_components)
    # print(pipeline_parameters)
    with warnings.catch_warnings():
        # Squash warning messages. Turn this off when debugging!
        warnings.simplefilter('ignore')
        cv = StratifiedKFold(n_splits=10, shuffle=True,random_state=random_state)
        hyperparameters = {}
        for k,v in pipeline_parameters.items():
            for param,pvals in v.items():
                hyperparameters.update({k+'__'+param:pvals})
        pipeline =  Pipeline(pipeline_components, memory=memory)

        # run Randomized Search CV to tune the hyperparameter settings
        est = RandomizedSearchCV(estimator=pipeline, param_distributions = hyperparameters, n_iter = n_combos, 
                             cv=cv, random_state=random_state, refit=True,
                             error_score=0.0)
        est.fit(features, labels)
        best_est = est.best_estimator_
        # generate cross-validated predictions for each data point using the best estimator 
        cv_predictions = cross_val_predict(estimator=best_est, X=features, y=labels, cv=cv)

        # get cv probabilities
        skip = False
        if getattr(best_est, "predict_proba", None):
            method = "predict_proba"
        elif getattr(best_est, "decision_function", None):
            method = "decision_function"
        else:
            skip = True
    
        if not skip:
            cv_probabilities = cross_val_predict(estimator=best_est, X=features, y=labels, method=method, cv=cv)
            if method == "predict_proba":
                cv_probabilities = cv_probabilities[:,1]

        accuracy = accuracy_score(labels, cv_predictions)
        macro_f1 = f1_score(labels, cv_predictions, average='macro')
        balanced_accuracy = balanced_accuracy_score(labels, cv_predictions)
        roc_auc = roc_auc_score(labels,cv_probabilities)

        preprocessor_classes = [p[0] for p in pipeline_components[:-1]]
        
        preprocessor_param_string = 'default'
        for preprocessor_class in preprocessor_classes:
            if preprocessor_class in pipeline_parameters.keys():
                preprocessor_param_string = ','.join(['{}={}'.format(parameter, '|'.join([x.strip() for x in str(value).split(',')]))
                                                     for parameter, value in pipeline_parameters[preprocessor_class].items()])

        classifier_class = pipeline_components[-1][0]
        param_string = ','.join(['{}={}'.format(p, v) for p,v in est.best_params_.items()])
                                # for parameter, value in pipeline_parameters[classifier_class].items()])

        out_text = '\t'.join([dataset.split('/')[-1].split('.')[0],
                              ','.join(preprocessor_classes),
                              preprocessor_param_string,
                              classifier_class,
                              param_string,
                              str(random_state), 
                              str(accuracy),
                              str(macro_f1),
                              str(balanced_accuracy),
                              str(roc_auc)])
        print(out_text)
        with open(save_file, 'a') as out:
            out.write(out_text+'\n')
        sys.stdout.flush()
        
        # write feature importances
        est_name = classifier_class
        feature_importance(save_file, best_est, est_name, feature_names, features, labels, random_state, ','.join(preprocessor_classes), preprocessor_param_string,classifier_class, param_string)
        # write roc curves
        if not skip:
            roc(save_file, best_est, labels, cv_probabilities, random_state, ','.join(preprocessor_classes), preprocessor_param_string,classifier_class, param_string)
    # Delete the temporary cache before exiting
    rmtree(cachedir)
