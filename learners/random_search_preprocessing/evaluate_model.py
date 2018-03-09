import sys
import itertools
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from metrics import balanced_accuracy_score
import warnings

from tempfile import mkdtemp
from shutil import rmtree
from sklearn.externals.joblib import Memory
from read_file import read_file
from utils import feature_importance , roc

def evaluate_model(dataset, save_file, random_state, pipeline_components, pipeline_parameters):

    features, labels, feature_names = read_file(dataset)

    pipelines = [dict(zip(pipeline_parameters.keys(), list(parameter_combination)))
                 for parameter_combination in itertools.product(*pipeline_parameters.values())]

    # Create a temporary folder to store the transformers of the pipeline
    cachedir = mkdtemp()
    memory = Memory(cachedir=cachedir, verbose=0)

    with warnings.catch_warnings():
        # Squash warning messages. Turn this off when debugging!
        warnings.simplefilter('ignore')

        for pipe_parameters in pipelines:
            pipeline = []
            for component in pipeline_components:
                if component in pipe_parameters:
                    args = pipe_parameters[component]
                    pipeline.append(component(**args))
                else:
                    pipeline.append(component())

            try:
                clf = make_pipeline(*pipeline, memory=memory)
                cv_predictions = cross_val_predict(estimator=clf, X=features, y=labels, cv=StratifiedKFold(n_splits=10, 
                    shuffle=True, random_state=random_state))
                est = clf.fit(features,labels)

                # get cv probabilities
                skip = False
                if getattr(clf, "predict_proba", None):
                    method = "predict_proba"
                elif getattr(clf, "decision_function", None):
                    method = "decision_function"
                else:
                    skip = True
                    
                if not skip:
                    cv_probabilities = cross_val_predict(estimator=clf, X=features, y=labels, method=method, 
                                                        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state))
                    if method == "predict_proba":
                        cv_probabilities = cv_probabilities[:,1]

                accuracy = accuracy_score(labels, cv_predictions)
                macro_f1 = f1_score(labels, cv_predictions, average='macro')
                balanced_accuracy = balanced_accuracy_score(labels, cv_predictions)
            except KeyboardInterrupt:
                sys.exit(1)
            # This is a catch-all to make sure that the evaluation won't crash due to a bad parameter
            # combination or bad data. Turn this off when debugging!
            #except Exception as e:
            #    continue

            preprocessor_classes = [p.__name__ for p in pipeline_components[:-1]]
            
            preprocessor_param_string = 'default'
            for preprocessor_class in preprocessor_classes:
                if preprocessor_class in pipe_parameters:
                    preprocessor_param_string = ','.join(['{}={}'.format(parameter, '|'.join([x.strip() for x in str(value).split(',')]))
                                                         for parameter, value in pipe_parameters[preprocessor_class].items()])

            classifier_class = pipeline_components[-1]
            param_string = ','.join(['{}={}'.format(parameter, value)
                                    for parameter, value in pipe_parameters[classifier_class].items()])

            out_text = '\t'.join([dataset.split('/')[-1].split('.')[0],
                                  ','.join(preprocessor_classes),
                                  preprocessor_param_string,
                                  classifier_class.__name__,
                                  param_string,
                                  str(random_state), 
                                  str(accuracy),
                                  str(macro_f1),
                                  str(balanced_accuracy)])
            print(out_text)
            with open(save_file, 'a') as out:
                out.write(out_text+'\n')
            sys.stdout.flush()
            
            # write feature importances
            est_name = classifier_class.__name__.lower()
            feature_importance(save_file, est, est_name, feature_names, features, labels, random_state, ','.join(preprocessor_classes), preprocessor_param_string,classifier_class.__name__, param_string)
            # write roc curves
            if not skip:
                roc(save_file, est, labels, cv_probabilities, random_state, ','.join(preprocessor_classes), preprocessor_param_string,classifier_class.__name__, param_string)
    # Delete the temporary cache before exiting
    rmtree(cachedir)
