import numpy as np
from read_file import read_file
from eli5.sklearn import PermutationImportance
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def feature_importance(save_file, model, feature_names, training_features, training_classes, random_state):
    """ prints feature importance information for a trained estimator (model)"""
    coefs = compute_imp_score(model, training_features, training_classes, random_state)
#    plot_imp_score(save_file, coefs, feature_names, random_state)

    out_text=''
    # algorithm seed    feature score
    for i,c in enumerate(coefs):
        out_text += '\t'.join([save_file.split('/')[-1][:-4],    # model name
                              str(random_state),
                              feature_names[i],
                              str(c)])+'\n'
        
    with open(save_file.split('.')[0] + '.imp_score','a') as out:
        out.write(out_text)

def compute_imp_score(model, training_features, training_classes, random_state):
    if hasattr(model, 'coef_'):
        coefs = np.abs(model.coef_)
    else:
        coefs = getattr(model, 'feature_importances_', None)
    if coefs is None:
        perm = PermutationImportance(
                                    estimator=model,
                                    n_iter=5,
                                    random_state=random_state,
                                    refit=False
                                    )
        perm.fit(training_features, training_classes)
        coefs = perm.feature_importances_

    if coefs.ndim > 1:
        coefs = safe_sqr(coefs).sum(axis=0)

    return (coefs-np.min(coefs))/(np.max(coefs)-np.min(coefs))

# def plot_imp_score(save_file, coefs, feature_names, seed):
#     # plot bar charts for top 10 importanct features
#     num_bar = min(10, len(coefs))
#     indices = np.argsort(coefs)[-num_bar:]
#     h=plt.figure()
#     plt.title("Feature importances")
#     plt.barh(range(num_bar), coefs[indices], color="r", align="center")
#     plt.yticks(range(num_bar), feature_names[indices])
#     plt.ylim([-1, num_bar])
#     h.tight_layout()
#     plt.savefig(save_file.split('.')[0] + '_imp_score_' + str(seed) + '.pdf')

######################################################################################### ROC Curve

def roc(save_file, model, y_true, probabilities, random_state):
    """prints receiver operator chacteristic curve data"""

    fpr,tpr,_ = roc_curve(y_true, probabilities)

    AUC = auc(fpr,tpr)
    model_name = save_file.split('/')[-1][:-4]
    # print results
    out_text=''
    for f,t in zip(fpr,tpr):
        out_text += '\t'.join([model_name,
                          str(random_state),
                          str(f),
                          str(t),
                          str(AUC)])+'\n'

    with open(save_file.split('.')[0] + '.roc','a') as out:
        out.write(out_text)


