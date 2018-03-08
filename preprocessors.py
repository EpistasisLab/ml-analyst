from sklearn.preprocessing import Binarizer, MaxAbsScaler, MinMaxScaler
from sklearn.preprocessing import Normalizer, PolynomialFeatures, RobustScaler, StandardScaler
from sklearn.decomposition import FastICA, PCA
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.cluster import FeatureAgglomeration
from sklearn.feature_selection import SelectFwe, SelectPercentile, VarianceThreshold
from sklearn.feature_selection import SelectFromModel, RFE

preprocessor_dict = {
        'Binarizer':Binarizer, 
        'MaxAbsScaler':MaxAbsScaler, 
        'MinMaxScaler':MinMaxScaler,
        'Normalizer':Normalizer,
        'PolynomialFeatures':PolynomialFeatures,
        'RobustScaler':RobustScaler,
        'StandardScaler':StandardScaler,
        'FastICA':FastICA,
        'PCA':PCA,
        'RBFSampler':RBFSampler,
        'Nystroem':Nystroem,
        'FeatureAgglomeration':FeatureAgglomeration,
        'SelectFwe':SelectFwe,
        'SelectPercentile':SelectPercentile,
        'VarianceThreshold':VarianceThreshold,
        'SelectFromModel':SelectFromModel,
        'RFE':RFE,
        } 
