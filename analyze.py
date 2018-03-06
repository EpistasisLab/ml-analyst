import pandas as pd
import numpy as np
import argparse

if __name__ == 'main':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="An analyst for quick ML applications.",
                                     add_help=False)
    parser.add_argument('INPUT_FILE', type=str,
                        help='Data file to analyze; ensure that the '
                        'target/label column is labeled as "label" or "class".')    
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-learners', action='store', dest='LEARNERS',default=None,type=string, help='Comma-separated list of ML methods to use (should correspond to a py file name in learners/)')
    parser.add_argument('-prep', action='store', dest='PREP', default=None, type=string, help = 'Comma-separated list of preprocessors to apply to data')
    parser.add_argument('--lsf' action='store_true', dest='LSF', default=False, type=string, help='Run on an LSF HPC (using bsub commands)')
    parser.add_argument('-metric',action='store', dest='METRIC', default='f1_macro', type=string, help='Metric to compare algorithms')
    parser.add_argument('-k',action='store', dest='K', default=5, type=int, help='Number of folds for cross validation')
    parser.add_argument('-search',action='store',dest='SEARCH',default='grid',options=['grid','random'],help='Hyperparameter search strategy')
    parser.add_argument('--r',action='store_true',dest='REGRESSION',default=False,help='Run regression instead of classification.')

    args = parser.parse_args()


