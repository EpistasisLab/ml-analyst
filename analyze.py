import pandas as pd
import numpy as np
import argparse
import os, errno, sys
from sklearn.externals.joblib import Parallel, delayed


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser(description="An analyst for quick ML applications.",
                                     add_help=False)
    parser.add_argument('INPUT_FILE', type=str,
                        help='Data file to analyze; ensure that the '
                        'target/label column is labeled as "class".')    
    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')
    parser.add_argument('-ml', action='store', dest='LEARNERS',default=None,type=str, 
            help='Comma-separated list of ML methods to use (should correspond to a py file name in
            ml/)')
    parser.add_argument('-prep', action='store', dest='PREP', default=None, type=str, 
            help = 'Comma-separated list of preprocessors to apply to data')
    parser.add_argument('--lsf', action='store_true', dest='LSF', default=False, 
            help='Run on an LSF HPC (using bsub commands)')
    parser.add_argument('-metric',action='store', dest='METRIC', default='f1_macro', type=str, 
            help='Metric to compare algorithms')
    parser.add_argument('-k',action='store', dest='K', default=5, type=int, 
            help='Number of folds for cross validation')
    parser.add_argument('-search',action='store',dest='SEARCH',default='random',choices=['grid','random'],
            help='Hyperparameter search strategy')
    parser.add_argument('--r',action='store_true',dest='REGRESSION',default=False,
            help='Run regression instead of classification.')
    parser.add_argument('-n_jobs',action='store',dest='N_JOBS',default=4,type=int,
            help='Number of parallel jobs')
    parser.add_argument('-n_trials',action='store',dest='N_TRIALS',default=1,type=int,
            help='Number of parallel jobs')
    parser.add_argument('-n_combos',action='store',dest='N_COMBOS',default=4,type=int,
            help='Number of hyperparameters to try')
    parser.add_argument('-rs',action='store',dest='RANDOM_STATE',default=None,type=int,
            help='random state')
    parser.add_argument('-label',action='store',dest='LABEL',default='class',type=str,help='Name of class label column')

    args = parser.parse_args()
      
    if args.RANDOM_STATE:
        random_state = args.RANDOM_STATE
    else:
        random_state = np.random.randint(2**32 - 1)

    learners = [ml for ml in args.LEARNERS.split(',')]  # learners

    if args.SEARCH == 'random':
        if args.PREP:
            model_dir = 'ml/random_search_preprocessing/'
        else:
            model_dir = 'ml/random_search/'
    else:
        model_dir= 'ml/grid_search/' 

    dataset = args.INPUT_FILE.split('/')[-1].split('.')[0]
    RANDOM_STATE = args.RANDOM_STATE
    results_path = '/'.join(['results', dataset]) + '/'
    # make the results_path directory if it doesn't exit 
    try:
        os.makedirs(results_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    # initialize output files
    for ml in learners:
        #write headers
        if args.PREP:
            save_file = results_path + '-'.join(args.PREP.split(',')) + '_' + ml + '.csv'  
        else:
            save_file = results_path + '_' + ml + '.csv'  
        feat_file =  save_file.split('.')[0]+'.imp_score'        
        roc_file =  save_file.split('.')[0]+'.roc'        
        
        with open(save_file.split('.')[0] + '.imp_score','w') as out:
            out.write('preprocessor\tprep-parameters\talgorithm\talg-parameters\tseed\tfeature\tscore\n')
         
        with open(save_file.split('.')[0] + '.roc','w') as out:
            out.write('preprocessor\tprep-parameters\talgorithm\talg-parameters\tseed\tfpr\ttpr\tauc\n')
   
        with open(save_file,'w') as out:
            if args.PREP:
                out.write('dataset\tpreprocessor\tprep-parameters\talgorithm\talg-parameters\tseed\taccuracy\tf1_macro\tbal_accuracy\troc_auc\n')
            else:
                out.write('dataset\talgorithm\tparameters\taccuracy\tf1_macro\tseed\tbal_accuracy\troc_auc\n')
        
    # write run commands
    all_commands = []
    for t in range(args.N_TRIALS):
        random_state = np.random.randint(2**32-1)
        for ml in learners:
            if args.PREP:
                save_file = results_path + '-'.join(args.PREP.split(',')) + '_' + ml + '.csv'  
            else:
                save_file = results_path + '_' + ml + '.csv'          
            
            if args.PREP: 
                all_commands.append('python {PATH}/{ML}.py {DATASET} {SAVEFILE} {N_COMBOS} {RS} {PREP} {LABEL}'.format(PATH=model_dir,ML=ml,DATASET=args.INPUT_FILE,SAVEFILE=save_file,N_COMBOS=args.N_COMBOS,RS=random_state,PREP=args.PREP,LABEL=args.LABEL)) 
            else :
                all_commands.append('python {ML}.py {DATASET} {SAVEFILE} {N_COMBOS} {RS}'.format(PATH=model_dir,ML=ml,DATASET=args.INPUT_FILE,SAVEFILE=save_file,N_COMBOS=args.N_COMBOS,RS=random_state))

    if args.LSF:    # bsub commands
        for run_cmd in all_commands:
            job_name = ml + '_' + dataset
            out_file = results_path + job_name + '_%J.out'
            error_file = out_file[:-4] + '.err'
            
            bsub_cmd = ('bsub -o {OUT_FILE} -e {ERROR_FILE} -n {N_CORES} -J {JOB_NAME} -q {QUEUE} '
                       '-R "span[hosts=1]" ').format(OUT_FILE=out_file,
                                             ERROR_FILE=error_file,
                                             JOB_NAME=job_name,
                                             QUEUE=args.QUEUE,
                                             N_CORES=n_cores)
            
            bsub_cmd +=  '"' + run_cmd + '"'
            print(bsub_cmd)
            os.system(bsub_cmd)     # submit jobs 
    else:   # run locally  
        Parallel(n_jobs=args.N_JOBS)(delayed(os.system)(run_cmd) for run_cmd in all_commands )
