from glob import glob
import os
import sys
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Submit long jobs.",
                                     add_help=False)
    parser.add_argument('DATA_PATH',type=str)
    parser.add_argument('-ml',action='store',dest='mls', type=str, 
            default='LogisticRegression,RandomForestClassifier,GradientBoostingClassifier,'
                    'DecisionTreeClassifier,SVC')
    parser.add_argument('--long',action='store_true',dest='LONG', default=False)
    parser.add_argument('-n_trials',action='store',dest='TRIALS', default=1)
    parser.add_argument('-results',action='store',dest='RDIR',default='../results/',type=str,
                        help='Results directory')
    parser.add_argument('--lsf', action='store_true', dest='LSF', default=False, 
            help='Run on an LSF HPC (using bsub commands)')
    parser.add_argument('-m',action='store',dest='M',default=4096,type=int,
                        help='LSF memory request and limit (MB)')
    args = parser.parse_args()

    datapath = args.DATA_PATH 

    if args.LONG:
        q = 'moore_long'
    else:
        q = 'moore_normal'

    lpc_options = '--lsf -q {Q} -m {M} -n_jobs 1'.format(Q=q,M=args.M)
    
    lsf = '--lsf' if args.LSF else ''

    mls = ','.join([ml for ml in args.mls.split(',')])
    for f in glob(datapath + "/*.csv"):
        jobline =  ('python analyze.py {DATA} '
                   '-ml {ML} '
                   '-results {RDIR} -n_trials {NT} '
                   '-search grid {LPC} {LSF}').format(DATA=f,
                                                      LPC=lpc_options,
                                                      ML=mls,
                                                      RDIR=args.RDIR,
                                                      NT=args.TRIALS,
                                                      LSF=lsf)
        print(jobline)
        os.system(jobline)

