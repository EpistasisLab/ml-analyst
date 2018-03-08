import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from glob import glob
import pdb 

def main():
    """Analyzes results and generates figures."""
 
    parser = argparse.ArgumentParser(description="An analyst for quick ML applications.",
                                     add_help=False)
  
    parser.add_argument('-name', action='store', dest='NAME', type=str, help='Data file that was analyzed; ensure that the '
                        'results have been generated in results/[filename].')    

    args = parser.parse_args()
   
    dataset = args.NAME
    # dataset = args.INPUT_FILE.split('/')[-1].split('.')[0] 
    run_dir = 'results/' + dataset + '/' 

    print('loading data from',run_dir)

    frames = []     # data frames to combine
    count = 0
    for f in glob(run_dir + '*.csv'):
        if 'imp_score' not in f:
            frames.append(pd.read_csv(f,sep='\t',index_col=False))
            count = count + 1

    df = pd.concat(frames, join='outer', ignore_index=True)

    pdb.set_trace()

    print('loaded',count,'result files with results from these learners:',df['algorithm'].unique())

    restricted_cols = ['preprocessor', 'prep-parameters', 'algorithm', 'alg-parameters','dataset',
                      'trial','seed']
    columns_to_plot = [c for c in df.columns if c not in restricted_cols ] 
    #['accuracy','f1_macro','bal_accuracy']
    print('generating boxplots for these columns:',columns_to_plot)

    for col in columns_to_plot:
        plt.figure()
        sns.boxplot(data=df,x="algorithm",y=col)
        plt.title(dataset,size=16)
        plt.gca().set_xticklabels(df.algorithm.unique(),size=14,rotation=45)
        plt.ylabel(col,size=16)
        #plt.ylim(0.5,1.0)
        plt.xlabel('')
        plt.tight_layout() 
        plt.savefig('_'.join([run_dir, dataset, col,'boxplots.pdf']))
    
    print('done.') 
    
if __name__ == '__main__':
    main()
