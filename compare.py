import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

import argparse
from glob import glob
import pdb
def main():
    """Analyzes results and generates figures."""
 
    parser = argparse.ArgumentParser(description="An analyst for quick ML applications.",
                                     add_help=False)
  
    parser.add_argument('RUN_DIR', action='store',  type=str, help='Path to results from analysis.')    

    args = parser.parse_args()
   
    # dataset = args.NAME
    # dataset = args.NAME.split('/')[-1].split('.')[0] 
    # run_dir = 'results/' + dataset + '/' 
    run_dir = args.RUN_DIR
    if run_dir[-1] != '/': 
        run_dir += '/'
    dataset = run_dir.split('/')[-2]
    print('dataset:',dataset)
    print('loading data from',run_dir)

    frames = []     # data frames to combine
    count = 0
    for f in glob(run_dir + '*.csv'):
        if 'imp_score' not in f:
            frames.append(pd.read_csv(f,sep='\t',index_col=False))
            count = count + 1

    df = pd.concat(frames, join='outer', ignore_index=True)

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
        plt.savefig(run_dir + '_'.join([ dataset, col,'boxplots.pdf']))
    

    ####################################################################### feature importance plots
    frames = []     # data frames to combine
    count = 0
    for f in glob(run_dir + '*.imp_score'):
        frames.append(pd.read_csv(f,sep='\t',index_col=False))
        count = count + 1

    df = pd.concat(frames, join='outer', ignore_index=True)
    df['prep_alg'] = df['preprocessor'] + '_' + df['algorithm']
    # pdb.set_trace()
    print('loaded',count,'feature importance files with results from these learners:',df['prep_alg'].unique())

    dfp =  df.groupby(['prep_alg','feature']).median().unstack('prep_alg')

    h = dfp['score'].plot(kind='bar', stacked=True, sort_columns=True)
    leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.ylabel('Importance Score')
    plt.savefig(run_dir + '_'.join([ dataset, col,'importance_scores.pdf']),bbox_extra_artists=(leg,h), bbox_inches='tight')

    ############################################################# roc curves
    frames = []     # data frames to combine
    count = 0
    for f in glob(run_dir + '*.roc'):
        frames.append(pd.read_csv(f,sep='\t',index_col=False))
        count = count + 1

    df = pd.concat(frames, join='outer', ignore_index=True)
    df['prep_alg'] = df['preprocessor'] + '_' + df['algorithm']
    
    print('loaded',count,'roc files with results from these learners:',df['prep_alg'].unique())


    plt.figure()
    plt.plot([0, 1],[0, 1],'--k')
    colors = ('r','y','b','g','c','k')
    n_algs = len(df['prep_alg'].unique())
    for i, (alg,df_g) in enumerate(df.groupby('prep_alg')):
    
        auc = df_g.auc.median()
        tpr  = [df_g.tpr.values[s] for s in np.argsort(df_g.fpr.values)]
        fpr = np.sort(df_g.fpr)
        plt.plot(fpr, tpr, label='{:s} (AUC = {:0.2f})'
                   ''.format(alg,auc),
                   color=colors[i % n_algs], linestyle=':', linewidth=1)

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.savefig(run_dir + '_'.join([ dataset, col,'roc_curves.pdf']),bbox_extra_artists=(leg,h), bbox_inches='tight')

    print('done!')    

if __name__ == '__main__':
    main()
