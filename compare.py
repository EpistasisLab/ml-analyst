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

    h, ax = plt.subplots()
    ax.plot([0, 1],[0, 1],'--k',label='_nolegend_')
    colors = ('r','y','b','g','c','k')
    colors = plt.cm.Spectral(np.linspace(0.1, 0.9, len(df['prep_alg'].unique())))

    n_algs = len(df['prep_alg'].unique())
    for i, (alg,df_g) in enumerate(df.groupby('prep_alg')):
   
        aucs = df_g.auc.values
        seed_max = df_g.loc[df_g.auc.idxmax()]['seed']
        seed_min = df_g.loc[df_g.auc.idxmin()]['seed']

        auc = df_g.auc.median()
        # fpr = df_g['fpr'].unique()
        tprs,fprs=[],[]
        fpr_min = df_g.loc[df_g.seed == seed_min,:]['fpr']
        fpr_max = df_g.loc[df_g.seed == seed_max,:]['fpr']
        tpr_min = df_g.loc[df_g.seed == seed_min,:]['tpr']
        tpr_max = df_g.loc[df_g.seed == seed_max,:]['tpr']
        
        ax.plot(fpr_max,tpr_max, color=colors[i % n_algs], linestyle='--', linewidth=1,
                label='_nolegend_')
        ax.plot(fpr_min,tpr_min,color=colors[i % n_algs], linestyle='--', linewidth=1,
                label='{:s} (AUC = {:0.2f})'.format(alg,auc))
        # for seed,df_g_s in df_g.groupby('seed'):
        #     tprs.append(df_g_s['tpr'].values)
        #     fprs.append(df_g_s['fpr'].values)
        # print('tprs list:',len(tprs),[t.shape for t in tprs])        
        # print('fprs list:',len(fprs),[f.shape for f in fprs])        
        # tprs = np.array(tprs)
        # fprs = np.array(fprs)
        # print('tprs',tprs.shape,'fprs',fprs.shape)
        # fpr = np.mean(fprs,axis=1)
        # tpr = np.mean(tprs,axis=1)
        # tpr_std = np.std(tprs,axis=1)
        # # for f,t in zip(fprs,tprs):
        # ax.plot(fpr,tpr, label='{:s} (AUC = {:0.2f})'.format(alg,auc))
        # print('fpr size:',len(fpr))
        # print('fpr:',fpr)
        # tpr = df_g.groupby('fpr')['tpr'].mean().values
        # tpr_std = df_g.groupby('fpr')['tpr'].std().values
        # print('tpr size:',len(tpr))
        # print('tpr:',tpr)

        # tpr  = np.array([tpr[s] for s in np.argsort(fpr)])
        # tpr_std  = np.array([tpr_std[s] for s in np.argsort(fpr)])
        # fpr = np.sort(fpr)
        # ax.plot(fpr,tpr+tpr_std)
        # ax.fill_between(fpr, tpr+tpr_std, tpr-tpr_std, alpha=0.1,
        #            label='{:s} (AUC = {:0.2f})'
        #            ''.format(alg,auc),
        #            color=colors[i % n_algs], linestyle=':', linewidth=1)

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.savefig(run_dir + '_'.join([ dataset, col,'roc_curves.pdf']), bbox_inches='tight')

    print('done!')    

if __name__ == '__main__':
    main()
