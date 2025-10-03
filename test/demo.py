from biokit.pymlm import MLM
from biokit.pymlm import PCA
from biokit.pymlm import KIN
from biokit.greader import vcfreader,breader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange

def test_greader(vcf=True,ref_adjust=None):
    if vcf:
        out = vcfreader('example/maize350_9496_SNP',ref_adjust=ref_adjust)
    else:
        out = breader('example/maize350_9496_SNP',ref_adjust=ref_adjust)
    return out

def test_pca():
    geno_matrix = test_greader(False).iloc[:,2:].T
    model = PCA()
    model.fit_transfer(geno_matrix.values)
    pc3 = pd.DataFrame(model.egvec[:,:4],index=geno_matrix.index)
    sns.scatterplot(pc3,x=0,y=2)
    plt.savefig('test.pdf')

def test_kinship():
    geno_matrix = test_greader(True,'example/Zea_mays.B73_RefGen_v4.dna.toplevel2.fa').iloc[:,2:].T
    model = KIN(geno_matrix.values,method='gemma1')
    kin = model.chunk_kinship(10)
    print(kin[:5,:5])
    kin = model.kinship()
    print(kin[:5,:5])

def test_lm():
    import numpy as np
    import scipy.stats
    geno = breader('example/maize350_9496_SNP',).iloc[:,2:].T
    pheno = pd.read_csv('example/pheno.tsv',sep='\t',index_col=0).iloc[:,[0]].dropna()
    cind = list(set(geno.index) & set(pheno.index))
    x = geno.loc[cind].values
    y = pheno.loc[cind].values
    pca_model = PCA(10)
    pca_model.fit_transfer(x)
    cov = pca_model.egvec[:,:10]
    beta = []
    beta_se = []
    R2 = []
    for i in trange(x.shape[1]):
        # lm_model = MLM(x[:,[i]],y)
        lm_model = MLM(np.concatenate([x[:,[i]],cov],axis=1),y)
        lm_model.fit_lm()
        beta.append(lm_model.beta[0,0])
        beta_se.append(lm_model.SE[0,0])
        R2.append(lm_model.R2[0,0])
    gwas = pd.DataFrame([beta,beta_se,R2],columns=geno.columns,index=['beta','se','r2']).T
    gwas['t'] = gwas['beta']/gwas['se']
    gwas['p'] = scipy.stats.t.sf(gwas['t'].abs(),df=len(cind))*2
    print(gwas)

def mlm_test():
    from biokit.pymlm import BLUP
    geno = breader('example/maize350_9496_SNP',).iloc[:,2:].T
    pheno = pd.read_csv('example/pheno.tsv',sep='\t',index_col=0).iloc[:,[4]].dropna()
    print(pheno.columns)
    g_p = pd.concat([geno,pheno],axis=1).dropna()
    x = g_p.iloc[:,:-1].values
    y = g_p.iloc[:,[-1]].values
    pca_model = PCA(10)
    pca_model.fit_transfer(x,95)
    cov = pca_model.egvec[:,:1]
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5,shuffle=True,random_state=430)  # 初始化KFold
    for i in [None,'pearson','VanRanden','gemma1','gemma2']:
        _ = []
        _hat = []
        for train_index , test_index in kf.split(x,y):
            model = BLUP(y[train_index,:],None,x[train_index,:],kinship=i) # gemma1 gemma2 VanRanden
            model.fit()
            y_hat = model.predict(None,x[test_index,:])
            _+=y[test_index,:].tolist()
            _hat+=y_hat.tolist()
        import matplotlib.pyplot as plt
        from scipy.stats import pearsonr
        print(f'{i}({round(model.pve,3)})',pearsonr(_,_hat).statistic**2)
    plt.scatter(_,_hat)
    plt.plot(_,_,color='red',linestyle='dashed')
    plt.savefig('test.png')

def PCSHOW_test():
    from biokit.bioplot import PCSHOW
    import matplotlib.pyplot as plt
    # geno_matrix = breader('1604.507.GAIN/admixture/1604_507_91_87').iloc[:,2:].T
    # model = PCA()
    # model.fit_transfer(geno_matrix.values)
    # pc3 = pd.DataFrame(model.egvec[:,:3],index=geno_matrix.index)
    pc3 = pd.read_csv('1604.507.GAIN/1604_507_91_87.eigenvec',sep=r'\s+',index_col=0,header=None).iloc[:,1:4]
    pc3 = pc3[~pc3.index.duplicated(keep='first')] # 保留第一个
    anno1 = pd.read_csv('1604.507.GAIN/1604.tag',sep=r'\s+',index_col=0,header=None)
    anno2 = pd.read_csv('1604.507.GAIN/507.tag',sep=r'\s+',index_col=0,header=None)
    anno = pd.concat([anno1,anno2])
    anno[anno[1]=='Admix'] = 'Mixed'
    anno = anno[~anno.index.duplicated(keep='first')] # 保留第一个
    color_d = dict(zip(anno[1].unique(),sns.color_palette("hls", len(anno[1].unique()))))
    alpha_d = dict(zip(anno[1].unique(),[.6 for i in anno[1].unique()]))
    size_d = dict(zip(anno[1].unique(),[8 for i in anno[1].unique()]))
    anno.columns = ['group']
    color_d['Mixed'] = 'grey'
    color_d['YZG'] = 'lime'
    alpha_d['Mixed'] = .2
    size_d['Mixed'] = 4
    size_d['YZG'] = 32
    group_order = ['PA','PB','SS','NSS','SPT','Mixed','Iodent','TST','YZG']
    ff = plt.figure(figsize=(10,4),dpi=300)
    pmodel = PCSHOW(pc3,anno)
    anno_text = anno.loc[['B73','Mo17','Zheng58','Huangzaosi','PHP02','Ki3',]]
    anno_text2 = anno.loc[['QD001','D437']]
    anno_text.loc[:,'anno'] = anno_text.index
    anno_text2.loc[:,'anno'] = anno_text2.index
    for ind,i in enumerate([3,4]):
        ax = ff.add_subplot(1,2,ind+1)
        pmodel.pcplot(x=2,y=i,group='group',color=color_d,alpha=alpha_d,ax=ax,group_order=group_order,size=size_d,linewidths=0)
        pmodel.text_anno(x=2,y=i,anno=anno_text,anno_tag='anno',ax=ax,fontsize=12,)
        pmodel.text_anno(x=2,y=i,anno=anno_text2,anno_tag='anno',ax=ax,fontsize=6,color='black')
    plt.tight_layout()
    legend = plt.legend()
    for handle in legend.legend_handles:
        handle.set_sizes([30])
        handle.set_alpha(1)
    plt.savefig('test.png')

def manhanden_test():
    data = pd.read_csv('example/jmpheno.assoc.txt',sep='\t')
    from biokit.bioplot import GWASPLOT
    import matplotlib.pyplot as plt
    mplot = GWASPLOT(data,'chr','ps','p_wald',interval_rate=0.1)
    ff = plt.figure(figsize=(8,4),dpi=300)
    ax = ff.add_subplot(121)
    mplot.manhattan(3.5,['black','grey'],ax=ax)
    ax = ff.add_subplot(122)
    mplot.qq(ax=ax)
    plt.tight_layout()
    plt.savefig('test.pdf')

test_pca()