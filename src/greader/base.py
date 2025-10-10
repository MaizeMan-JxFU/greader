from .ref_adjust import GENOMETOOL
from bed_reader import open_bed
import pandas as pd
import gzip



def breader(prefix:str,ref_adjust:str=None) -> pd.DataFrame:
    '''ref_adjust: 基于基因组矫正, 需提供参考基因组路径'''
    with open_bed(f"{prefix}.bed",count_A1=False) as bed:
        genotype = bed.read(dtype='int8')
    fam = pd.read_csv(f'{prefix}.fam',sep=r'\s+',header=None)
    bim = pd.read_csv(f'{prefix}.bim',sep=r'\s+',header=None)
    genotype = pd.DataFrame(genotype,index=fam[0],).T
    genotype = pd.concat([bim[[0,3,4,5]],genotype],axis=1)
    genotype.columns = ['#CHROM','POS','A0','A1']+fam[0].to_list()
    genotype = genotype.set_index(['#CHROM','POS'])
    if ref_adjust is not None:
        adjust_m = GENOMETOOL(ref_adjust)
        genotype.iloc[:,:2] = adjust_m.refalt_adjust(genotype.iloc[:,:2])
        genotype.loc[adjust_m.exchange_loc,genotype.columns[2:]] = 2 - genotype.loc[adjust_m.exchange_loc,genotype.columns[2:]]
        genotype.columns = ['REF','ALT']+genotype.columns[2:].to_list()
    return genotype

def vcfreader(vcfPath:str,chunksize=10_000,ref_adjust:str=None) -> pd.DataFrame:
    '''ref_adjust: 基于基因组矫正, 需提供参考基因组路径'''
    if '.gz' == vcfPath[-3:]:
        compression = 'gzip'
        with gzip.open(vcfPath) as f:
            for line in f:
                line = line.decode('utf-8')
                if "#CHROM" in line:
                    col = line.replace('\n','').split('\t')
                    break
    else:
        compression = None
        with open(vcfPath) as f:
            for line in f:
                if "#CHROM" in line:
                    col = line.replace('\n','').split('\t')
                    break
    ncol = [0,1,3,4]+list(range(col.index('FORMAT')+1,len(col)))
    col = [col[i] for i in ncol]
    vcf_chunks = pd.read_csv(vcfPath,sep=r'\s+',comment='#',header=None,usecols=ncol,low_memory=False,compression=compression,chunksize=chunksize)
    genotype = []
    for vcf_chunk in vcf_chunks: # 分块处理vcf
        vcf_chunk:pd.DataFrame = vcf_chunk.set_index([0,1]).fillna('-9')
        ref_alt = vcf_chunk.iloc[:,:2]
        def transG(col:pd.Series):
            vcf_transdict = {'0/0':0,'1/1':2,'0/1':1,'1/0':1,'./.':-9, # Non-phased genotype
                             '0|0':0,'1|1':2,'0|1':1,'1|0':1,'.|.':-9} # Phased genotype
            return col.map(vcf_transdict).fillna(-9).astype('int8')
        vcf_chunk = vcf_chunk.iloc[:,2:].apply(transG,axis=0)
        vcf_chunk = pd.concat([ref_alt,vcf_chunk],axis=1)
        genotype.append(vcf_chunk)
    genotype = pd.concat(genotype,axis=0)
    genotype.columns = col[2:]
    genotype.index = genotype.index.rename(['#CHROM','POS'])
    genotype.columns = ['A0','A1'] + genotype.columns[2:].to_list()
    if ref_adjust is not None:
        adjust_m = GENOMETOOL(ref_adjust)
        genotype.iloc[:,:2] = adjust_m.refalt_adjust(genotype.iloc[:,:2])
        genotype.loc[adjust_m.exchange_loc,genotype.columns[2:]] = 2 - genotype.loc[adjust_m.exchange_loc,genotype.columns[2:]]
        genotype.columns = ['REF','ALT']+genotype.columns[2:].to_list()
    return genotype

def hmpreader(hmp:str,sample_start:int=None,chr:str='chrom',ps:str='position',ref:str='ref',chunksize=10_000,ref_adjust:str=None):
    raws = pd.read_csv(hmp,sep='\t',chunksize=chunksize)
    _ = []
    for raw in raws:
        samples = raw.columns[sample_start:raw.shape[0]]
        genotype = raw[samples].fillna('XX')
        def filterindel(col:pd.Series):
            col[col.str.len()!=2] = 'XX'
            return col
        genotype = genotype.apply(filterindel,axis=0)
        ref_alt:pd.Series = genotype.sum(axis=1).apply(set).apply(''.join).str.replace('X','')
        biallele = ref_alt[ref_alt.str.len()==2]
        moallele = ref_alt[ref_alt.str.len()==1]
        moallele+=moallele
        mbiallele = pd.concat([biallele,moallele])
        mbiallele = mbiallele.str.split('',expand=True)[[1,2]]
        alt:pd.Series = mbiallele[1]*(mbiallele[1]!=raw.loc[mbiallele.index,ref])+mbiallele[2]*(mbiallele[2]!=raw.loc[mbiallele.index,ref])
        alt.loc[moallele.index] = raw.loc[moallele.index,ref]
        ref_alt:pd.DataFrame = pd.concat([raw[ref],alt],axis=1).dropna()
        ref_alt.columns = ['A0','A1']
        rr = ref_alt['A0']+ref_alt['A0']
        ra = ref_alt['A0']+ref_alt['A1']
        ar = ref_alt['A1']+ref_alt['A0']
        aa = ref_alt['A1']+ref_alt['A1']
        def hmp2genotype(col:pd.Series):
            return ((col==ra)|(col==ar)).astype('int8')+2*(col==aa).astype('int8')
        genotype = genotype.loc[ref_alt.index]
        xxmask = (genotype=='XX')
        genotype = genotype.apply(hmp2genotype,axis=0)
        genotype[xxmask] = -9
        genotype[genotype==3] = 0 # Fixed some bugs: 3 is combination of REF+REF
        chr_loc = raw.loc[genotype.index,[chr,ps]]
        chr_loc.columns = ['#CHROM','POS']
        genotype = pd.concat([chr_loc,ref_alt,genotype],axis=1)
        _.append(genotype.set_index(['#CHROM','POS']))
    genotype = pd.concat(_)
    if ref_adjust is not None:
        adjust_m = GENOMETOOL(ref_adjust)
        genotype.iloc[:,:2] = adjust_m.refalt_adjust(genotype.iloc[:,:2])
        genotype.loc[adjust_m.exchange_loc,genotype.columns[2:]] = 2 - genotype.loc[adjust_m.exchange_loc,genotype.columns[2:]]
        genotype.columns = ['REF','ALT']+genotype.columns[2:].to_list()
    return genotype

def genotype2vcf(geno:pd.DataFrame,outPath:str=None,chunksize:int=10_000):
    import warnings
    warnings.filterwarnings('ignore')
    vcf_head = 'ID QUAL FILTER INFO FORMAT'.split(' ')
    geno = geno.reset_index()
    geno_ = geno.iloc[:,4:].copy()
    geno_[geno_<0] = -9
    samples = geno_.columns
    geno.columns = ['#CHROM','POS','REF','ALT']+geno.columns[4:].tolist()
    vcf = pd.DataFrame([['.','.','.','PR','GT'] for i in geno.index],columns=vcf_head)
    vcf = pd.concat([geno[['#CHROM','POS']],vcf['ID'],geno[['REF','ALT']],vcf[['QUAL','FILTER','INFO','FORMAT']],geno_],axis=1)
    def transG(col:pd.Series):
        vcf_transdict = {0:'0/0',2:'1/1',1:'0/1',-9:'./.'}
        return col.map(vcf_transdict).fillna('./.')
    if chunksize >= vcf.shape[0]:
        vcf[samples] = vcf[samples].apply(transG,axis=0)
        vcf.to_csv(f'{outPath}.vcf',sep='\t',index=None)
    else:
        for i in range(0,vcf.shape[0],chunksize):
            vcf_chunk = vcf.iloc[i:i+chunksize,:]
            vcf_chunk[samples] = vcf_chunk[samples].apply(transG,axis=0)
            if i == 0:
                vcf_chunk.to_csv(f'{outPath}.vcf',sep='\t',index=None)
            else:
                vcf_chunk.to_csv(f'{outPath}.vcf',sep='\t',index=None,header=False,mode='a')
        
if __name__ == "__main__":
    pass
