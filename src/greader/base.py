from .ref_adjust import GENOMETOOL
from bed_reader import open_bed
import numpy as np
import pandas as pd
import gzip



def breader(prefix:str,ref_adjust:str=None) -> pd.DataFrame:
    '''ref_adjust: 基于基因组矫正, 需提供参考基因组路径'''
    with open_bed(f"{prefix}.bed",count_A1=False) as bed:
        genotype = bed.read(dtype=np.int8)
    fam = pd.read_csv(f'{prefix}.fam',sep=r'\s+',header=None)
    bim = pd.read_csv(f'{prefix}.bim',sep=r'\s+',header=None)
    genotype = pd.DataFrame(genotype,index=fam[0],).T
    genotype = pd.concat([bim[[0,3,4,5]],genotype],axis=1)
    genotype.columns = ['CHROM','POS','A0','A1']+fam[0].to_list()
    genotype = genotype.set_index(['CHROM','POS'])
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
    col = np.array(col)[ncol]
    vcf_chunks = pd.read_csv(vcfPath,sep=r'\s+',comment='#',header=None,usecols=ncol,low_memory=True,compression=compression,chunksize=chunksize)
    genotype = []
    for vcf_chunk in vcf_chunks: # 分块处理vcf
        vcf_chunk:pd.DataFrame = vcf_chunk.set_index([0,1]).fillna('-9')
        ref_alt = vcf_chunk.iloc[:,:2]
        def transG(col:pd.Series):
            vcf_transdict = {'0/0':0,'1/1':2,'0/1':1,'1/0':1,'./.':-9,'0|0':0,'1|1':2,'0|1':1,'1|0':1,'.|.':-9}
            return col.map(vcf_transdict).astype(np.int8)
        vcf_chunk = vcf_chunk.iloc[:,2:].apply(transG,axis=0)
        vcf_chunk = pd.concat([ref_alt,vcf_chunk],axis=1)
        genotype.append(vcf_chunk)
    genotype = pd.concat(genotype,axis=0)
    genotype.columns = col[2:]
    genotype.index = genotype.index.rename(['CHROM','POS'])
    genotype.columns = ['A0','A1'] + genotype.columns[2:].to_list()
    if ref_adjust is not None:
        adjust_m = GENOMETOOL(ref_adjust)
        genotype.iloc[:,:2] = adjust_m.refalt_adjust(genotype.iloc[:,:2])
        genotype.loc[adjust_m.exchange_loc,genotype.columns[2:]] = 2 - genotype.loc[adjust_m.exchange_loc,genotype.columns[2:]]
        genotype.columns = ['REF','ALT']+genotype.columns[2:].to_list()
    return genotype

def hmpreader(hmp:str):
    geno = pd.read_csv(hmp,sep='\t')
    print(geno)
    return

