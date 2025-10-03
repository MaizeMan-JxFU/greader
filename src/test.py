from greader import hmpreader
from greader.base import genotype2vcf

if __name__ == "__main__":
    genotype = hmpreader(r'''C:\Users\82788\Desktop\Pyscript\greader\test\genotype.hmp''',4)
    genotype2vcf(genotype,'test')
    pass