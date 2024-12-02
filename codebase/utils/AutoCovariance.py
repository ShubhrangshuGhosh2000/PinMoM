import sys
from pathlib import Path
path_root = Path(__file__).parents[1]  
sys.path.insert(0, str(path_root))

from utils.Covariance import Covariance

def AutoCovariance(fastas, aaIDs = ['GUO_H1','HOPT810101','KRIW790103','GRAR740102','CHAM820101','ROSG850103_GUO_SASA','GUO_NCI'], lag=30, deviceType='cpu'):
    return Covariance(fastas,aaIDs,lag,separate=False,calcType='AutoCovariance',deviceType=deviceType)