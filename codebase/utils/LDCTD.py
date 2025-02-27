import sys
from pathlib import Path

path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils.CTD_Composition import CTD_Composition
from utils.CTD_Distribution import CTD_Distribution
from utils.CTD_Transition import CTD_Transition
from utils.PreprocessUtils import LDEncode10, STDecode


def LDCTD(fastas, encodeFunc=LDEncode10, groupings = ['AGV','ILFP','YMTS','HNQW','RK','DE','C'], deviceType='cpu'):
    if encodeFunc is not None:
        encoded, encodedSize = encodeFunc(fastas)
        
    comp = CTD_Composition(encoded,groupings,deviceType)
    tran = CTD_Transition(encoded,groupings,deviceType)
    dist = CTD_Distribution(encoded,groupings,deviceType)
    
    if encodeFunc is not None:
        comp = STDecode(comp,encodedSize)
        tran = STDecode(tran,encodedSize)
        dist = STDecode(dist,encodedSize)
        
    return (comp, tran, dist)
