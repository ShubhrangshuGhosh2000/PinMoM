import sys, os
from pathlib import Path
path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)
import torch
import joblib

from utils.PreprocessUtils import loadPSSM
from utils import PPIPUtils

def PSSM(fastas, directory, fix_mut_prot_id=None, processPSSM=True,deviceType='cpu',psiblast_exec_path='./',blosumMatrix=None):
    pssm_dict = {}
    #calculate the sum of all PSSM data
    for ind in range(len(fastas)):
        item = fastas[ind]
        name = item[0]
        if(fix_mut_prot_id is not None):
            name = fix_mut_prot_id + '_' + item[0]
        seq = item[1]
        # print(f'@@@@@@@@@@@ :: name: {name}')
        # seq = 'ILWMAVARDNHPDCYSLHYNSCQHDYCLIMKKHLIVYNLGLKFYHCNRGEKPTTLNENKPWRQCCFCAACSVCQPNEN'
        data = loadPSSM(name, seq, directory+'PSSM/',usePSIBlast=processPSSM
                        ,psiblast_exec_path=psiblast_exec_path, blosumMatrix=blosumMatrix)
        pssm_dict[item[0]] = {'pssm_val': torch.tensor(data)}

    return pssm_dict
