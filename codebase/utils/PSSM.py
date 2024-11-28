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
    # For PSSM feature extraction
    # =============================
    # ######### MUST RUN IN THE SAME SHELL WHERE THE PROGRAM WILL RUN TO SET THE makeblastdb AND psiblast PATH 
    # export PATH=$PATH:/scratch/pralaycs/Shubh_Working_Remote/ncbi-blast-2.13.0+/bin
    # echo $PATH

    # In the path "dataset/preproc_data/derived_feat/PSSM/" put the following files so that database creation
    # using 'makeblastdb' command will not be required:
    # uniprotSprotFull.pdb, uniprotSprotFull.phr, uniprotSprotFull.pin, uniprotSprotFull.pjs,
    # uniprotSprotFull.pot, uniprotSprotFull.psq, uniprotSprotFull.ptf, uniprotSprotFull.pto

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
        # print('\n\n################### Completed ' + str(ind+1) + ' out of ' + str(len(fastas)) + '\n\n')

    return pssm_dict
