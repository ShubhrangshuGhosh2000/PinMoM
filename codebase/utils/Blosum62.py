import sys, os
import torch

from pathlib import Path
path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))

from utils.PreprocessUtils import calcBlosum62

def Blosum62(fastas, directory, blosumMatrix=None):
    # ######### MUST RUN THE FOLLOWING IN THE SAME SHELL WHERE THE PROGRAM WILL RUN TO SET THE makeblastdb AND psiblast PATH (see genBlosum62() method of PreprocessUtils.py)
    # export PATH=$PATH:/specify/path/to/ncbi-blast-2.13.0+/bin
    # echo $PATH
    blosum62_dict = {}
    for ind in range(len(fastas)):
        item = fastas[ind]
        name = item[0]
        seq = item[1]
        data = calcBlosum62(name, seq, blosumMatrix=blosumMatrix)
        blosum62_dict[name] = {# 'seq': seq, 
                           'blosum62_val': torch.tensor(data)}
    return blosum62_dict
