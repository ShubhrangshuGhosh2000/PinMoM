# this is the main module for manual feature extraction/engineering
import sys
from pathlib import Path

path_root = Path(__file__).parents[1]  
sys.path.insert(0, str(path_root))

from utils.AutoCovariance import AutoCovariance
from utils.ConjointTriad import ConjointTriad
from utils.LDCTD import LDCTD
from utils.PSEAAC import PSEAAC

from utils.SkipGram import SkipGram
from utils.LabelEncoding import LabelEncoding
from utils.PSSM import PSSM
from utils.Blosum62 import Blosum62


def extract_prot_seq_1D_manual_feat(fastas=None, deviceType='cpu'):
    man_1d_feat_dict = {}

    ac = AutoCovariance(fastas, lag=30, deviceType=deviceType)    
    man_1d_feat_dict['AC30'] = ac[1][1:]
    
    paac = PSEAAC(fastas,lag=15)
    man_1d_feat_dict['PSAAC15'] = paac[1][1:]
    
    ct = ConjointTriad(fastas,deviceType=deviceType)
    man_1d_feat_dict['ConjointTriad'] = ct[1][1:]
    
    (comp, tran, dist) = LDCTD(fastas)
    man_1d_feat_dict['LD10_CTD_ConjointTriad_C'] = comp[1][1:]
    man_1d_feat_dict['LD10_CTD_ConjointTriad_T'] = tran[1][1:]
    man_1d_feat_dict['LD10_CTD_ConjointTriad_D'] = dist[1][1:]
    
    return man_1d_feat_dict


def extract_prot_seq_2D_manual_feat(fix_mut_prot_id=None, folderName='./', fastas=None, skipgrm_lookup_prsnt=False
                                    , use_psiblast_for_pssm=False, psiblast_exec_path='./', labelEncode_lookup_prsnt=False, blosumMatrix=None):
    man_2d_feat_dict = {}
    
    skipgram_feat_dict = SkipGram(fastas,folderName+'SkipGramAA7H5.encode',hiddenSize=5, preTrained=True, deviceType='cpu',fullGPU=False
                                    , skipgrm_lookup_prsnt=skipgrm_lookup_prsnt)
    man_2d_feat_dict['SkipGramAA7'] = skipgram_feat_dict

    labelEnc_feat_dict = LabelEncoding(fastas,folderName+'LabelEncoding.encode'
                                       , labelEncode_lookup_prsnt=labelEncode_lookup_prsnt)
    man_2d_feat_dict['LabelEncoding'] = labelEnc_feat_dict

    pssm_dict = PSSM(fastas,folderName, fix_mut_prot_id=fix_mut_prot_id, processPSSM=use_psiblast_for_pssm,deviceType='cpu',psiblast_exec_path=psiblast_exec_path,blosumMatrix=blosumMatrix)
    man_2d_feat_dict['PSSM'] = pssm_dict

    blosum62_dict = Blosum62(fastas,folderName,blosumMatrix=blosumMatrix)
    man_2d_feat_dict['Blosum62'] = blosum62_dict

    return man_2d_feat_dict


# if __name__ == '__main__':
#     root_path = os.path.join('/project/root/directory/path/here')

#     prot_seq = 'MIFTPFLPPADLSVFQNVKGLQNDPE'
#     feature_type_lst = ['AC30', 'PSAAC15', 'ConjointTriad', 'LD10_CTD']
#     man_1d_feat_dict = extract_prot_seq_1D_manual_feat(root_path, prot_seq = prot_seq, feature_type_lst = feature_type_lst, deviceType='cpu')

