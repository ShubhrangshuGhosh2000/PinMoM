import sys, os
from pathlib import Path
path_root = Path(__file__).parents[3]  
sys.path.insert(0, str(path_root))
from utils import dl_reproducible_result_util
from proc.mat_p2ip_DS.mat_p2ip_origMan_auxTlOtherMan.mat_MatP2ipNetwork_origMan_auxTlOtherMan_DS_test import MatP2ipNetworkModule
from utils.ProjectDataLoader import *
from proc.mat_p2ip_DS.mat_p2ip_origMan_auxTlOtherMan import mat_RunTrainTest_origMan_auxTlOtherMan_DS
import torch

def execute(root_path, model, featureDir, fixed_prot_id, mut_prot_id_lst, plm_feat_dict, man_2d_feat_dict_prot, man_1d_feat_dict_prot):
    trainSets, testSets, saves, pfs, featureFolder = loadTestData_matpip(featureDir, fixed_prot_id, mut_prot_id_lst)
    outResultsName = None
    feature_data = (plm_feat_dict, man_2d_feat_dict_prot, man_1d_feat_dict_prot)
    mcp = mat_RunTrainTest_origMan_auxTlOtherMan_DS.runTest_matpip(model,outResultsName,trainSets,testSets,featureFolder
                                                             ,loads=None,fixed_prot_id=fixed_prot_id, feature_data=feature_data)
    return mcp


def load_matpip_model(root_path):
    hyp = {'fullGPU':True,'deviceType':'cuda'} 
    hyp['maxProteinLength'] = 800  
    hyp['hiddenSize'] = 70  
    hyp['numLayers'] = 4  
    hyp['n_heads'] = 1  
    hyp['layer_1_size'] = 1024  
    hyp['batchSize'] = 256  
    hyp['numEpochs'] = 10  
    hyp['inSize'] = 46
    hyp['aux_oneDencodingsize'] = 2242
    model = MatP2ipNetworkModule(hyperParams=hyp)
    model.batchIdx = 0
    human_full_model_loc = os.path.join(root_path, 'dataset/proc_data/mat_res_origMan_auxTlOtherMan_human/DS_human_full.out')
    model.loadModelFromFile(human_full_model_loc)
    return model


def load_dscript_model(root_path):
    modelPath = os.path.join(root_path, 'codebase/dscript/human_v1.sav')  
    try:
        model = torch.load(modelPath).cuda()
        model.use_cuda = True
    except FileNotFoundError:
        print(f"D-Script Model {modelPath} not found")
        sys.exit(1)
    return model
