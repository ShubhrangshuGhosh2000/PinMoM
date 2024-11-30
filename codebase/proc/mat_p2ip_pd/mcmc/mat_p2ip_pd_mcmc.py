import os, sys

from pathlib import Path
path_root = Path(__file__).parents[3]  
sys.path.insert(0, str(path_root))


from utils import dl_reproducible_result_util
import copy
import math
import multiprocessing
import numpy as np
import pandas as pd
import random
import time
import torch

from proc.mat_p2ip_DS.mat_p2ip_origMan_auxTlOtherMan import matpip_RunTests
from utils import prot_design_util, PPIPUtils
from utils import preproc_plm_util, PreprocessUtils
from utils import feat_engg_manual_main_pd
from dscript.commands import predict
from dscript import pretrained
from dscript import alphabets


def run_prot_design_simuln(**kwargs):
    print('inside run_prot_design_simuln() method - Start')
    for arg_name, arg_value in kwargs.items():
        if((arg_value is None) and (arg_name not in ['resource_monitor'])):
            raise ValueError(f"Argument '{arg_name}' must be provided with a value.")
    root_path = kwargs.get('root_path'); dim_prot_complx_nm = kwargs.get('dim_prot_complx_nm'); use_prot_unit=kwargs.get('use_prot_unit'); num_of_itr = kwargs.get('num_of_itr')
    percent_len_for_calc_mut_pts = kwargs.get('percent_len_for_calc_mut_pts'); exec_mode_type = kwargs.get('exec_mode_type'); plm_file_location = kwargs.get('plm_file_location')
    plm_name = kwargs.get('plm_name'); batch_size = kwargs.get('batch_size'); batch_dump_interval = kwargs.get('batch_dump_interval')
    result_dump_dir = kwargs.get('result_dump_dir'); cuda_index = kwargs.get('cuda_index')
    use_psiblast_for_pssm = kwargs.get('use_psiblast_for_pssm'); psiblast_exec_path = kwargs.get('psiblast_exec_path')
    pdb_file_location = kwargs.get('pdb_file_location')
    fixed_temp_mcmc = kwargs.get('fixed_temp_mcmc')
    resourceMonitor_inst = kwargs.get('resource_monitor')
    max_thrshld_for_num_of_mut_pts = kwargs.get('max_thrshld_for_num_of_mut_pts')
    aa_lst = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    num_cpu_cores = multiprocessing.cpu_count()
    percent_len_for_calc_mut_pts_orig = percent_len_for_calc_mut_pts
    chain_sequences_dict = prot_design_util.extract_chain_sequences(dim_prot_complx_nm, pdb_file_location)
    pdb_chain_nm_lst = list(chain_sequences_dict.keys())
    unique_chain_name_tuples_lst = prot_design_util.gen_unique_tuples_from_keys(chain_sequences_dict)
    chain_nm_pu_dict = {}
    device = torch.device(f'cuda:{cuda_index}')
    protTrans_model, tokenizer = preproc_plm_util.load_protTrans_model(protTrans_model_path=plm_file_location
                        , protTrans_model_name=plm_name, device=device)
    model_type = 'matds_hybrid' if(exec_mode_type == 'thorough') else 'matpip'
    matpip_model = None
    matpip_model = matpip_RunTests.load_matpip_model(root_path)
    dscript_model = None
    if(model_type == 'matds_hybrid'):
        dscript_model = matpip_RunTests.load_dscript_model(root_path)
        pretrained_bb_model = pretrained.get_pretrained("lm_v1")
        pretrained_bb_model = pretrained_bb_model.cuda()
        pretrained_bb_model.eval()
        alphabet = alphabets.Uniprot21()
    for unique_chain_name_tuple_idx in range(len(unique_chain_name_tuples_lst)):
        unique_chain_name_tuple = unique_chain_name_tuples_lst[unique_chain_name_tuple_idx]
        fixed_prot_id, mut_prot_id = unique_chain_name_tuple
        fix_mut_prot_id_tag = f' complex_{dim_prot_complx_nm}_fixed_prot_id_{fixed_prot_id}_mut_prot_id_{mut_prot_id}:: '
        fix_mut_prot_id = f'cmplx_{dim_prot_complx_nm}_fixed_{fixed_prot_id}_mut_{mut_prot_id}'
        percent_len_for_calc_mut_pts = percent_len_for_calc_mut_pts_orig
        if(num_of_itr % batch_size != 0):
            num_of_itr = batch_size * int(num_of_itr / batch_size)
        num_of_batches = int(num_of_itr / batch_size)
        iter_spec_result_dump_dir = os.path.join(result_dump_dir, f'complex_{dim_prot_complx_nm}/fixed_{fixed_prot_id}_mut_{mut_prot_id}'
                                                 , f'res_totItr{num_of_itr}_batchSz{batch_size}_percLnForMutPts{percent_len_for_calc_mut_pts}_pu{use_prot_unit}')
        if os.path.exists(iter_spec_result_dump_dir):
            continue  
        PPIPUtils.createFolder(iter_spec_result_dump_dir)
        t_entire_init = time.time()
        crit_lst, exec_time_lst = [], []
        misc_info_lst, misc_info_val_lst = [], []
        max_cpu_cores_used = 1; max_gpu_cores_used = 1
        fixed_prot_seq = chain_sequences_dict[fixed_prot_id]
        fixed_prot_seq = fixed_prot_seq.replace(" ", "").upper()
        mut_prot_seq = chain_sequences_dict[mut_prot_id]
        mut_prot_seq = mut_prot_seq.replace(" ", "").upper()
        mut_prot_len = len(mut_prot_seq)
        fixed_plm_1d_embedd_tensor_lst = preproc_plm_util.extract_protTrans_plm_feat(prot_seq_lst=[fixed_prot_seq], model=protTrans_model
                                                                                    , tokenizer=tokenizer, device=device)
        featureDir = os.path.join(root_path, 'dataset/preproc_data/derived_feat/')
        fixed_prot_fastas = [(fixed_prot_id, fixed_prot_seq)]  
        blosumMatrix = PreprocessUtils.loadBlosum62()
        PPIPUtils.createFolder(os.path.join(featureDir, 'PSSM'))
        man_2d_feat_dict_prot_fixed = feat_engg_manual_main_pd.extract_prot_seq_2D_manual_feat(fix_mut_prot_id=fix_mut_prot_id, folderName=featureDir, fastas=fixed_prot_fastas
                                                                , skipgrm_lookup_prsnt = False, use_psiblast_for_pssm=use_psiblast_for_pssm, psiblast_exec_path=psiblast_exec_path
                                                                , labelEncode_lookup_prsnt = False, blosumMatrix=blosumMatrix)
        man_1d_feat_dict_prot_fixed = feat_engg_manual_main_pd.extract_prot_seq_1D_manual_feat(fastas=fixed_prot_fastas)
        batch_out_dict_lst = []
        t1 = time.time()
        mut_idx = 0
        prv_best_mutated_prot_seq = mut_prot_seq
        p_stable = -1.0
        t_mcmc = fixed_temp_mcmc
        early_stop = False; early_stop_crit = None
        trigger_fixed_mut_num_mna = False;  enable_early_stopping_check = True
        resourceMonitor_inst_dcopy = None
        if(resourceMonitor_inst is not None):
            resourceMonitor_inst_dcopy = copy.deepcopy(resourceMonitor_inst)
        t_batch_init = time.time()
        for batch_idx in range(num_of_batches):
            t1 = time.time()
            ppi_score_arr = np.empty((0,), dtype=float)
            mut_prot_fastas, mut_pos_lst, aa_idx_lst,  = [], [], []
            mod_batch_size = batch_size
            if(batch_idx == 0):
                mut_prot_fastas.append((mut_prot_id, mut_prot_seq))
                mod_batch_size = batch_size -1
                mut_idx += 1
                mut_pos_lst.append([-1])  
                aa_idx_lst.append([-1])  
            for itr in range(mod_batch_size):
                prv_best_mutated_prot_seq_as_lst = list(prv_best_mutated_prot_seq)
                mut_seq_range = range(mut_prot_len)
                if(percent_len_for_calc_mut_pts < 0):
                    num_of_mut_pts = -1 * percent_len_for_calc_mut_pts
                else:
                    x_percent_of_mut_prot_len = int(percent_len_for_calc_mut_pts / 100.0 * mut_prot_len)
                    if(x_percent_of_mut_prot_len < 1):
                        x_percent_of_mut_prot_len = 1
                    x_percent_of_mut_prot_len = min(x_percent_of_mut_prot_len, max_thrshld_for_num_of_mut_pts)
                    num_of_mut_pts = random.sample(range(1, (x_percent_of_mut_prot_len+1)), 1)[0]
                print(f'{fix_mut_prot_id_tag} num_of_mut_pts: {num_of_mut_pts}')
                sel_mut_seq_pos_lst = random.sample(mut_seq_range, num_of_mut_pts)  
                sel_aa_lst_indices = []
                for sel_pos in sel_mut_seq_pos_lst:
                    prv_aa_idx_at_sel_pos = aa_lst.index(prv_best_mutated_prot_seq_as_lst[sel_pos])
                    mod_aa_idx_lst = [aa_idx for aa_idx in range(len(aa_lst)) if aa_idx not in [prv_aa_idx_at_sel_pos]]
                    new_aa_idx_at_sel_pos = random.sample(mod_aa_idx_lst, 1)[0]
                    sel_aa_lst_indices.append(new_aa_idx_at_sel_pos)
                for i in range(num_of_mut_pts):
                    prv_best_mutated_prot_seq_as_lst[sel_mut_seq_pos_lst[i]] = aa_lst[sel_aa_lst_indices[i]]
                mutated_prot_seq = "".join(prv_best_mutated_prot_seq_as_lst)
                mut_prot_fastas.append((f'm{mut_idx}', mutated_prot_seq))
                mut_pos_lst.append(sel_mut_seq_pos_lst)
                aa_idx_lst.append(sel_aa_lst_indices)
                mut_idx += 1
            t2 = time.time()
            mut_prot_id_lst, mut_prot_seq_lst = [], []
            for mutated_prot_id, mutated_prot_seq in mut_prot_fastas:
                mut_prot_id_lst.append(mutated_prot_id)
                mut_prot_seq_lst.append(mutated_prot_seq)
            mut_plm_1d_embedd_tensor_lst = preproc_plm_util.extract_protTrans_plm_feat(prot_seq_lst=mut_prot_seq_lst, model=protTrans_model
                                                                                    , tokenizer=tokenizer, device=device)
            plm_feat_dict = dict(zip([fixed_prot_id] + mut_prot_id_lst, fixed_plm_1d_embedd_tensor_lst + mut_plm_1d_embedd_tensor_lst))
            t3 = time.time()
            man_2d_feat_dict_prot = man_1d_feat_dict_prot = None
            if(batch_size == 1):
                man_2d_feat_dict_prot, man_1d_feat_dict_prot = extract_manual_feat_serially(fix_mut_prot_id, featureDir, mut_prot_fastas, use_psiblast_for_pssm, psiblast_exec_path, blosumMatrix, mut_prot_id_lst)
            else:
                num_process = 0
                if((batch_size > 1) and (batch_size < num_cpu_cores)):
                    num_process = batch_size
                else:
                    num_process = num_cpu_cores - 2
                if(num_process > max_cpu_cores_used):
                    max_cpu_cores_used = num_process
                man_2d_feat_dict_prot, man_1d_feat_dict_prot = extract_manual_feat_mp(fix_mut_prot_id, featureDir, mut_prot_fastas, use_psiblast_for_pssm, psiblast_exec_path, blosumMatrix, mut_prot_id_lst, num_process)
            man_2d_feat_dict_prot[fixed_prot_id] = man_2d_feat_dict_prot_fixed
            man_1d_feat_dict_prot[fixed_prot_id] = man_1d_feat_dict_prot_fixed
            t4 = time.time()
            mcp = matpip_RunTests.execute(root_path, matpip_model, featureDir, fixed_prot_id, mut_prot_id_lst
                                                , plm_feat_dict, man_2d_feat_dict_prot, man_1d_feat_dict_prot)
            t5 = time.time()
            if(model_type == 'matpip'):
                ppi_score_arr = np.hstack((ppi_score_arr, mcp))
                t7 = time.time()
            else:
                dscript_seqs = mut_prot_fastas + fixed_prot_fastas
                dscript_pairs = [(fixed_prot_id, mut_prot_id) for mut_prot_id in mut_prot_id_lst]
                threshold = 0.5
                dcp = predict.main_pd(pairs=dscript_pairs, model=dscript_model, pre_trn_mdl=pretrained_bb_model, alphabet=alphabet
                                                        , seqs=dscript_seqs, threshold=threshold)
                t6 = time.time()
                hcpa = calc_matds_hybrid_score(mcp, dcp)
                t7 = time.time()
                ppi_score_arr = np.hstack((ppi_score_arr, hcpa))
            p_stable_changed = False
            min_val_index = np.argmin(ppi_score_arr)
            min_value = ppi_score_arr[min_val_index]
            p_i = min_value
            if(p_i > p_stable):
                p_stable = p_i
                p_stable_changed = True
            else:
                rand = np.random.rand()
                del_p = -(p_i - p_stable)  
                if(rand < math.exp(-(del_p / t_mcmc))):  
                    p_stable = p_i
                    p_stable_changed = True
                else:
                    p_stable_changed = False
            t8 = time.time()
            if(p_stable_changed):
                batch_out_dict = {'batch_idx': batch_idx}
                batch_out_dict['simuln_idx'] = (batch_size * batch_idx) + min_val_index
                batch_out_dict['mut_pos_lst'] = mut_pos_lst[min_val_index]
                batch_out_dict['aa_idx_lst'] = aa_idx_lst[min_val_index]
                batch_out_dict['ppi_score'] = p_i
                batch_out_dict['prot_seq'] = mut_prot_fastas[min_val_index][1]
                batch_out_dict_lst.append(batch_out_dict)
                prv_best_mutated_prot_seq = mut_prot_fastas[min_val_index][1]
            crit_lst.append(f'batch_idx_{batch_idx}'); exec_time_lst.append(round(t8-t1, 3))
            if(trigger_fixed_mut_num_mna):
                prv_best_mutated_prot_seq = mut_prot_seq
                p_stable = -1.0
            if(enable_early_stopping_check and early_stop):
                break
            if((batch_idx + 1) % batch_dump_interval == 0):
                if(len(batch_out_dict_lst) > 0):  
                    t9_1 = time.time()
                    batch_out_df = pd.DataFrame(batch_out_dict_lst)
                    batch_out_csv_nm = os.path.join(iter_spec_result_dump_dir, f'batchIdx_{batch_idx + 1 - batch_dump_interval}_{batch_idx}_batchSize_{batch_size}.csv')
                    batch_out_df.to_csv(batch_out_csv_nm, index=False)
                    batch_out_dict_lst = []
                    t9_2 = time.time()
                    if(resourceMonitor_inst_dcopy != None):  
                        t10_1 = time.time()
                        resourceMonitor_inst_dcopy.monitor_peak_ram_usage()
                        resourceMonitor_inst_dcopy.monitor_peak_gpu_memory_usage()
                        t10_2 = time.time()
        if(len(batch_out_dict_lst) > 0):
            batch_out_df = pd.DataFrame(batch_out_dict_lst)
            batch_out_csv_nm = os.path.join(iter_spec_result_dump_dir, f'batchIdx_{batch_dump_interval * (batch_idx // batch_dump_interval)}_{batch_idx}_batchSize_{batch_size}.csv')
            batch_out_df.to_csv(batch_out_csv_nm, index=False)
        t_final = time.time()
        crit_lst.append('batch_total_time'); exec_time_lst.append(round(t_final - t_batch_init, 3))
        crit_lst.append('simuln_total_time'); exec_time_lst.append(round(t_final - t_entire_init, 3))
        time_df = pd.DataFrame({'criterion': crit_lst, 'time_in_sec': exec_time_lst})
        time_df.to_csv(os.path.join(iter_spec_result_dump_dir, 'time_records.csv'), index=False)
        misc_info_lst.append('tot_num_batches_executed'); misc_info_val_lst.append(batch_idx + 1)
        misc_info_lst.append('batch_size'); misc_info_val_lst.append(batch_size)
        misc_info_lst.append('batch_total_time_in_sec'); misc_info_val_lst.append(round(t_final - t_batch_init, 3))
        misc_info_lst.append('tot_num_itr_executed'); misc_info_val_lst.append((batch_idx + 1) * batch_size)
        misc_info_lst.append('max_cpu_cores_used'); misc_info_val_lst.append(max_cpu_cores_used)
        misc_info_lst.append('max_gpu_cores_used'); misc_info_val_lst.append(max_gpu_cores_used)
        misc_info_lst.append('peak_ram_usage_in_GB'); misc_info_val_lst.append(round(resourceMonitor_inst_dcopy.peak_ram_usage / (1024 * 1024 * 1024.0), 3))
        misc_info_lst.append('peak_gpu_usage_in_GB'); misc_info_val_lst.append(round(resourceMonitor_inst_dcopy.peak_gpu_usage / 1024.0, 3))
        misc_info_df = pd.DataFrame({'misc_info': misc_info_lst, 'misc_info_val': misc_info_val_lst})
        misc_info_df.to_csv(os.path.join(iter_spec_result_dump_dir, 'misc_info.csv'), index=False)
    print('inside run_prot_design_simuln() method - End')


def extract_manual_feat_serially(fix_mut_prot_id, featureDir, mut_prot_fastas, use_psiblast_for_pssm, psiblast_exec_path, blosumMatrix, mut_prot_id_lst):
    man_2d_feat_dict_prot = dict()
    man_1d_feat_dict_prot = dict()
    skipgrm_lookup_prsnt = labelEncode_lookup_prsnt = True
    for i in range(len(mut_prot_id_lst)):
        indiv_mut_prot_fastas = [mut_prot_fastas[i]]
        man_2d_feat_dict_prot_mut = feat_engg_manual_main_pd.extract_prot_seq_2D_manual_feat(fix_mut_prot_id=fix_mut_prot_id, folderName=featureDir, fastas=indiv_mut_prot_fastas
                                                            , skipgrm_lookup_prsnt=skipgrm_lookup_prsnt, use_psiblast_for_pssm=use_psiblast_for_pssm, psiblast_exec_path=psiblast_exec_path
                                                            , labelEncode_lookup_prsnt=labelEncode_lookup_prsnt, blosumMatrix=blosumMatrix)
        man_1d_feat_dict_prot_mut = feat_engg_manual_main_pd.extract_prot_seq_1D_manual_feat(fastas=indiv_mut_prot_fastas)
        man_2d_feat_dict_prot[indiv_mut_prot_fastas[0][0]] = man_2d_feat_dict_prot_mut
        man_1d_feat_dict_prot[indiv_mut_prot_fastas[0][0]] = man_1d_feat_dict_prot_mut
    return (man_2d_feat_dict_prot, man_1d_feat_dict_prot)


def extract_manual_feat_mp(fix_mut_prot_id, featureDir, mut_prot_fastas, use_psiblast_for_pssm, psiblast_exec_path, blosumMatrix, mut_prot_id_lst, num_process):
    manager = multiprocessing.Manager()
    man_2d_feat_dict_prot = manager.dict()  
    man_1d_feat_dict_prot = manager.dict()  
    shared_data_lock = manager.Lock()
    skipgrm_lookup_prsnt = labelEncode_lookup_prsnt = True
    with multiprocessing.Pool(processes=num_process) as pool:
        pool.starmap(ext_man_feat
                     , [(idx, fix_mut_prot_id, featureDir, mut_prot_fastas, skipgrm_lookup_prsnt, use_psiblast_for_pssm, psiblast_exec_path, labelEncode_lookup_prsnt, blosumMatrix  
                         , man_2d_feat_dict_prot, man_1d_feat_dict_prot, shared_data_lock  
                         ) for idx in range(len(mut_prot_id_lst))])
    return (man_2d_feat_dict_prot, man_1d_feat_dict_prot)


def ext_man_feat(idx, fix_mut_prot_id, featureDir, mut_prot_fastas, skipgrm_lookup_prsnt, use_psiblast_for_pssm, psiblast_exec_path, labelEncode_lookup_prsnt, blosumMatrix  
                             , man_2d_feat_dict_prot, man_1d_feat_dict_prot, shared_data_lock):  
    indiv_mut_prot_fastas = [mut_prot_fastas[idx]]
    man_2d_feat_dict_prot_mut = feat_engg_manual_main_pd.extract_prot_seq_2D_manual_feat(fix_mut_prot_id=fix_mut_prot_id, folderName=featureDir, fastas=indiv_mut_prot_fastas
                                                             , skipgrm_lookup_prsnt=skipgrm_lookup_prsnt
                                                             , use_psiblast_for_pssm=use_psiblast_for_pssm, psiblast_exec_path=psiblast_exec_path
                                                             , labelEncode_lookup_prsnt=labelEncode_lookup_prsnt, blosumMatrix=blosumMatrix)
    man_1d_feat_dict_prot_mut = feat_engg_manual_main_pd.extract_prot_seq_1D_manual_feat(fastas=indiv_mut_prot_fastas)
    with shared_data_lock:
        man_2d_feat_dict_prot[indiv_mut_prot_fastas[0][0]] = man_2d_feat_dict_prot_mut
        man_1d_feat_dict_prot[indiv_mut_prot_fastas[0][0]] = man_1d_feat_dict_prot_mut
    

def calc_matds_hybrid_score(mcp, dcp):
    mccp = 1 - mcp
    dccp = 1 - dcp
    hcpa = mcp
    hcpa = np.where(
        (mcp > 0.5) & (dccp > 0.5),  
        mcp - (dccp - 0.5),  
        np.where(  
            (mccp > 0.5) & (dcp > 0.5),  
            dcp - (mccp - 0.5),  
            hcpa  
        )
    )
    return hcpa
