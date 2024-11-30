import os, sys

from pathlib import Path
path_root = Path(__file__).parents[3]  
sys.path.insert(0, str(path_root))


from utils import dl_reproducible_result_util
import copy
import gc
import math
import multiprocessing
import numpy as np
import pandas as pd
import random
import shutil
import time
import torch

from proc.mat_p2ip_DS.mat_p2ip_origMan_auxTlOtherMan import matpip_RunTests
from utils import prot_design_util, PPIPUtils, interface_residue
from utils import preproc_plm_util, PreprocessUtils
from utils import feat_engg_manual_main_pd
from dscript.commands import predict
from dscript import pretrained
from dscript import alphabets
import torch.multiprocessing as mp


def run_prot_design_simuln(**kwargs):
    for arg_name, arg_value in kwargs.items():
        if((arg_value is None) and (arg_name not in ['resource_monitor'])):
            raise ValueError(f"Argument '{arg_name}' must be provided with a value.")
    root_path = kwargs.get('root_path'); dim_prot_complx_nm = kwargs.get('dim_prot_complx_nm'); use_prot_unit=kwargs.get('use_prot_unit'); num_of_itr = kwargs.get('num_of_itr')
    percent_len_for_calc_mut_pts = kwargs.get('percent_len_for_calc_mut_pts'); exec_mode_type = kwargs.get('exec_mode_type'); plm_file_location = kwargs.get('plm_file_location')
    plm_name = kwargs.get('plm_name'); local_step_size = kwargs.get('local_step_size')
    result_dump_dir = kwargs.get('result_dump_dir'); cuda_index = kwargs.get('cuda_index')
    use_psiblast_for_pssm = kwargs.get('use_psiblast_for_pssm'); psiblast_exec_path = kwargs.get('psiblast_exec_path')
    pdb_file_location = kwargs.get('pdb_file_location')
    mut_only_at_intrfc_resid_idx = kwargs.get('mut_only_at_intrfc_resid_idx'); naccess_path = kwargs.get('naccess_path')
    resourceMonitor_inst = kwargs.get('resource_monitor')
    max_thrshld_for_num_of_mut_pts = kwargs.get('max_thrshld_for_num_of_mut_pts')
    temp_min_remc = kwargs.get('temp_min_remc'); temp_max_remc = kwargs.get('temp_max_remc'); num_of_replica_remc = kwargs.get('num_of_replica_remc')
    result_save_interval_remc = kwargs.get('result_save_interval_remc')
    aa_lst = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    num_cpu_cores = multiprocessing.cpu_count()
    percent_len_for_calc_mut_pts_orig = percent_len_for_calc_mut_pts
    local_step_size = 1
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
        iter_spec_result_dump_dir = os.path.join(result_dump_dir, f'complex_{dim_prot_complx_nm}/fixed_{fixed_prot_id}_mut_{mut_prot_id}', f'res_totItr{num_of_itr}_batchSz{local_step_size}_percLnForMutPts{percent_len_for_calc_mut_pts}_pu{use_prot_unit}_mutIntrfc{mut_only_at_intrfc_resid_idx}')
        if os.path.exists(iter_spec_result_dump_dir):
            continue  
        PPIPUtils.createFolder(iter_spec_result_dump_dir)
        fixed_prot_seq = chain_sequences_dict[fixed_prot_id]
        fixed_prot_seq = fixed_prot_seq.replace(" ", "").upper()
        mut_prot_seq = chain_sequences_dict[mut_prot_id]
        mut_prot_seq = mut_prot_seq.replace(" ", "").upper()
        mut_prot_len = len(mut_prot_seq)
        fixed_plm_1d_embedd_tensor_lst = preproc_plm_util.extract_protTrans_plm_feat(prot_seq_lst=[fixed_prot_seq], model=protTrans_model, tokenizer=tokenizer, device=device)
        featureDir = os.path.join(root_path, 'dataset/preproc_data/derived_feat/')
        fixed_prot_fastas = [(fixed_prot_id, fixed_prot_seq)]  
        blosumMatrix = PreprocessUtils.loadBlosum62()
        PPIPUtils.createFolder(os.path.join(featureDir, 'PSSM'))
        man_2d_feat_dict_prot_fixed = feat_engg_manual_main_pd.extract_prot_seq_2D_manual_feat(fix_mut_prot_id=fix_mut_prot_id, folderName=featureDir, fastas=fixed_prot_fastas
                                                                , skipgrm_lookup_prsnt = False, use_psiblast_for_pssm=use_psiblast_for_pssm, psiblast_exec_path=psiblast_exec_path
                                                                , labelEncode_lookup_prsnt = False, blosumMatrix=blosumMatrix)
        man_1d_feat_dict_prot_fixed = feat_engg_manual_main_pd.extract_prot_seq_1D_manual_feat(fastas=fixed_prot_fastas)
        total_wrapper_steps = num_of_itr // local_step_size
        replica_temp_lst = [0.0] * num_of_replica_remc
        for repl_indx in range(num_of_replica_remc):
            replica_temp_lst[repl_indx] = temp_min_remc + (repl_indx * ((temp_max_remc -temp_min_remc) / (num_of_replica_remc -1)))
        replica_seq_lst = [''] * num_of_replica_remc
        for repl_indx in range(num_of_replica_remc):
            replica_seq_lst[repl_indx] = mut_prot_seq
        replica_prob_lst = [0.0] * num_of_replica_remc
        for repl_indx in range(num_of_replica_remc):
            replica_prob_lst[repl_indx] = 0.0
        remc_inp_arg_dict_lst = [{}] * num_of_replica_remc
        for repl_indx in range(num_of_replica_remc):
            indiv_replica_arg_dict = copy.deepcopy(kwargs)
            indiv_replica_arg_dict['fixed_prot_id'] = fixed_prot_id
            indiv_replica_arg_dict['mut_prot_id'] = mut_prot_id
            indiv_replica_arg_dict['mut_prot_len'] = mut_prot_len
            indiv_replica_arg_dict['chain_nm_pu_dict'] = chain_nm_pu_dict
            indiv_replica_arg_dict['aa_lst'] = aa_lst
            indiv_replica_arg_dict['fix_mut_prot_id'] = fix_mut_prot_id
            indiv_replica_arg_dict['fix_mut_prot_id_tag'] = fix_mut_prot_id_tag
            indiv_replica_arg_dict['protTrans_model'] = protTrans_model
            indiv_replica_arg_dict['tokenizer'] = tokenizer
            indiv_replica_arg_dict['device'] = device
            indiv_replica_arg_dict['num_cpu_cores'] = num_cpu_cores
            indiv_replica_arg_dict['fixed_plm_1d_embedd_tensor_lst'] = fixed_plm_1d_embedd_tensor_lst
            indiv_replica_arg_dict['featureDir'] = featureDir
            indiv_replica_arg_dict['blosumMatrix'] = blosumMatrix
            indiv_replica_arg_dict['man_2d_feat_dict_prot_fixed'] = man_2d_feat_dict_prot_fixed
            indiv_replica_arg_dict['man_1d_feat_dict_prot_fixed'] = man_1d_feat_dict_prot_fixed
            indiv_replica_arg_dict['matpip_model'] = matpip_model
            indiv_replica_arg_dict['model_type'] = model_type
            indiv_replica_arg_dict['fixed_prot_fastas'] = fixed_prot_fastas
            indiv_replica_arg_dict['dscript_model'] = dscript_model
            indiv_replica_arg_dict['alphabet'] = alphabet
            indiv_replica_arg_dict['pretrained_bb_model'] = pretrained_bb_model
            indiv_replica_arg_dict['iter_spec_result_dump_dir'] = iter_spec_result_dump_dir
            indiv_replica_arg_dict['replica_index_remc'] = repl_indx
            indiv_replica_arg_dict['replica_temp'] = replica_temp_lst[repl_indx]
            indiv_replica_arg_dict['replica_mut_seq'] = replica_seq_lst[repl_indx]
            indiv_replica_arg_dict['replica_prob'] = replica_prob_lst[repl_indx]
            remc_inp_arg_dict_lst[repl_indx] = indiv_replica_arg_dict
        batch_out_df_lst, misc_info_df_lst, time_df_lst = [], [], []
        for wrapper_step_idx in range(total_wrapper_steps):
            for indiv_replica_arg_dict in remc_inp_arg_dict_lst: 
                indiv_replica_arg_dict['wrapper_step_idx'] = wrapper_step_idx
            with mp.get_context('spawn').Pool(processes=num_of_replica_remc) as pool:
                result_dict_lst = pool.map(indiv_replica_work, remc_inp_arg_dict_lst)
            for res_idx, result_dict in enumerate(result_dict_lst):
                if 'error' in result_dict:
                    return
            ordered_result_dict_lst = [{}] * num_of_replica_remc  
            for result_dict in result_dict_lst:
                replica_idx = result_dict['replica_index_remc']
                ordered_result_dict_lst[replica_idx] = result_dict
            repl_prob_seq_tupl_lst = [(0.0, '')] * num_of_replica_remc
            for repl_indx in range(num_of_replica_remc):
                repl_prob_seq_tupl_lst[repl_indx] = (ordered_result_dict_lst[repl_indx]['replica_prob'], ordered_result_dict_lst[repl_indx]['replica_mut_seq'])
            indexed_lst = [(orig_idx, tup) for orig_idx, tup in enumerate(repl_prob_seq_tupl_lst)]
            sorted_indexed_lst = sorted(indexed_lst, key=lambda x: x[1][0])
            new_order = [original_index for original_index, _ in sorted_indexed_lst]
            changes = [(new_order[i], i) for i in range(len(sorted_indexed_lst)) if new_order[i] != i]
            sorted_repl_prob_seq_tupl_lst = [t for _, t in sorted_indexed_lst]
            for repl_indx in range(num_of_replica_remc):
                remc_inp_arg_dict_lst[repl_indx]['replica_prob'] = sorted_repl_prob_seq_tupl_lst[repl_indx][0]  
                remc_inp_arg_dict_lst[repl_indx]['replica_mut_seq'] = sorted_repl_prob_seq_tupl_lst[repl_indx][1]  
            for repl_indx in range(num_of_replica_remc):
                result_dict = ordered_result_dict_lst[repl_indx]
                if('batch_out_df' in result_dict):
                    batch_out_df_lst.append(result_dict['batch_out_df'])
                misc_info_df_lst.append(result_dict['misc_info_df'])
                time_df_lst.append(result_dict['time_df'])
            if((wrapper_step_idx + 1) % result_save_interval_remc == 0):
                if(len(batch_out_df_lst) > 0):  
                    t9_1 = time.time()
                    wrapper_out_df = pd.concat(batch_out_df_lst)
                    wrapper_out_csv_nm = os.path.join(iter_spec_result_dump_dir, f'wrapperStepIdx_{wrapper_step_idx + 1 - result_save_interval_remc}_{wrapper_step_idx}_localStepSize_{local_step_size}.csv')
                    wrapper_out_df.to_csv(wrapper_out_csv_nm, index=False)
                    batch_out_df_lst = []
                    t9_2 = time.time()
        con_misc_info_df = pd.concat(misc_info_df_lst)
        con_misc_info_csv_nm = os.path.join(iter_spec_result_dump_dir, f'misc_info.csv')
        con_misc_info_df.to_csv(con_misc_info_csv_nm, index=False)
        con_time_df = pd.concat(time_df_lst)
        con_time_csv_nm = os.path.join(iter_spec_result_dump_dir, f'time_records.csv')
        con_time_df.to_csv(con_time_csv_nm, index=False)



def indiv_replica_work(indiv_replica_arg_dict):
    root_path = indiv_replica_arg_dict.get('root_path'); dim_prot_complx_nm = indiv_replica_arg_dict.get('dim_prot_complx_nm'); use_prot_unit=indiv_replica_arg_dict.get('use_prot_unit'); num_of_itr = indiv_replica_arg_dict.get('num_of_itr')
    percent_len_for_calc_mut_pts = indiv_replica_arg_dict.get('percent_len_for_calc_mut_pts'); exec_mode_type = indiv_replica_arg_dict.get('exec_mode_type'); plm_file_location = indiv_replica_arg_dict.get('plm_file_location')
    plm_name = indiv_replica_arg_dict.get('plm_name'); local_step_size = indiv_replica_arg_dict.get('local_step_size'); local_step_size = indiv_replica_arg_dict.get('local_step_size')
    result_dump_dir = indiv_replica_arg_dict.get('result_dump_dir'); cuda_index = indiv_replica_arg_dict.get('cuda_index')
    use_psiblast_for_pssm = indiv_replica_arg_dict.get('use_psiblast_for_pssm'); psiblast_exec_path = indiv_replica_arg_dict.get('psiblast_exec_path')
    pdb_file_location = indiv_replica_arg_dict.get('pdb_file_location')
    resourceMonitor_inst = indiv_replica_arg_dict.get('resource_monitor')
    max_thrshld_for_num_of_mut_pts = indiv_replica_arg_dict.get('max_thrshld_for_num_of_mut_pts')
    local_step_size = indiv_replica_arg_dict['local_step_size']
    fixed_prot_id = indiv_replica_arg_dict['fixed_prot_id']
    mut_prot_id = indiv_replica_arg_dict['mut_prot_id']
    mut_prot_len = indiv_replica_arg_dict['mut_prot_len']
    aa_lst = indiv_replica_arg_dict['aa_lst']
    fix_mut_prot_id = indiv_replica_arg_dict['fix_mut_prot_id']
    fix_mut_prot_id_tag = indiv_replica_arg_dict['fix_mut_prot_id_tag']
    protTrans_model = indiv_replica_arg_dict['protTrans_model']
    tokenizer = indiv_replica_arg_dict['tokenizer']
    device = indiv_replica_arg_dict['device']
    num_cpu_cores = indiv_replica_arg_dict['num_cpu_cores']
    fixed_plm_1d_embedd_tensor_lst = indiv_replica_arg_dict['fixed_plm_1d_embedd_tensor_lst']
    featureDir = indiv_replica_arg_dict['featureDir']
    blosumMatrix = indiv_replica_arg_dict['blosumMatrix']
    man_2d_feat_dict_prot_fixed = indiv_replica_arg_dict['man_2d_feat_dict_prot_fixed']
    man_1d_feat_dict_prot_fixed = indiv_replica_arg_dict['man_1d_feat_dict_prot_fixed']
    matpip_model = indiv_replica_arg_dict['matpip_model']
    model_type = indiv_replica_arg_dict['model_type']
    fixed_prot_fastas = indiv_replica_arg_dict['fixed_prot_fastas']
    dscript_model = indiv_replica_arg_dict['dscript_model']
    alphabet = indiv_replica_arg_dict['alphabet']
    pretrained_bb_model = indiv_replica_arg_dict['pretrained_bb_model']
    replica_index_remc = indiv_replica_arg_dict['replica_index_remc']
    replica_temp = indiv_replica_arg_dict['replica_temp']
    replica_mut_seq = indiv_replica_arg_dict['replica_mut_seq']
    replica_prob = indiv_replica_arg_dict['replica_prob']
    wrapper_step_idx = indiv_replica_arg_dict['wrapper_step_idx']
    fix_mut_prot_id_tag = f' complex_{dim_prot_complx_nm}_fixed_prot_id_{fixed_prot_id}_mut_prot_id_{mut_prot_id}_replica_{replica_index_remc}_wrapper_step_{wrapper_step_idx}:: '
    try:
        out_replica_work_dict = {}
        out_replica_work_dict['replica_index_remc'] = replica_index_remc; out_replica_work_dict['replica_temp'] = replica_temp; out_replica_work_dict['replica_mut_seq'] = replica_mut_seq
        out_replica_work_dict['replica_prob'] = replica_prob; out_replica_work_dict['wrapper_step_idx'] = wrapper_step_idx; 
        batch_out_dict_lst = []
        t_entire_init = time.time()
        crit_lst, exec_time_lst = [], []
        misc_info_lst, misc_info_val_lst = [], []
        max_cpu_cores_used = 1; max_gpu_cores_used = 1
        mut_idx = 0
        mut_prot_seq = replica_mut_seq
        prv_best_mutated_prot_seq = mut_prot_seq
        p_stable = replica_prob
        t_mcmc = replica_temp
        trigger_fixed_mut_num_mna = False
        resourceMonitor_inst_dcopy = None
        if(resourceMonitor_inst is not None):
            resourceMonitor_inst_dcopy = copy.deepcopy(resourceMonitor_inst)
        t_batch_init = time.time()
        for local_step_idx in range(local_step_size):
            t1 = time.time()
            ppi_score_arr = np.empty((0,), dtype=float)
            mut_prot_fastas, mut_pos_lst, aa_idx_lst,  = [], [], []
            mod_local_step_size = local_step_size
            if(local_step_idx == 0):
                mut_prot_fastas.append((mut_prot_id, mut_prot_seq))
                mod_local_step_size = local_step_size -1
                mut_idx += 1
                mut_pos_lst.append([-1])  
                aa_idx_lst.append([-1])  
            for itr in range(mod_local_step_size):
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
            ##
            mut_plm_1d_embedd_tensor_lst = preproc_plm_util.extract_protTrans_plm_feat(prot_seq_lst=mut_prot_seq_lst, model=protTrans_model
                                                                                    , tokenizer=tokenizer, device=device)
            plm_feat_dict = dict(zip([fixed_prot_id] + mut_prot_id_lst, fixed_plm_1d_embedd_tensor_lst + mut_plm_1d_embedd_tensor_lst))
            t3 = time.time()
            man_2d_feat_dict_prot = man_1d_feat_dict_prot = None
            if(local_step_size >= 1): 
                man_2d_feat_dict_prot, man_1d_feat_dict_prot = extract_manual_feat_serially(fix_mut_prot_id, featureDir, mut_prot_fastas, use_psiblast_for_pssm, psiblast_exec_path, blosumMatrix, mut_prot_id_lst)
            else:  
                num_process = 0
                if((local_step_size > 1) and (local_step_size < num_cpu_cores)):
                    num_process = local_step_size
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
                batch_out_dict = {'local_step_idx': local_step_idx}
                batch_out_dict['simuln_idx'] = (local_step_size * local_step_idx) + min_val_index
                batch_out_dict['mut_pos_lst'] = mut_pos_lst[min_val_index]
                batch_out_dict['aa_idx_lst'] = aa_idx_lst[min_val_index]
                batch_out_dict['ppi_score'] = p_i
                batch_out_dict['prot_seq'] = mut_prot_fastas[min_val_index][1]
                batch_out_dict_lst.append(batch_out_dict)
                prv_best_mutated_prot_seq = mut_prot_fastas[min_val_index][1]
            crit_lst.append(f'local_step_idx_{local_step_idx}'); exec_time_lst.append(round(t8-t1, 3))
            if(trigger_fixed_mut_num_mna):
                prv_best_mutated_prot_seq = mut_prot_seq
                p_stable = -1.0
            if((local_step_idx + 1) % local_step_size == 0):
                batch_out_df = None
                if(len(batch_out_dict_lst) > 0):  
                    t9_1 = time.time()
                    batch_out_df = pd.DataFrame(batch_out_dict_lst)
                    batch_out_df.insert(loc=0, column='replica_index', value=[replica_index_remc] * batch_out_df.shape[0])
                    batch_out_df.insert(loc=0, column='wrapper_step_idx', value=[wrapper_step_idx] * batch_out_df.shape[0])
                    out_replica_work_dict['batch_out_df'] = batch_out_df
                    out_replica_work_dict['replica_mut_seq'] = batch_out_dict_lst[-1]['prot_seq']
                    out_replica_work_dict['replica_prob'] = batch_out_dict_lst[-1]['ppi_score']
                    if(resourceMonitor_inst_dcopy != None):  
                        t10_1 = time.time()
                        resourceMonitor_inst_dcopy.monitor_peak_ram_usage()
                        resourceMonitor_inst_dcopy.monitor_peak_gpu_memory_usage()
                        t10_2 = time.time()
        t_final = time.time()
        crit_lst.append('batch_total_time'); exec_time_lst.append(round(t_final - t_batch_init, 3))
        crit_lst.append('simuln_total_time'); exec_time_lst.append(round(t_final - t_entire_init, 3))
        time_df = pd.DataFrame({'criterion': crit_lst, 'time_in_sec': exec_time_lst})
        time_df.insert(loc=0, column='replica_index', value=[replica_index_remc] * time_df.shape[0])
        time_df.insert(loc=0, column='wrapper_step_idx', value=[wrapper_step_idx] * time_df.shape[0])
        out_replica_work_dict['time_df'] = time_df
        misc_info_lst.append('tot_num_batches_executed'); misc_info_val_lst.append(local_step_idx + 1)
        misc_info_lst.append('local_step_size'); misc_info_val_lst.append(local_step_size)
        misc_info_lst.append('batch_total_time_in_sec'); misc_info_val_lst.append(round(t_final - t_batch_init, 3))
        misc_info_lst.append('tot_num_itr_executed'); misc_info_val_lst.append((local_step_idx + 1) * local_step_size)
        misc_info_df = pd.DataFrame({'misc_info': misc_info_lst, 'misc_info_val': misc_info_val_lst})
        misc_info_df.insert(loc=0, column='replica_index', value=[replica_index_remc] * misc_info_df.shape[0])
        misc_info_df.insert(loc=0, column='wrapper_step_idx', value=[wrapper_step_idx] * misc_info_df.shape[0])
        out_replica_work_dict['misc_info_df'] = misc_info_df
        return out_replica_work_dict
    except Exception as ex:
        error_dict = {}
        error_dict['fix_mut_prot_id_tag'] = fix_mut_prot_id_tag
        error_dict['wrapper_step_idx'] = wrapper_step_idx
        error_dict['replica_index'] = replica_index_remc
        error_dict['error'] = str(ex)
        return error_dict


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
