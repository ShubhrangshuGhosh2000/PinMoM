import os, sys

from pathlib import Path
path_root = Path(__file__).parents[2]  
sys.path.insert(0, str(path_root))


from Bio.Blast import NCBIXML
import glob
import pandas as pd
import subprocess
from utils import PPIPUtils, postproc_clustering


def run_postproc_after_simuln(**kwargs):
    for arg_name, arg_value in kwargs.items():
        if(arg_value is None):
            raise ValueError(f"Argument '{arg_name}' must be provided with a value.")
        
    root_path = kwargs.get('root_path'); dim_prot_complx_nm = kwargs.get('dim_prot_complx_nm')
    simulation_exec_mode = kwargs.get('simulation_exec_mode')
    sim_result_dir = kwargs.get('sim_result_dir'); postproc_result_dir = kwargs.get('postproc_result_dir')
    ppi_score_threshold = kwargs.get('ppi_score_threshold')
    postproc_psiblast_enabled = kwargs.get('postproc_psiblast_enabled')
    postproc_psiblast_uniprot_db_path = kwargs.get('postproc_psiblast_uniprot_db_path')
    postproc_psiblast_result_dir = kwargs.get('postproc_psiblast_result_dir');
    max_num_of_seq_for_af2_chain_struct_pred = kwargs.get('max_num_of_seq_for_af2_chain_struct_pred')
    inp_dir_mut_seq_struct_pred_af2 = kwargs.get('inp_dir_mut_seq_struct_pred_af2')
    os.environ['BLASTDB'] = postproc_psiblast_uniprot_db_path
    complx_postproc_result_dir = os.path.join(postproc_result_dir, f'complex_{dim_prot_complx_nm}')
    PPIPUtils.createFolder(complx_postproc_result_dir, recreate_if_exists=False)
    misc_info_complx_postproc_1_lst, misc_info_val_complx_postproc_1_lst = [], []
    simuln_res_dir = os.path.join(root_path, sim_result_dir, f'complex_{dim_prot_complx_nm}')

    if(not os.path.exists(simuln_res_dir)):
        misc_info_complx_postproc_1_lst.append('sim_res_dir_search_result'); misc_info_val_complx_postproc_1_lst.append(f'{simuln_res_dir} NOT FOUND')
        misc_info_complx_postproc_1_lst.append('status'); misc_info_val_complx_postproc_1_lst.append('ERROR')
        misc_info_complx_postproc_1_df = pd.DataFrame({'misc_info': misc_info_complx_postproc_1_lst, 'misc_info_val': misc_info_val_complx_postproc_1_lst})
        misc_info_complx_postproc_1_df.to_csv(os.path.join(complx_postproc_result_dir, 'misc_info_complex_postproc_1.csv'), index=False)
        return  
    misc_info_complx_postproc_1_lst.append('sim_res_dir_search_result'); misc_info_val_complx_postproc_1_lst.append(f'{simuln_res_dir} FOUND')
    chain_combo_lst = PPIPUtils.get_top_level_directories(simuln_res_dir)
    if(len(chain_combo_lst) == 0):
        misc_info_complx_postproc_1_lst.append('chain_combo_inside_sim_res_dir'); misc_info_val_complx_postproc_1_lst.append(f'NOT FOUND')
        misc_info_complx_postproc_1_lst.append('status'); misc_info_val_complx_postproc_1_lst.append('ERROR')
        misc_info_complx_postproc_1_df = pd.DataFrame({'misc_info': misc_info_complx_postproc_1_lst, 'misc_info_val': misc_info_val_complx_postproc_1_lst})
        misc_info_complx_postproc_1_df.to_csv(os.path.join(complx_postproc_result_dir, 'misc_info_complex_postproc_1.csv'), index=False)
        return  
    
    misc_info_complx_postproc_1_lst.append('chain_combo_inside_sim_res_dir'); misc_info_val_complx_postproc_1_lst.append(f'{chain_combo_lst}')
    misc_info_complx_postproc_1_df = pd.DataFrame({'misc_info': misc_info_complx_postproc_1_lst, 'misc_info_val': misc_info_val_complx_postproc_1_lst})
    misc_info_complx_postproc_1_df.to_csv(os.path.join(complx_postproc_result_dir, 'misc_info_complex_postproc_1.csv'), index=False)
    
    for chain_combo_itr in range(len(chain_combo_lst)):
        curnt_chain_combo = chain_combo_lst[chain_combo_itr]
        curnt_chain_combo_splitted_lst = curnt_chain_combo.split('_')
        fixed_prot_id, mut_prot_id = curnt_chain_combo_splitted_lst[1], curnt_chain_combo_splitted_lst[-1]
        fix_mut_prot_id_tag = f' Postproc_1: complex_{dim_prot_complx_nm}_fixed_prot_id_{fixed_prot_id}_mut_prot_id_{mut_prot_id}:: '
        curnt_chain_combo_postproc_result_dir = os.path.join(complx_postproc_result_dir, f'{curnt_chain_combo}')
        PPIPUtils.createFolder(curnt_chain_combo_postproc_result_dir, recreate_if_exists=False)
        curnt_chain_combo_postproc_1_info_lst, curnt_chain_combo_postproc_1_val_lst = [], []
        res_totItr_folders = glob.glob(os.path.join(simuln_res_dir, curnt_chain_combo, 'res_totItr*'), recursive=False)
        num_res_totItr_folders = len(res_totItr_folders)
        curnt_chain_combo_postproc_1_info_lst.append('num_res_totItr_dir_inside_chain_combo_dir'); curnt_chain_combo_postproc_1_val_lst.append(f'{num_res_totItr_folders}')
        if(num_res_totItr_folders != 1):
            curnt_chain_combo_postproc_1_info_lst.append('status'); curnt_chain_combo_postproc_1_val_lst.append('ERROR')
            curnt_chain_combo_postproc_1_info_df = pd.DataFrame({'misc_info': curnt_chain_combo_postproc_1_info_lst, 'misc_info_val': curnt_chain_combo_postproc_1_val_lst})
            curnt_chain_combo_postproc_1_info_df.to_csv(os.path.join(curnt_chain_combo_postproc_result_dir, 'chain_combo_postproc_1.csv'), index=False)
            continue  
        res_totItr_dir = res_totItr_folders[0]
        batch_res_csv_files = None
        if(simulation_exec_mode == 'remc'):
            batch_res_csv_files = glob.glob(os.path.join(res_totItr_dir, 'wrapperStepIdx_*_*_localStepSize_*.csv'))
        else:
            batch_res_csv_files = glob.glob(os.path.join(res_totItr_dir, 'batchIdx_*_*_batchSize_*.csv'))
        prob_thresholded_df_lst = []
        for indiv_batch_res_csv_file in batch_res_csv_files:
            indiv_batch_res_df = pd.read_csv(indiv_batch_res_csv_file)
            indiv_prob_thresholded_df = indiv_batch_res_df[(indiv_batch_res_df['ppi_score'] > ppi_score_threshold) & (~((indiv_batch_res_df['batch_idx'] == 0) & (indiv_batch_res_df['simuln_idx'] == 0)))]
            indiv_prob_thresholded_df = indiv_prob_thresholded_df.drop_duplicates(subset=['prot_seq'], keep='first')
            indiv_prob_thresholded_df = indiv_prob_thresholded_df.reset_index(drop=True)
            if(indiv_prob_thresholded_df.shape[0] > 0):
                prob_thresholded_df_lst.append(indiv_prob_thresholded_df)
        curnt_chain_combo_postproc_1_info_lst.append('len(prob_thresholded_df_lst)'); curnt_chain_combo_postproc_1_val_lst.append(f'{len(prob_thresholded_df_lst)}')
        if(len(prob_thresholded_df_lst) == 0):
            curnt_chain_combo_postproc_1_info_lst.append('status'); curnt_chain_combo_postproc_1_val_lst.append('ERROR')
            curnt_chain_combo_postproc_1_info_df = pd.DataFrame({'misc_info': curnt_chain_combo_postproc_1_info_lst, 'misc_info_val': curnt_chain_combo_postproc_1_val_lst})
            curnt_chain_combo_postproc_1_info_df.to_csv(os.path.join(curnt_chain_combo_postproc_result_dir, 'chain_combo_postproc_1.csv'), index=False)
            continue  
        prob_thresholded_df = pd.concat(prob_thresholded_df_lst, ignore_index=True, sort=False)
        prob_thresholded_df = prob_thresholded_df.drop_duplicates(subset=['prot_seq'], keep='first')
        prob_thresholded_df = prob_thresholded_df.reset_index(drop=True)
        prob_thresholded_df.sort_values(by=['ppi_score'], ascending=False, inplace=True, ignore_index=True)
        prob_thresholded_df.to_csv(os.path.join(curnt_chain_combo_postproc_result_dir, 'prob_thresholded_df.csv'), index=False)
        PPIPUtils.createFolder(postproc_psiblast_result_dir, recreate_if_exists=False)
        kwargs['fix_mut_prot_id_tag'] = fix_mut_prot_id_tag; kwargs['curnt_chain_combo'] = curnt_chain_combo; kwargs['prob_thresholded_df'] = prob_thresholded_df
        psi_blast_thresholded_df = None
        if(not postproc_psiblast_enabled):
            psi_blast_thresholded_df = prob_thresholded_df
        else:
            psi_blast_thresholded_df = apply_postproc_psi_blast(**kwargs)
        if(psi_blast_thresholded_df is None):
            curnt_chain_combo_postproc_1_info_lst.append('psi_blast_thresholded_df.shape[0]'); curnt_chain_combo_postproc_1_val_lst.append(str(0))
            curnt_chain_combo_postproc_1_info_lst.append('status'); curnt_chain_combo_postproc_1_val_lst.append('ERROR')
            curnt_chain_combo_postproc_1_info_df = pd.DataFrame({'misc_info': curnt_chain_combo_postproc_1_info_lst, 'misc_info_val': curnt_chain_combo_postproc_1_val_lst})
            curnt_chain_combo_postproc_1_info_df.to_csv(os.path.join(curnt_chain_combo_postproc_result_dir, 'chain_combo_postproc_1.csv'), index=False)
            continue  
        num_rows_psi_blast_thresholded_df = psi_blast_thresholded_df.shape[0]
        curnt_chain_combo_postproc_1_info_lst.append('psi_blast_thresholded_df.shape[0]'); curnt_chain_combo_postproc_1_val_lst.append(f'{num_rows_psi_blast_thresholded_df}')
        psi_blast_thresholded_df.to_csv(os.path.join(curnt_chain_combo_postproc_result_dir, 'psi_blast_thresholded_df.csv'), index=False)
        k = max_num_of_seq_for_af2_chain_struct_pred
        sel_k_seq_psiBlast_thresholded_df_for_af2_pred = None
        if(num_rows_psi_blast_thresholded_df <= k):
            sel_k_seq_psiBlast_thresholded_df_for_af2_pred = psi_blast_thresholded_df
        else:
            kwargs['psi_blast_thresholded_df'] = psi_blast_thresholded_df
            sel_k_seq_psiBlast_thresholded_df_for_af2_pred = postproc_clustering.perform_clustering(**kwargs)
        sel_k_seq_psiBlast_thresholded_df_for_af2_pred.to_csv(os.path.join(curnt_chain_combo_postproc_result_dir,f'sel_{k}_seq_psiBlast_thresholded.csv'), index=False)
        curnt_chain_combo_postproc_1_info_df = pd.DataFrame({'misc_info': curnt_chain_combo_postproc_1_info_lst, 'misc_info_val': curnt_chain_combo_postproc_1_val_lst})
        curnt_chain_combo_postproc_1_info_df.to_csv(os.path.join(curnt_chain_combo_postproc_result_dir, 'chain_combo_postproc_1.csv'), index=False)
        mut_seq_struct_af2_inp_df = pd.DataFrame()
        mut_seq_struct_af2_inp_df['id'] = f'cmplx_{dim_prot_complx_nm}_fpi_{fixed_prot_id}_mpi_{mut_prot_id}_batch_idx_' + sel_k_seq_psiBlast_thresholded_df_for_af2_pred['batch_idx'].astype(str) + '_simuln_idx_' + sel_k_seq_psiBlast_thresholded_df_for_af2_pred['simuln_idx'].astype(str)
        mut_seq_struct_af2_inp_df['sequence'] = sel_k_seq_psiBlast_thresholded_df_for_af2_pred['prot_seq']
        mut_seq_struct_af2_inp_df.to_csv(os.path.join(curnt_chain_combo_postproc_result_dir, f'af2inp_cmpl_{dim_prot_complx_nm}_fpi_{fixed_prot_id}_mpi_{mut_prot_id}.csv'), index=False)
        mut_seq_struct_af2_inp_df.to_csv(os.path.join(inp_dir_mut_seq_struct_pred_af2, f'af2inp_cmpl_{dim_prot_complx_nm}_fpi_{fixed_prot_id}_mpi_{mut_prot_id}.csv'), index=False)
        

def apply_postproc_psi_blast(**kwargs):
    simulation_exec_mode = kwargs.get('simulation_exec_mode')
    postproc_psiblast_exec_path = kwargs.get('postproc_psiblast_exec_path')
    postproc_psiblast_result_dir = kwargs.get('postproc_psiblast_result_dir')
    prob_thresholded_df = kwargs.get('prob_thresholded_df')
    dim_prot_complx_nm = kwargs.get('dim_prot_complx_nm'); curnt_chain_combo = kwargs.get('curnt_chain_combo')
    psiBlast_percent_similarity_score_threshold = kwargs.get('psiBlast_percent_similarity_score_threshold')
    psi_blast_thresholded_df = None
    psi_blast_thresholded_dict_lst = []
    crnt_successful_seq_count = 0
    
    for index, row in prob_thresholded_df.iterrows():
        row_dict = row.to_dict()
        sequence = row_dict['prot_seq']
        skip_psi_blast_for_crnt_row = False
        psi_blast_result_xml_file_nm = None
        if(simulation_exec_mode == 'remc'):
            psi_blast_result_xml_file_nm = os.path.join(postproc_psiblast_result_dir
                                        , f'cmplx_{dim_prot_complx_nm}_{curnt_chain_combo}_wrapper_step_idx_{row_dict["wrapper_step_idx"]}_replica_index_{row_dict["replica_index"]}_batch_idx_{row_dict["batch_idx"]}_simuln_idx_{row_dict["simuln_idx"]}_blast_res.xml')
        else:
            psi_blast_result_xml_file_nm = os.path.join(postproc_psiblast_result_dir
                                        , f'cmplx_{dim_prot_complx_nm}_{curnt_chain_combo}_batch_idx_{row_dict["batch_idx"]}_simuln_idx_{row_dict["simuln_idx"]}_blast_res.xml')
        query_sequence = f">query_sequence\n{sequence}"
        psiblast_cmd = [
            f'{postproc_psiblast_exec_path}psiblast',
            '-db', 'uniprotSprotFull_v2',
            '-evalue', str(0.001),
            '-num_iterations', str(4),  
            '-outfmt', str(5),  
            '-out', psi_blast_result_xml_file_nm,  
            '-num_threads', str(4)
        ]
        try:
            result = subprocess.run(psiblast_cmd, input=query_sequence, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0:
                raise Exception(f"!!! Error running PSI-BLAST: {result.stderr}")
            with open(psi_blast_result_xml_file_nm) as result_handle:
                blast_records = NCBIXML.parse(result_handle)
                for record in blast_records:
                    for alignment in record.alignments:
                        for hsp in alignment.hsps:
                            percent_similarity_score = (hsp.identities / hsp.align_length) * 100.0
                            if(percent_similarity_score > psiBlast_percent_similarity_score_threshold):
                                skip_psi_blast_for_crnt_row = True
                                break                            
                        if(skip_psi_blast_for_crnt_row): break
                    if(skip_psi_blast_for_crnt_row): break
            if(not skip_psi_blast_for_crnt_row):
                psi_blast_thresholded_dict_lst.append(row_dict)
                crnt_successful_seq_count += 1
        except Exception as ex:
            print(f"*********** !!!! apply_postproc_psi_blast: An exception occurred: {ex}")
    if(len(psi_blast_thresholded_dict_lst) > 0):
        psi_blast_thresholded_df = pd.DataFrame(psi_blast_thresholded_dict_lst)
    return psi_blast_thresholded_df

