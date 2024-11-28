import os, sys

from pathlib import Path
path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))

import glob
import pandas as pd
import numpy as np
from utils import prot_design_util


def calc_score(root_path, iteration_tag):
    result_dump_folder = os.path.join(root_path, 'dataset/proc_data', 'result_dump_' + iteration_tag + '/*')
    res_totItr_folders_lst = glob.glob(os.path.join(result_dump_folder), recursive=False)

    complx_chain_combo_sim_res_dict_lst = []
    for complx_sim_res in res_totItr_folders_lst:
        complx_nm = complx_sim_res.split('/')[-1]
        print(f'complx_nm: {complx_nm}')
        
        # if(complx_nm in ['complex_1OPH', 'complex_1S1Q', 'complex_1GPW']):
        #     continue
        
        chain_combo_itr_folder_lst = glob.glob(os.path.join(complx_sim_res + '/*'), recursive=False)
        # for indiv_chain_combo_itr_folder in chain_combo_itr_folder_lst:
        for itr in range(1, len(chain_combo_itr_folder_lst)):
            indiv_chain_combo_itr_folder = chain_combo_itr_folder_lst[itr]
            chain_combo_name = indiv_chain_combo_itr_folder.split('/')[-1]
            chain_combo_name_lst = chain_combo_name.replace('fixed_', '').replace('mut_', '').split('_')
            print(f'chain_combo_name: {chain_combo_name_lst}')
            
            chain_combo_itr_res_folder = glob.glob(os.path.join(indiv_chain_combo_itr_folder, 'res_totItr*'), recursive=False)[0]
            misc_info_csv_path = os.path.join(chain_combo_itr_res_folder, 'misc_info.csv')
            if(not os.path.exists(misc_info_csv_path)):
                # misc_info.csv file does not exist
                continue
            misc_info_df = pd.read_csv(misc_info_csv_path)
            tot_num_itr_executed_row = misc_info_df[misc_info_df['misc_info'] == 'tot_num_itr_executed']
            tot_num_itr_executed = int(tot_num_itr_executed_row['misc_info_val'])
            print(f'tot_num_itr_executed: {tot_num_itr_executed}')
            
            batch_sim_res_csv_file_lst = glob.glob(os.path.join(chain_combo_itr_res_folder, 'batchIdx_*.csv'), recursive=False)
            entire_ppi_score_lst = []

            for indiv_batch_sim_res_csv_file in batch_sim_res_csv_file_lst:
                indiv_batch_sim_res_df = pd.read_csv(indiv_batch_sim_res_csv_file)
                indiv_batch_ppi_lst = indiv_batch_sim_res_df['ppi_score'].tolist() 
                entire_ppi_score_lst += indiv_batch_ppi_lst
            # end of for loop: for indiv_batch_sim_res_csv_file in batch_sim_res_csv_file_lst:

            complx_chain_combo_sim_res_dict = {}
            complx_chain_combo_sim_res_dict['complx_nm'] = complx_nm
            complx_chain_combo_sim_res_dict['chain_combo_nm'] = chain_combo_name_lst
            complx_chain_combo_sim_res_dict['ppi_score'] = entire_ppi_score_lst[0]
            
            complx_chain_combo_sim_res_dict_lst.append(complx_chain_combo_sim_res_dict)
        # end of for loop: for indiv_chain_combo_itr_folder in chain_combo_itr_folder_lst:
    # end of for loop: for complx_sim_res in res_totItr_folders_lst:
    complx_chain_combo_sim_res_df = pd.DataFrame(complx_chain_combo_sim_res_dict_lst)
    complx_chain_combo_sim_res_csv_path = os.path.join(root_path, 'dataset/proc_data', 'result_dump_' + iteration_tag, 'score_oriiginal_pair_only.csv')
    complx_chain_combo_sim_res_df.to_csv(complx_chain_combo_sim_res_csv_path, index=False)


def parse_pdb(root_path):
    pdb_file_location = os.path.join(root_path, "dataset/preproc_data/pdb_files")
    dim_prot_complx_nm_lst_10_orig = ['2I25', '4Y7M', '5JMO', '5SV3', '6CWG', '6DBG', '1AVX', '1AY7', '1BUH', '1BVN'] \
        + ['1CLV', '1D6R', '1DFJ', '1E6E', '1EAW', '1EWY', '1F34', '1FLE', '1GL1', '1GLA'] \
        + ['1GXD', '1JTG', '1MAH', '1OC0', '1OPH', '1OYV', '1PPE', '1R0R', '1TMQ'] \
        + ['1UDI', '1US7', '1YVB', '1Z5Y', '2A9K', '2ABZ', '2AYO', '2B42', '2J0T', '2O8V'] \
        + ['2OOB', '2OUL', '2PCC', '2SIC', '2SNI', '2UUY', '3SGQ', '4CPA', '7CEI', '1AK4'] \
        + ['1E96', '1EFN', '1FFW', '1FQJ', '1GCQ', '1GHQ', '1GPW', '1H9D', '1HE1', '1J2J'] \
        + ['1KAC', '1KTZ', '1KXP', '1PVH', '1QA9', '1S1Q', '1SBB', '1T6B', '1XD3', '1Z0K'] \
        + ['1ZHH', '1ZHI', '2A5T', '2AJF', '2BTF', '2FJU', '2G77', '2HLE', '2HQS', '2VDB', '3D5S']
    dim_prot_complx_nm_lst_done =   []
    dim_prot_complx_nm_lst_notConsidered = ['5JMO', '1PPE', '1FQJ', '1OYV']
    dim_prot_complx_nm_lst_chainBreak = ['1AVX', '1AY7', '1BUH', '1D6R', '1EAW', '1EFN', '1F34', '1FLE', '1GL1', '1GLA', '1GPW', '1GXD', '1H9D', '1JTG', '1KTZ', '1KXP', '1MAH', '1OC0', '1OPH', '1OYV' \
                                        ,'1PPE', '1R0R', '1S1Q', '1SBB', '1T6B', '1US7', '1XD3', '1YVB', '1Z5Y', '1ZHH', '1ZHI', '2AST', '2ABZ', '2AJF', '2AYO', '2B42', '2BTF', '2FJU', '2G77', '2HLE' \
                                        ,'2HQS', '2O8V', '2PCC', '2UUY', '2VDB', '3SGQ', '4Y7M', '5JMO', '5SV3', '6CWG', '6DBG']
    dim_prot_complx_nm_lst_excluded = dim_prot_complx_nm_lst_done + dim_prot_complx_nm_lst_notConsidered
    dim_prot_complx_nm_lst_effective = [prot_id for prot_id in dim_prot_complx_nm_lst_10_orig if prot_id not in dim_prot_complx_nm_lst_excluded]

    for dim_prot_complx_nm in dim_prot_complx_nm_lst_effective:
        print(f'\n ######################### dim_prot_complx_nm : {dim_prot_complx_nm} #################### - Start \n')
        chain_sequences_dict = prot_design_util.extract_chain_sequences(dim_prot_complx_nm, pdb_file_location)
        print(f'\n ######################### dim_prot_complx_nm : {dim_prot_complx_nm} #################### - End \n')



if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/matpip_pd_prj') 

    # iteration_tag_lst = ['fullLen_puTrue_thorough', 'fullLen_puTrue_fast', 'intrfc_puFalse_batch2_thorough', 'intrfc_puFalse_batch2_fast', 'fullLen_puFalse_batch5_thorough', 'fullLen_puFalse_batch5_fast']
    iteration_tag_lst = ['orig_pair_thorough']
    for iteration_tag in iteration_tag_lst:
        calc_score(root_path, iteration_tag)
    
    # parse_pdb(root_path)

