import os, sys

from pathlib import Path
path_root = Path(__file__).parents[4]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

# from utils import dl_reproducible_result_util
import glob
import pandas as pd
from utils import prot_design_util, PPIPUtils

def prep_initial_pul_data(root_path='./', itr_tag=None, fixed_str=None):
    print('inside prep_initial_pul_data() method - Start')
    pdb_file_location = os.path.join(root_path, "dataset/preproc_data/pdb_files")
    data_point_dict_list = []
    lower_bound = 9.0  # 10.0, 11.0, 12.0
    upper_bound = 10.0  # 10.0, 12.0, 15.0, 17.0
    sel_count = 50  # 500, 1000, 3000, 5000

    # First, original dimers from the benchmark are taken with their 2 chains as positive samples
    print(f'\n ################# First, original dimers from the benchmark are taken with their 2 chains as positive samples ###########\n')
    # dim_prot_complx_nm_lst_orig = ['2I25', '4Y7M', '5JMO']
    dim_prot_complx_nm_lst_orig = ['2I25', '4Y7M', '5JMO', '5SV3', '6CWG', '6DBG', '1AVX', '1AY7', '1BUH', '1BVN'] \
    + ['1CLV', '1D6R', '1DFJ', '1E6E', '1EAW', '1EWY', '1F34', '1FLE', '1GL1', '1GLA'] \
    + ['1GXD', '1JTG', '1MAH', '1OC0', '1OPH', '1OYV', '1PPE', '1R0R', '1TMQ'] \
    + ['1UDI', '1US7', '1YVB', '1Z5Y', '2A9K', '2ABZ', '2AYO', '2B42', '2J0T', '2O8V'] \
    + ['2OOB', '2OUL', '2PCC', '2SIC', '2SNI', '2UUY', '3SGQ', '4CPA', '7CEI', '1AK4'] \
    + ['1E96', '1EFN', '1FFW', '1FQJ', '1GCQ', '1GHQ', '1GPW', '1H9D', '1HE1', '1J2J'] \
    + ['1KAC', '1KTZ', '1KXP', '1PVH', '1QA9', '1S1Q', '1SBB', '1T6B', '1XD3', '1Z0K'] \
    + ['1ZHH', '1ZHI', '2A5T', '2AJF', '2BTF', '2FJU', '2G77', '2HLE', '2HQS', '2VDB', '3D5S']

    for orig_dimer_itr, dim_prot_complx_nm in enumerate(dim_prot_complx_nm_lst_orig):
        print(f'\n dim_prot_complx_nm: {dim_prot_complx_nm}  :: {orig_dimer_itr+1}  out of {len(dim_prot_complx_nm_lst_orig)}\n')
        chain_sequences_dict = prot_design_util.extract_chain_sequences(dim_prot_complx_nm, pdb_file_location)
        chain_nm_lst= list(chain_sequences_dict.keys())
        data_point_dict = {}
        data_point_dict['prot_id'] = dim_prot_complx_nm
        data_point_dict['chain_1_seq'] = chain_sequences_dict[chain_nm_lst[0]]
        data_point_dict['chain_2_seq'] = chain_sequences_dict[chain_nm_lst[1]]
        data_point_dict['label'] = 1  # 0 for negative, 1 for positive and -1 for unlabelled
        data_point_dict['ppi_score'] = 1  # For sure event 
        data_point_dict['num_mut_pts'] = 0  # For orginal complex, no mutation
        data_point_dict_list.append(data_point_dict)
    # End of for loop: for orig_dimer_itr, dim_prot_complx_nm in enumerate(dim_prot_complx_nm_lst_orig):

    # Next, from MD-Simulation result add some positive samples
    print(f'\n ################# Next, from MD-Simulation result add some positive samples ###########\n')
    md_sim_res_csv_path = os.path.join(root_path, 'dataset/postproc_data/result_dump', itr_tag, 'overall_md_sim_avg_rmsd_rmsf_rg_postproc_6.csv')
    md_sim_res_df = pd.read_csv(md_sim_res_csv_path)
    # 'overall_accept_complx_struct_comp_res_postproc_5_part2' will be required for the sequence retrieval
    overall_accept_complx_struct_comp_res_postproc_5_part2_csv_path = os.path.join(root_path
    , f'dataset/postproc_data/result_dump/{itr_tag}', 'overall_accept_complx_struct_comp_res_postproc_5_part2.csv')
    overall_accept_complx_struct_comp_res_postproc_5_part2_df = pd.read_csv(overall_accept_complx_struct_comp_res_postproc_5_part2_csv_path)

    # positive md-simulation result will have avg. rmsd value < 2.0 and should NOT be the original complex
    positive_md_sim_df = md_sim_res_df[(md_sim_res_df['avg_rmsd_nm'] <= 2.0)
                                        & (~(md_sim_res_df['complx_id'].str.contains('(orig)')))
                                        & (~(md_sim_res_df['complx_id'].str.contains('|'.join(['1AY7', '1YVB', '1ZHI']))))]
    positive_md_sim_df = positive_md_sim_df.reset_index(drop=True)

    for index, row in positive_md_sim_df.iterrows():
        print(f'\n MD-Simulation: Positive samples: Iteration {index + 1} out of {positive_md_sim_df.shape[0]}\n')
        complx_id = row['complx_id'].split(':')[-1].strip()  # row['complx_id'] => '1EFN : cmplx_1EFN_fpi_B_mpi_A_batch_idx_5128_simuln_idx_25644'
        print(f'complx_id: {complx_id}')
        data_point_dict = {}
        data_point_dict['prot_id'] = complx_id
        # Retrieve fixed and mutating chain sequence
        specific_row = overall_accept_complx_struct_comp_res_postproc_5_part2_df[overall_accept_complx_struct_comp_res_postproc_5_part2_df['id'] == complx_id]
        fixed_mut_seq = specific_row['sequence'].values[0]
        fixed_mut_seq_split_lst = fixed_mut_seq.split(':')
        data_point_dict['chain_1_seq'] = fixed_mut_seq_split_lst[0]
        data_point_dict['chain_2_seq'] = fixed_mut_seq_split_lst[1]
        data_point_dict['label'] = 1  # 0 for negative, 1 for positive and -1 for unlabelled
        data_point_dict['ppi_score'] = specific_row['ppi_score'].values[0]
        data_point_dict['num_mut_pts'] = find_num_mut_pts(root_path=root_path, itr_tag=itr_tag, fixed_str=fixed_str, complx_id=complx_id)
        data_point_dict_list.append(data_point_dict)
    # End of for loop: for index, row in positive_md_sim_df.iterrows():

    # Next, from MD-Simulation result add some negative samples
    print(f'\n ################# Next, from MD-Simulation result add some negative samples ###########\n')
    # negative md-simulation result will have avg. rmsd value > 2.0 and should NOT be the original complex
    negative_md_sim_df = md_sim_res_df[(md_sim_res_df['avg_rmsd_nm'] > 2.0)
                                        & (~(md_sim_res_df['complx_id'].str.contains('(orig)')))
                                        & (~(md_sim_res_df['complx_id'].str.contains('|'.join(['1AY7', '1YVB', '1ZHI']))))]
    negative_md_sim_df = negative_md_sim_df.reset_index(drop=True)

    for index, row in negative_md_sim_df.iterrows():
        print(f'\n MD-Simulation: Negative samples: Iteration {index + 1} out of {negative_md_sim_df.shape[0]}\n')
        complx_id = row['complx_id'].split(':')[-1].strip()  # row['complx_id'] => '1EFN : cmplx_1EFN_fpi_B_mpi_A_batch_idx_5128_simuln_idx_25644'
        print(f'complx_id: {complx_id}')
        data_point_dict = {}
        data_point_dict['prot_id'] = complx_id
        # Retrieve fixed and mutating chain sequence
        specific_row = overall_accept_complx_struct_comp_res_postproc_5_part2_df[overall_accept_complx_struct_comp_res_postproc_5_part2_df['id'] == complx_id]
        fixed_mut_seq = specific_row['sequence'].values[0]
        fixed_mut_seq_split_lst = fixed_mut_seq.split(':')
        data_point_dict['chain_1_seq'] = fixed_mut_seq_split_lst[0]
        data_point_dict['chain_2_seq'] = fixed_mut_seq_split_lst[1]
        data_point_dict['label'] = 0  # 0 for negative, 1 for positive and -1 for unlabelled
        data_point_dict['ppi_score'] = specific_row['ppi_score'].values[0]
        data_point_dict['num_mut_pts'] = find_num_mut_pts(root_path=root_path, itr_tag=itr_tag, fixed_str=fixed_str, complx_id=complx_id)
        data_point_dict_list.append(data_point_dict)
    # End of for loop: for index, row in negative_md_sim_df.iterrows():

    # Next, add those candidates as negative samples which are rejected during chain-structure comparison
    print(f'\n ################# Next, add those candidates as negative samples which are rejected during chain-structure comparison ###########\n')
    overall_chain_struct_comp_res_postproc_3_csv_path = os.path.join(root_path
                                                        , f'dataset/postproc_data/result_dump/{itr_tag}', 'overall_chain_struct_comp_res_postproc_3.csv')

    overall_chain_struct_comp_res_postproc_3_df = pd.read_csv(overall_chain_struct_comp_res_postproc_3_csv_path)
    negative_chain_samples_df = overall_chain_struct_comp_res_postproc_3_df[(overall_chain_struct_comp_res_postproc_3_df['bb_rmsd_superimp'] > lower_bound)
                                                                             & (overall_chain_struct_comp_res_postproc_3_df['bb_rmsd_superimp'] <= upper_bound)]
    negative_chain_samples_df = negative_chain_samples_df.reset_index(drop=True)
    for index, row in negative_chain_samples_df.iterrows():
        print(f'\n Chain comparison: Negative samples: Iteration {index + 1} out of {negative_chain_samples_df.shape[0]}\n')
        complx_id = f"cmplx_{row['cmplx']}_fpi_{row['fpi']}_mpi_{row['mpi']}_batch_idx_{row['batch_idx']}_simuln_idx_{row['simuln_idx']}"
        chain_sequences_dict = prot_design_util.extract_chain_sequences(row['cmplx'], pdb_file_location)
        data_point_dict = {}
        data_point_dict['prot_id'] = complx_id
        data_point_dict['chain_1_seq'] = chain_sequences_dict[row['fpi']]
        data_point_dict['chain_2_seq'] = chain_sequences_dict[row['mpi']]
        data_point_dict['label'] = 0  # 0 for negative, 1 for positive and -1 for unlabelled
        data_point_dict['ppi_score'] = row['ppi_score']
        data_point_dict['num_mut_pts'] = find_num_mut_pts(root_path=root_path, itr_tag=itr_tag, fixed_str=fixed_str, complx_id=complx_id)
        data_point_dict_list.append(data_point_dict)
    # End of for loop: for index, row in negative_md_sim_df.iterrows():

    # Next, add unalabelled samples which are not part of the clustering result
    print(f'\n ################# Next, add unalabelled samples which are not part of the clustering result ###########\n')
    proc_res_dump_path = os.path.join(root_path, 'dataset/proc_data', f'result_dump_{itr_tag}')
    postproc_res_dump_path = os.path.join(root_path, f'dataset/postproc_data/result_dump/{itr_tag}')
    # Find all the cmplx directories inside proc_res_dump_path, e.g. dataset/proc_data/result_dump_mcmc_fullLen_puFalse_batch5_mutPrcntLen10
    all_cmplx_dir_inside_proc_res_dump_path = glob.glob(os.path.join(proc_res_dump_path, '*/'))
    # Iterate through each proc_cmplx_dir
    for proc_cmplx_itr, proc_cmplx_dir in enumerate(all_cmplx_dir_inside_proc_res_dump_path):
        complx_id = proc_cmplx_dir.split('/')[-2]  # complex_1AK4
        complx_nm = complx_id.replace('complex_', '')  # 1AK4
        print(f'\n complx_nm: {complx_nm} : Iteration {proc_cmplx_itr +1} out of {len(all_cmplx_dir_inside_proc_res_dump_path)}\n')
        chain_sequences_dict = prot_design_util.extract_chain_sequences(complx_nm, pdb_file_location)
        # Find all fixed_mut directories inside each proc_cmplx_dir, e.g. complex_1AK4 => fixed_A_mut_C, fixed_C_mut_A
        fixed_mut_dirs = glob.glob(os.path.join(proc_cmplx_dir, '*/'))
        
        # Iterate through each fixed_mut_dirs
        for fixed_mut_dir in fixed_mut_dirs:
            fixed_mut_chain_combo_nm = fixed_mut_dir.split('/')[-2]  # 'fixed_A_mut_B'
            print(f'\n complx_nm: {complx_nm} :: fixed_mut_chain_combo_nm: {fixed_mut_chain_combo_nm}\n')
            fixed_mut_chain_nm_lst = fixed_mut_chain_combo_nm.split('_')  # ['fixed', 'A', 'mut, 'B']
            fpi = fixed_mut_chain_nm_lst[1]; mpi = fixed_mut_chain_nm_lst[-1]
            # Construct the path to the sel_1.csv file
            sel_seq_psiBlast_thresholded_csv_file_path = os.path.join(postproc_res_dump_path, complx_id, fixed_mut_chain_combo_nm, 'sel_50_seq_psiBlast_thresholded.csv')
            already_sel_seq_df = pd.read_csv(sel_seq_psiBlast_thresholded_csv_file_path)  # csv1

            
            proc_complx_fixed_mut_chain_combo_path = os.path.join(proc_res_dump_path, f'complex_{complx_nm}', fixed_mut_chain_combo_nm, fixed_str)
            proc_batch_csv_path = os.path.join(proc_complx_fixed_mut_chain_combo_path, 'batchIdx_3000_3999_batchSize_5.csv')
            proc_batch_df = pd.read_csv(proc_batch_csv_path)  # csv2

            # Remove rows in proc_batch_df that are also present in already_sel_seq_df based on all column values
            unique_proc_batch_df = proc_batch_df[~proc_batch_df.isin(already_sel_seq_df).all(axis=1)]
            unique_proc_batch_df = unique_proc_batch_df.reset_index(drop=True)
            unlabelled_sample_sel_df = unique_proc_batch_df.sample(n=sel_count, random_state=456)
            unlabelled_sample_sel_df = unlabelled_sample_sel_df.reset_index(drop=True)
            
            # Iterate through the unlabelled_sample_sel_df row-wise
            for index, row in unlabelled_sample_sel_df.iterrows():
                print(f'\n Unlabelled samples: Iteration {index + 1} out of {unlabelled_sample_sel_df.shape[0]}\n')
                id = f"cmplx_{complx_nm}_fpi_{fpi}_mpi_{mpi}_batch_idx_{row['batch_idx']}_simuln_idx_{row['simuln_idx']}"
                data_point_dict = {}
                data_point_dict['prot_id'] = id
                data_point_dict['chain_1_seq'] = chain_sequences_dict[fpi]
                data_point_dict['chain_2_seq'] = chain_sequences_dict[mpi]
                data_point_dict['label'] = -1  # 0 for negative, 1 for positive and -1 for unlabelled
                data_point_dict['ppi_score'] = row['ppi_score']
                data_point_dict['num_mut_pts'] = len(row['mut_pos_lst'].split(','))
                data_point_dict_list.append(data_point_dict)
            # End of for loop: for index, row in unlabelled_sample_sel_df.iterrows():
        # End of for loop: for fixed_mut_dir in fixed_mut_dirs:
    # End of for loop: for proc_cmplx_itr, proc_cmplx_dir in enumerate(all_cmplx_dir_inside_proc_res_dump_path):

    # Convert data_point_dict_list in a data-frame and save the data-frame
    preproc_pul_seq_df = pd.DataFrame(data_point_dict_list)
    preproc_pul_dir_path = os.path.join(root_path, 'dataset/postproc_data/pul_result', f'{itr_tag}/preproc_pul')
    PPIPUtils.createFolder(preproc_pul_dir_path)
    preproc_pul_seq_csv_path = os.path.join(preproc_pul_dir_path, 'preproc_1_pul_seq.csv')
    preproc_pul_seq_df.to_csv(preproc_pul_seq_csv_path, index=False)
    print('inside prep_initial_pul_data() method - End')


def find_num_mut_pts(root_path='./', itr_tag=None, fixed_str=None, complx_id=None):
    # Example: complx_id = cmplx_1EFN_fpi_B_mpi_A_batch_idx_5128_simuln_idx_25644 
    complx_id_splitted_lst = complx_id.split('_')
    complex_nm = f'complex_{complx_id_splitted_lst[1]}'
    fixed_mut_nm = f'fixed_{complx_id_splitted_lst[3]}_mut_{complx_id_splitted_lst[5]}'
    batch_idx = int(complx_id_splitted_lst[8])
    simuln_idx = int(complx_id_splitted_lst[-1])
    
    # Find the batchIdx csv file name
    range_size = 1000
    # Calculate the start and end of the log file range
    start = (batch_idx // range_size) * range_size
    end = start + range_size - 1
    # Format the log file name
    batchIdx_csv_file_nm = f"batchIdx_{start}_{end}_batchSize_5.csv"

    # Read the batchIdx_csv_file
    batchIdx_csv_file_loc = os.path.join(root_path, 'dataset/proc_data', f'result_dump_{itr_tag}', complex_nm
                                         , fixed_mut_nm, fixed_str) 

    batch_idx_spec_df = pd.read_csv(os.path.join(batchIdx_csv_file_loc, batchIdx_csv_file_nm))
    # Retrieve the specific row
    spec_row = batch_idx_spec_df[(batch_idx_spec_df['batch_idx'] == batch_idx)
                                 & (batch_idx_spec_df['simuln_idx'] == simuln_idx)]
    if(spec_row.empty):
        print(f'\n\n###################### find_num_mut_pts(): No specific row ###################\n\n')
        num_mut_pts = 2
    else:
        # Find the num_mut_pts
        mut_pos_lst_str = spec_row['mut_pos_lst'].values[0]
        num_mut_pts = len(mut_pos_lst_str.split(','))
    return num_mut_pts



if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/matpip_pd_prj')

    itr_tag = 'mcmc_fullLen_puFalse_batch5_mutPrcntLen10'
    fixed_str = 'res_totItr30000_batchSz5_percLnForMutPts10_puFalse_mutIntrfcFalse'


    # Prepare initial dataset required for the PU learning
    prep_initial_pul_data(root_path=root_path, itr_tag=itr_tag, fixed_str=fixed_str)
