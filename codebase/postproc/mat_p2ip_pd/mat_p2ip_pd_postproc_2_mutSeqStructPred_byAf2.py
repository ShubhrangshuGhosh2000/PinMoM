import os, sys

from pathlib import Path
path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
import glob
import pandas as pd
import subprocess
from utils import PPIPUtils


def run_postproc_2_mut_seq_struct_pred_by_af2(**kwargs):
    """Postprocessing stage where mutated sequence structure(s) is predicted by AlphaFold2.

    This method is called from a triggering method like mat_p2ip_pd_postproc_trigger.trigger_pd_postproc().
    """
    print('inside run_postproc_2_mut_seq_struct_pred_by_af2() method - Start')
    print('####################################')
    # Iterate over kwargs and raise ValueError if any of the input arguments (except a few) is None. Also print each keyword argument name and respective value.
    for arg_name, arg_value in kwargs.items():
        if(arg_value is None):
            raise ValueError(f"Argument '{arg_name}' must be provided with a value.")
        print(f"'{arg_name}': {arg_value}")
    # end of for loop: for arg_name, arg_value in kwargs.items():

    # retrieve all the keyword arguments
    root_path = kwargs.get('root_path'); dim_prot_complx_nm = kwargs.get('dim_prot_complx_nm')
    cuda_index = kwargs.get('cuda_index'); af2_use_scwrl = kwargs.get('af2_use_scwrl'); scwrl_exec_path = kwargs.get('scwrl_exec_path')
    af2_exec_mode = kwargs.get('af2_exec_mode')
    af2_num_recycle = kwargs.get('af2_num_recycle'); af2_num_ensemble = kwargs.get('af2_num_ensemble')
    af2_num_seeds = kwargs.get('af2_num_seeds'); af2_use_amber = kwargs.get('af2_use_amber'); af2_overwrite_existing_results = kwargs.get('af2_overwrite_existing_results') 
    inp_dir_mut_seq_struct_pred_af2 = kwargs.get('inp_dir_mut_seq_struct_pred_af2')
    out_dir_mut_seq_struct_pred_af2 = kwargs.get('out_dir_mut_seq_struct_pred_af2')
    print('####################################')

    dim_prot_complx_tag = f' Postproc_2: complex_{dim_prot_complx_nm}:: '
    print(f'\ndim_prot_complx_tag: {dim_prot_complx_tag}')
    # Create the AlphaFold2 predicted mutated sequence structure result directory for the current dimeric complex 
    print(f'\n{dim_prot_complx_tag} Creating the AlphaFold2 predicted mutated sequence structure result directory for the current dimeric complex.')
    curnt_dim_complx_mut_seq_struct_pred_result_dir = os.path.join(out_dir_mut_seq_struct_pred_af2, f'{dim_prot_complx_nm}')
    PPIPUtils.createFolder(curnt_dim_complx_mut_seq_struct_pred_result_dir, recreate_if_exists=False)
    # creating few lists to track the mutated sequence structure(s) prediction by AlphaFold2
    mut_seq_struct_pred_postproc_2_info_lst, mut_seq_struct_pred_postproc_2_val_lst = [], []

    # Search in all the CSV files with the specific name pattern 'af2inp_cmpl_{dim_prot_complx_nm}_*.csv' in the inp_dir_mut_seq_struct_pred_af2
    af2_specific_name_pattern = os.path.join(inp_dir_mut_seq_struct_pred_af2, f'af2inp_cmpl_{dim_prot_complx_nm}_*.csv')
    mut_seq_inp_csv_files_lst_for_chain_combo = glob.glob(af2_specific_name_pattern, recursive=False)
    # Find the number of resulting csv files
    num_mut_seq_inp_csv_files = len(mut_seq_inp_csv_files_lst_for_chain_combo)
    mut_seq_struct_pred_postproc_2_info_lst.append('num_mut_seq_inp_csv_files'); mut_seq_struct_pred_postproc_2_val_lst.append(f'{num_mut_seq_inp_csv_files}')
    # If the number of resulting csv files is zero, return
    if(num_mut_seq_inp_csv_files == 0):
        print(f"\n!!! {dim_prot_complx_tag} Found no CSV files with the specific name pattern 'af2inp_cmpl_{dim_prot_complx_nm}_*.csv' in the {inp_dir_mut_seq_struct_pred_af2}\
              \nSo, NOT proceeding further for the current dimeric complex: {dim_prot_complx_nm}\n")
        mut_seq_struct_pred_postproc_2_info_lst.append('status'); mut_seq_struct_pred_postproc_2_val_lst.append('ERROR')
        # save tracking info records and continue with the next dimeric complex
        mut_seq_struct_pred_postproc_2_info_df = pd.DataFrame({'misc_info': mut_seq_struct_pred_postproc_2_info_lst, 'misc_info_val': mut_seq_struct_pred_postproc_2_val_lst})
        mut_seq_struct_pred_postproc_2_info_df.to_csv(os.path.join(curnt_dim_complx_mut_seq_struct_pred_result_dir, 'mut_seq_struct_pred_postproc_2.csv'), index=False)
        return  # return back
    # end of if block: if(num_mut_seq_inp_csv_files == 0):

    print(f"\n{dim_prot_complx_tag} There are {num_mut_seq_inp_csv_files} CSV file(s) with the specific name pattern 'af2inp_cmpl_{dim_prot_complx_nm}_*.csv' in the {inp_dir_mut_seq_struct_pred_af2} \
           \n Parsing every CSV file with the specific name pattern 'af2inp_cmpl_{dim_prot_complx_nm}_*.csv' inside it in a loop ...")
    for indiv_mut_seq_inp_csv_file_name in mut_seq_inp_csv_files_lst_for_chain_combo:
        print(f'\n ###### {dim_prot_complx_tag} indiv_mut_seq_inp_csv_file_name: {indiv_mut_seq_inp_csv_file_name}')
        chain_combo_nm = indiv_mut_seq_inp_csv_file_name.split('/')[-1].replace(f'af2inp_cmpl_{dim_prot_complx_nm}_', '').replace('.csv', '')
        print(f'\n {dim_prot_complx_tag} chain_combo_nm: {chain_combo_nm}')
        # Create chain-combo specific output directory
        crnt_cmplx_chain_combo_mut_seq_struct_pred_result_dir = os.path.join(curnt_dim_complx_mut_seq_struct_pred_result_dir, chain_combo_nm)
        PPIPUtils.createFolder(crnt_cmplx_chain_combo_mut_seq_struct_pred_result_dir, recreate_if_exists=False)
        if( not af2_use_scwrl):  # If Scwrl is not used, then AF2 is used
            print(f'{dim_prot_complx_tag} ########################################## Calling AlphaFold2 for chain combo: {chain_combo_nm} -Start')
            print(f'{dim_prot_complx_tag} af2_exec_mode: {af2_exec_mode}')
            
            # Build command for the AlphaFold2 invocation based on the execution mode -Start
            af2_invoc_cmd = ["colabfold_batch"]

            if(af2_exec_mode == 'msa_pred_gen'):
                af2_invoc_cmd.extend(["--num-recycle", str(af2_num_recycle)])
                af2_invoc_cmd.extend(["--num-ensemble", str(af2_num_ensemble)])
                af2_invoc_cmd.extend(["--num-seeds", str(af2_num_seeds)])

                if(af2_use_amber):
                    af2_invoc_cmd.append("--amber")
                    af2_invoc_cmd.extend(["--num-relax", str(1)])
                    af2_invoc_cmd.append("--use-gpu-relax")
                
                if(af2_overwrite_existing_results):
                    af2_invoc_cmd.append("--overwrite-existing-results")
                # Set input csv file name with location in the af2_invoc_cmd
                af2_invoc_cmd.append(indiv_mut_seq_inp_csv_file_name)
                # Set outut result directory (predicted structure in pdb format) in the af2_invoc_cmd
                af2_invoc_cmd.append(crnt_cmplx_chain_combo_mut_seq_struct_pred_result_dir)
            
            elif(af2_exec_mode == 'msa_gen'):
                af2_invoc_cmd.append("--msa-only")
                af2_invoc_cmd.extend(["--host-url", 'https://api.colabfold.com'])
                # Set input csv file name with location in the af2_invoc_cmd
                af2_invoc_cmd.append(indiv_mut_seq_inp_csv_file_name)
                # Set outut result directory (predicted structure in pdb format) in the af2_invoc_cmd
                af2_invoc_cmd.append(crnt_cmplx_chain_combo_mut_seq_struct_pred_result_dir)
            
            elif(af2_exec_mode == 'pred_a4_msa_gen'):
                af2_invoc_cmd.extend(["--num-recycle", str(af2_num_recycle)])
                af2_invoc_cmd.extend(["--num-ensemble", str(af2_num_ensemble)])
                af2_invoc_cmd.extend(["--num-seeds", str(af2_num_seeds)])

                if(af2_use_amber):
                    af2_invoc_cmd.append("--amber")
                    af2_invoc_cmd.extend(["--num-relax", str(1)])
                    af2_invoc_cmd.append("--use-gpu-relax")
                
                if(af2_overwrite_existing_results):
                    af2_invoc_cmd.append("--overwrite-existing-results")
                # Set output result directory which contains pre-generated MSA files in the af2_invoc_cmd
                af2_invoc_cmd.append(crnt_cmplx_chain_combo_mut_seq_struct_pred_result_dir)
                # Set outut result directory which will contain predicted structure in pdb format in the af2_invoc_cmd
                af2_invoc_cmd.append(crnt_cmplx_chain_combo_mut_seq_struct_pred_result_dir)
            # end of if-else block
            # Build command for the AlphaFold2 invocation based on ethe xecution mode -End

            # Set environment variable CUDA_VISIBLE_DEVICES
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(cuda_index)

            # Execute the af2_invoc_cmd programmatically
            result = subprocess.run(af2_invoc_cmd, capture_output=True, text=True, env=env)
            
            # print the output of the AlphaFold2 invocation
            print("\nOutput of the AlphaFold2 invocation :")
            print(result.stdout)
            print(f'{dim_prot_complx_tag} ########################################## Calling AlphaFold2 for chain combo: {chain_combo_nm} -End')
        else:
            # Use Scwrl
            print(f'{dim_prot_complx_tag} ########################################## Calling Scwrl for chain combo: {chain_combo_nm} -Start')
            mut_chain_id = chain_combo_nm.split('_')[-1]
            print(f'\n {dim_prot_complx_tag} mut_chain_id: {mut_chain_id}')
            # orig_chain_pdb_path will be used with -i option of 'Scwrl4' command
            orig_chain_pdb_path = os.path.join(root_path, 'dataset/preproc_data/pdb_chain_files', f"{dim_prot_complx_nm}_{mut_chain_id}.pdb")

            # read the csv file (indiv_mut_seq_inp_csv_file_name) into a df
            indiv_mut_seq_inp_df = pd.read_csv(indiv_mut_seq_inp_csv_file_name)
            print(f'\n {dim_prot_complx_tag} Iterating {indiv_mut_seq_inp_csv_file_name} row-wise...')
            # iterate the df row-wise in an inner loop
            for index, row in indiv_mut_seq_inp_df.iterrows():
                id = row['id']; seq = row['sequence']
                print(f'\n {dim_prot_complx_tag} iteration index: {index} :: id: {id} :: seq: {seq}')
                # sample id: cmplx_2I25_fpi_L_mpi_N_batch_idx_2507_simuln_idx_7523
                # sample output file name: cmplx_2I25_fpi_L_mpi_N_batch_idx_2507_simuln_idx_7523_relaxed_rank_001_alphafold2_ptm_model_1_seed_000.pdb
                relaxed_unrelaxed = 'unrelaxed'
                if(af2_use_amber):
                    relaxed_unrelaxed = 'relaxed'
                scwrl_output_file_name = f'{id}_{relaxed_unrelaxed}_rank_001_alphafold2_ptm_model_1_seed_000.pdb'
                # scwrl_output_file_name_with_loc will be used with -o option of 'Scwrl4' command
                scwrl_output_file_name_with_loc = os.path.join(crnt_cmplx_chain_combo_mut_seq_struct_pred_result_dir, scwrl_output_file_name)
                
                # Create a temporary mutated sequence file and write the mutated sequence there
                # temp_mut_seq_file will be used with -i option of 'Scwrl4' command
                temp_mut_seq_file = os.path.join(root_path, 'temp_Scwrl4', f'{id}.txt')
                f = open(temp_mut_seq_file,'w')
                f.write(seq)
                f.close()

                # # log_file_nm will be used for logging during 'Scwrl4' command execution
                # log_file_nm = os.path.join(root_path, 'temp_Scwrl4', f'{id}_log.txt')
                
                # Build command for the Scwrl invocation -Start
                scwrl_invoc_cmd = [f'{scwrl_exec_path}Scwrl4']
                scwrl_invoc_cmd.extend(["-i", str(orig_chain_pdb_path)])
                scwrl_invoc_cmd.extend(["-s", str(temp_mut_seq_file)])
                scwrl_invoc_cmd.extend(["-o", str(scwrl_output_file_name_with_loc)])
                # Build command for the Scwrl invocation -End

                # Execute the scwrl_invoc_cmd programmatically
                result = subprocess.run(scwrl_invoc_cmd, capture_output=True, text=True)
                
                # print the output of the Scwrl invocation
                print("\nOutput of the Scwrl invocation :")
                print(result.stdout)
            # end of for loop: for index, row in indiv_mut_seq_inp_df.iterrows():
            print(f'{dim_prot_complx_tag} ########################################## Calling Scwrl for chain combo: {chain_combo_nm} -End')
        # end of if-else block: if( not af2_use_scwrl):
    # end of for loop: for indiv_mut_seq_inp_csv_file_name in mut_seq_inp_csv_files_lst_for_chain_combo:
    # save tracking info records
    mut_seq_struct_pred_postproc_2_info_df = pd.DataFrame({'misc_info': mut_seq_struct_pred_postproc_2_info_lst, 'misc_info_val': mut_seq_struct_pred_postproc_2_val_lst})
    mut_seq_struct_pred_postproc_2_info_df.to_csv(os.path.join(curnt_dim_complx_mut_seq_struct_pred_result_dir, 'mut_seq_struct_pred_postproc_2.csv'), index=False)
    print('inside run_postproc_2_mut_seq_struct_pred_by_af2() method - End')



# if __name__ == '__main__':
    
#     # ##################################### TO SELECT ONLY A FEW CANDIDATES FOR AF2 BASED MUTATED SEQUENCE STRUCTURE PREDICTION -START #####################################
#     import os
#     import pandas as pd
#     import glob 

#     root_path = '/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/matpip_pd_prj'
#     iteration_tag = 'mcmc_fullLen_puFalse_batch5_mutPrcntLen10'
#     FEWER_NUM_OF_CANDIDATES = 30

#     inp_dir_mut_seq_struct_pred_af2 = os.path.join(root_path, "dataset/postproc_data/alphafold2_io", iteration_tag, "mut_seq_struct_pred_inp")

#     # Search in all the CSV files with the specific name pattern 'af2inp_cmpl_{dim_prot_complx_nm}_*.csv' in the inp_dir_mut_seq_struct_pred_af2
#     af2_specific_name_pattern = os.path.join(inp_dir_mut_seq_struct_pred_af2, f'af2inp_cmpl_*.csv')
#     mut_seq_inp_csv_files_lst_for_chain_combo = glob.glob(af2_specific_name_pattern, recursive=False)

#     for indiv_mut_seq_inp_csv_file_name in mut_seq_inp_csv_files_lst_for_chain_combo:
#         indiv_mut_seq_inp_df = pd.read_csv(indiv_mut_seq_inp_csv_file_name)
#         # ###############################################################################
#         mod_indiv_mut_seq_inp_df = indiv_mut_seq_inp_df.head(FEWER_NUM_OF_CANDIDATES)
#         # ###############################################################################
#         # save orig indiv_mut_seq_inp_df
#         orig_file_nm_path = indiv_mut_seq_inp_csv_file_name.replace('.csv', '.csv_orig')
#         indiv_mut_seq_inp_df.to_csv(orig_file_nm_path, index=False)

#         # save mod_indiv_mut_seq_inp_df
#         mod_indiv_mut_seq_inp_df.to_csv(indiv_mut_seq_inp_csv_file_name, index=False)
#     # End of for loop
#     # ##################################### TO SELECT ONLY A FEW CANDIDATES FOR AF2 BASED MUTATED SEQUENCE STRUCTURE PREDICTION -END #####################################

