import os, sys

from pathlib import Path
path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
from Bio.PDB import PDBParser
import glob
import pandas as pd
import subprocess

from utils import PPIPUtils, prot_design_util


def run_postproc_5_mut_complx_struct_pred_by_af2(**kwargs):
    """Postprocessing stage where mutated complex structure is predicted by AlphaFold2.

    This method is called from a triggering method like mat_p2ip_pd_postproc_trigger.trigger_pd_postproc().
    """
    print('inside run_postproc_5_mut_complx_struct_pred_by_af2() method - Start')
    print('####################################')
    # Iterate over kwargs and raise ValueError if any of the input arguments (except a few) is None. Also print each keyword argument name and respective value.
    for arg_name, arg_value in kwargs.items():
        if(arg_value is None):
            raise ValueError(f"Argument '{arg_name}' must be provided with a value.")
        print(f"'{arg_name}': {arg_value}")
    # end of for loop: for arg_name, arg_value in kwargs.items():

    # retrieve all the keyword arguments
    root_path = kwargs.get('root_path'); dim_prot_complx_nm = kwargs.get('dim_prot_complx_nm')
    cuda_index = kwargs.get('cuda_index')
    postproc_result_dir = kwargs.get('postproc_result_dir'); pdb_file_location = kwargs.get('pdb_file_location')
    max_num_of_seq_below_rmsd_thrshld_chain_struct = kwargs.get('max_num_of_seq_below_rmsd_thrshld_chain_struct')
    af2_exec_mode = kwargs.get('af2_exec_mode')
    af2_num_recycle = kwargs.get('af2_num_recycle'); af2_num_ensemble = kwargs.get('af2_num_ensemble')
    af2_num_seeds = kwargs.get('af2_num_seeds'); af2_use_amber = kwargs.get('af2_use_amber'); af2_overwrite_existing_results = kwargs.get('af2_overwrite_existing_results') 
    inp_dir_mut_complx_struct_pred_af2 = kwargs.get('inp_dir_mut_complx_struct_pred_af2')
    out_dir_mut_complx_struct_pred_af2 = kwargs.get('out_dir_mut_complx_struct_pred_af2')
    print('####################################')

    dim_prot_complx_tag = f' Postproc_5: complex_{dim_prot_complx_nm}:: '
    print(f'\ndim_prot_complx_tag: {dim_prot_complx_tag}')
    # Create the AlphaFold2 predicted mutated complex structure result directory for the current dimeric complex 
    print(f'\n{dim_prot_complx_tag} Creating the AlphaFold2 predicted mutated complex structure result directory for the current dimeric complex.')
    curnt_dim_complx_mut_complx_struct_pred_result_dir = os.path.join(out_dir_mut_complx_struct_pred_af2, f'{dim_prot_complx_nm}')
    PPIPUtils.createFolder(curnt_dim_complx_mut_complx_struct_pred_result_dir, recreate_if_exists=False)
    # creating few lists to track the mutated complex structure(s) prediction by AlphaFold2
    mut_complx_struct_pred_postproc_5_info_lst, mut_complx_struct_pred_postproc_5_val_lst = [], []

    # First check whether the dimeric protein complex name is present in the 'overall_accept_chain_struct_comp_res_postproc_3.csv' file.
    print(f'\n{dim_prot_complx_tag} First check whether the dimeric protein complex name ({dim_prot_complx_nm}) is present in the "overall_accept_chain_struct_comp_res_postproc_3.csv" file')
    overall_accept_chain_struct_comp_res_csv = os.path.join(postproc_result_dir, 'overall_accept_chain_struct_comp_res_postproc_3.csv')
    overall_accept_chain_struct_comp_res_df = pd.read_csv(overall_accept_chain_struct_comp_res_csv)
    complx_nm_filtered_df = overall_accept_chain_struct_comp_res_df[overall_accept_chain_struct_comp_res_df['cmplx'] == dim_prot_complx_nm]
    complx_nm_filtered_df = complx_nm_filtered_df.reset_index(drop=True)
    
    # Find the number of entries in the filtered dataframe
    num_of_rows_complx_nm_filtered_df = complx_nm_filtered_df.shape[0]
    mut_complx_struct_pred_postproc_5_info_lst.append('num_of_rows_complx_nm_filtered_df'); mut_complx_struct_pred_postproc_5_val_lst.append(f'{num_of_rows_complx_nm_filtered_df}')
    # If the number of entries is zero, return
    if(num_of_rows_complx_nm_filtered_df == 0):
        print(f"\n!!! {dim_prot_complx_tag} Found no entry for the dimeric complex name {dim_prot_complx_nm} in the {overall_accept_chain_struct_comp_res_csv}.\
              \nSo, NOT proceeding further for the current dimeric complex: {dim_prot_complx_nm}\n")
        mut_complx_struct_pred_postproc_5_info_lst.append('status'); mut_complx_struct_pred_postproc_5_val_lst.append('ERROR')
        # save tracking info records and continue with the next dimeric complex
        mut_complx_struct_pred_postproc_5_info_df = pd.DataFrame({'misc_info': mut_complx_struct_pred_postproc_5_info_lst, 'misc_info_val': mut_complx_struct_pred_postproc_5_val_lst})
        mut_complx_struct_pred_postproc_5_info_df.to_csv(os.path.join(curnt_dim_complx_mut_complx_struct_pred_result_dir, 'mut_complx_struct_pred_postproc_5.csv'), index=False)
        return  # return back
    # end of if block: if(num_of_rows_complx_nm_filtered_df == 0):

    print(f"\n{dim_prot_complx_tag} There is/are {num_of_rows_complx_nm_filtered_df} entry/entries for the dimeric complex name {dim_prot_complx_nm} in the {overall_accept_chain_struct_comp_res_csv}. \
           \n So, proceeding further for the current dimeric complex: {dim_prot_complx_nm}...")
    print(f'But Before that, extracting the chain sequences of the original dimeric protein complex from the PDB file ({dim_prot_complx_nm}.pdb) which will be used later.')
    orig_complex_chain_seq_dict = prot_design_util.extract_chain_sequences(dim_prot_complx_nm, pdb_file_location)
    
    # Group the complx_nm_filtered_df by 'cmplx', 'fpi', and 'mpi' columns
    grouped = complx_nm_filtered_df.groupby(['cmplx', 'fpi', 'mpi'])
    # Create a list of dataframes based on the groups
    cmplx_fpi_mpi_grouped_df_list = []
    for _, cmplx_fpi_mpi_grouped_df in grouped:
        # Sort cmplx_fpi_mpi_grouped_df in the ascending order of the Backbone-RMSD values
        cmplx_fpi_mpi_grouped_df = cmplx_fpi_mpi_grouped_df.sort_values(by=['bb_rmsd_superimp'], ascending=True)
        cmplx_fpi_mpi_grouped_df = cmplx_fpi_mpi_grouped_df.reset_index(drop=True)
        cmplx_fpi_mpi_grouped_df_list.append(cmplx_fpi_mpi_grouped_df)
    
    # Iterate cmplx_fpi_mpi_grouped_df_list
    for cmplx_fpi_mpi_grouped_df in cmplx_fpi_mpi_grouped_df_list:
        fpi = cmplx_fpi_mpi_grouped_df.at[0, 'fpi']
        mpi = cmplx_fpi_mpi_grouped_df.at[0, 'mpi']
        print(f'\n ###### {dim_prot_complx_tag} fpi: {fpi} :: mpi: {mpi}')
        chain_combo_nm = f'fpi_{fpi}_mpi_{mpi}'
        print(f'\n {dim_prot_complx_tag} chain_combo_nm: {chain_combo_nm}')

        # Find the number of AF2 prediction eligible mutated chain structure candidate(s) for the given chain-combo
        num_of_af2_pred_eligbl_cand = cmplx_fpi_mpi_grouped_df.shape[0]
        if(num_of_af2_pred_eligbl_cand > max_num_of_seq_below_rmsd_thrshld_chain_struct):
            # if the number of eligible candidates is more than the max threshold, clip the count.
            num_of_af2_pred_eligbl_cand = max_num_of_seq_below_rmsd_thrshld_chain_struct
        print(f'\n {dim_prot_complx_tag} num_of_af2_pred_eligbl_cand: {num_of_af2_pred_eligbl_cand}')
        cmplx_fpi_mpi_grouped_df = cmplx_fpi_mpi_grouped_df.head(num_of_af2_pred_eligbl_cand)

        # Create chain-combo specific input csv file to be used by AlphaFold2 for mutated complex structure prediction
        # Each row in this csv file will have 'sequence' column value in the format [fixed chain sequence]:[mutated chain sequence]
        print(f'\n {dim_prot_complx_tag} Creating chain-combo specific input csv file to be used by AlphaFold2 for mutated complex structure prediction.\
              \n Each row in this csv file will have "sequence" column value in the format [fixed chain sequence]:[mutated chain sequence]')
        fixed_chain_nm = chain_combo_nm.split('_')[1]

        # #### Retrieve fixed chain sequence -Start
        print(f'{dim_prot_complx_tag} Retrieving fixed chain ({fixed_chain_nm}) sequence...')
        fixed_chain_seq = orig_complex_chain_seq_dict[fixed_chain_nm]
        print(f'{dim_prot_complx_tag} fixed_chain_seq ({fixed_chain_nm}): {fixed_chain_seq}')
        # #### Retrieve fixed chain sequence -End

        # Create a dataframe (indiv_mut_seq_inp_df) containing 2 columns, 'id' and 'sequence' from cmplx_fpi_mpi_grouped_df.
        # 'id' column will have sample value as 'cmplx_2I25_fpi_L_mpi_N_batch_idx_1001_simuln_idx_3005' and
        # 'sequence' column will have values from 'mutated_seq' column of cmplx_fpi_mpi_grouped_df.
        indiv_mut_seq_inp_df = pd.DataFrame()
        indiv_mut_seq_inp_df['id'] = cmplx_fpi_mpi_grouped_df.apply(lambda row: f"cmplx_{row['cmplx']}_fpi_{row['fpi']}_mpi_{row['mpi']}_batch_idx_{row['batch_idx']}_simuln_idx_{row['simuln_idx']}", axis=1)
        indiv_mut_seq_inp_df['sequence'] = cmplx_fpi_mpi_grouped_df['mutated_seq']
        indiv_mut_seq_inp_df = indiv_mut_seq_inp_df.reset_index(drop=True)

        # Prepend fixed_chain_seq to each entry of the 'sequence' column
        print(f'{dim_prot_complx_tag} Prepending fixed_chain_seq to each entry of the "sequence" column.')
        indiv_mut_seq_inp_df['sequence'] = str(fixed_chain_seq) + ':' + indiv_mut_seq_inp_df['sequence'].astype(str)
        # save this dataframe as it will be used by AlphaFold2 for mutated complex structure prediction
        csv_fl_name_parts_lst = ['af2inp', 'cmpl', dim_prot_complx_nm, 'fpi', fpi, 'mpi', f'{mpi}.csv'] # e.g. af2inp,cmpl,2I25,fpi,L,mpi,N.csv for 'af2inp_cmpl_2I25_fpi_L_mpi_N.csv'
        mut_complx_struct_pred_inp_csv_fl_nm = os.path.join(inp_dir_mut_complx_struct_pred_af2, f"{csv_fl_name_parts_lst[0]}_mut_{'_'.join(csv_fl_name_parts_lst[1:])}")
        indiv_mut_seq_inp_df.to_csv(mut_complx_struct_pred_inp_csv_fl_nm, index=False)

        # Create chain-combo specific output directory
        crnt_chain_combo_mut_complx_struct_pred_result_dir = os.path.join(curnt_dim_complx_mut_complx_struct_pred_result_dir, chain_combo_nm)
        PPIPUtils.createFolder(crnt_chain_combo_mut_complx_struct_pred_result_dir, recreate_if_exists=False)

        print(f'{dim_prot_complx_tag} ########################################## Calling AlphaFold2 for chain combo: {chain_combo_nm} -Start')
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
            af2_invoc_cmd.append(mut_complx_struct_pred_inp_csv_fl_nm)
            # Set outut result directory (predicted structure in pdb format) in the af2_invoc_cmd
            af2_invoc_cmd.append(crnt_chain_combo_mut_complx_struct_pred_result_dir)

        elif(af2_exec_mode == 'msa_gen'):
            af2_invoc_cmd.append("--msa-only")
            af2_invoc_cmd.extend(["--host-url", 'https://api.colabfold.com'])
            # Set input csv file name with location in the af2_invoc_cmd
            af2_invoc_cmd.append(mut_complx_struct_pred_inp_csv_fl_nm)
            # Set outut result directory (predicted structure in pdb format) in the af2_invoc_cmd
            af2_invoc_cmd.append(crnt_chain_combo_mut_complx_struct_pred_result_dir)
        
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
            af2_invoc_cmd.append(crnt_chain_combo_mut_complx_struct_pred_result_dir)
            # Set outut result directory which will contain predicted structure in pdb format in the af2_invoc_cmd
            af2_invoc_cmd.append(crnt_chain_combo_mut_complx_struct_pred_result_dir)
        # end of if-else block
        # Build command for the AlphaFold2 invocation based on the execution mode -End

        # Set environment variable CUDA_VISIBLE_DEVICES
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_index)

        # Execute the af2_invoc_cmd programmatically
        result = subprocess.run(af2_invoc_cmd, capture_output=True, text=True, env=env)
        
        # print the output of the AlphaFold2 invocation
        print("\nOutput of the AlphaFold2 invocation :")
        print(result.stdout)
        print(f'{dim_prot_complx_tag} ########################################## Calling AlphaFold2 for chain combo: {chain_combo_nm} -End')
    # end of for loop: for cmplx_fpi_mpi_grouped_df in cmplx_fpi_mpi_grouped_df_list:
    # save tracking info records
    mut_complx_struct_pred_postproc_5_info_df = pd.DataFrame({'misc_info': mut_complx_struct_pred_postproc_5_info_lst, 'misc_info_val': mut_complx_struct_pred_postproc_5_val_lst})
    mut_complx_struct_pred_postproc_5_info_df.to_csv(os.path.join(curnt_dim_complx_mut_complx_struct_pred_result_dir, 'mut_complx_struct_pred_postproc_5.csv'), index=False)
    print('inside run_postproc_5_mut_complx_struct_pred_by_af2() method - End')

