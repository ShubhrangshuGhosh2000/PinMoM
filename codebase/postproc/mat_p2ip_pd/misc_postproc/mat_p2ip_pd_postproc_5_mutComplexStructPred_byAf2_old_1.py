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
    cuda_index = kwargs.get('cuda_index'); pdb_file_location = kwargs.get('pdb_file_location')
    af2_num_recycle = kwargs.get('af2_num_recycle'); af2_num_ensemble = kwargs.get('af2_num_ensemble')
    af2_num_seeds = kwargs.get('af2_num_seeds'); af2_use_amber = kwargs.get('af2_use_amber'); af2_overwrite_existing_results = kwargs.get('af2_overwrite_existing_results') 
    inp_dir_mut_seq_struct_pred_af2 = kwargs.get('inp_dir_mut_seq_struct_pred_af2')
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

    # Search in all the CSV files with the specific name pattern 'af2inp_cmpl_{dim_prot_complx_nm}_*.csv' in the inp_dir_mut_seq_struct_pred_af2
    af2_specific_name_pattern = os.path.join(inp_dir_mut_seq_struct_pred_af2, f'af2inp_cmpl_{dim_prot_complx_nm}_*.csv')
    mut_seq_inp_csv_files_lst_for_chain_combo = glob.glob(af2_specific_name_pattern, recursive=False)
    # Find the number of resulting csv files
    num_mut_seq_inp_csv_files = len(mut_seq_inp_csv_files_lst_for_chain_combo)
    mut_complx_struct_pred_postproc_5_info_lst.append('num_mut_seq_inp_csv_files'); mut_complx_struct_pred_postproc_5_val_lst.append(f'{num_mut_seq_inp_csv_files}')
    # If the number of resulting csv files is zero, return
    if(num_mut_seq_inp_csv_files == 0):
        print(f"\n!!! {dim_prot_complx_tag} Found no CSV files with the specific name pattern 'af2inp_cmpl_{dim_prot_complx_nm}_*.csv' in the {inp_dir_mut_seq_struct_pred_af2}\
              \nSo, NOT proceeding further for the current dimeric complex: {dim_prot_complx_nm}\n")
        mut_complx_struct_pred_postproc_5_info_lst.append('status'); mut_complx_struct_pred_postproc_5_val_lst.append('ERROR')
        # save tracking info records and continue with the next dimeric complex
        mut_complx_struct_pred_postproc_5_info_df = pd.DataFrame({'misc_info': mut_complx_struct_pred_postproc_5_info_lst, 'misc_info_val': mut_complx_struct_pred_postproc_5_val_lst})
        mut_complx_struct_pred_postproc_5_info_df.to_csv(os.path.join(curnt_dim_complx_mut_complx_struct_pred_result_dir, 'mut_complx_struct_pred_postproc_5.csv'), index=False)
        return  # return back
    # end of if block: if(num_mut_seq_inp_csv_files == 0):

    print(f"\n{dim_prot_complx_tag} There are {num_mut_seq_inp_csv_files} CSV file(s) with the specific name pattern 'af2inp_cmpl_{dim_prot_complx_nm}_*.csv' in the {inp_dir_mut_seq_struct_pred_af2} \
           \n Parsing every CSV file with the specific name pattern 'af2inp_cmpl_{dim_prot_complx_nm}_*.csv' inside it in a loop ...")
    print(f'But Before that, extracting the chain sequences of the original dimeric protein complex from the PDB file ({dim_prot_complx_nm}.pdb) which will be used later.')
    orig_complex_chain_seq_dict = prot_design_util.extract_chain_sequences(dim_prot_complx_nm, pdb_file_location)

    for indiv_mut_seq_inp_csv_file_name in mut_seq_inp_csv_files_lst_for_chain_combo:
        print(f'\n ###### {dim_prot_complx_tag} indiv_mut_seq_inp_csv_file_name: {indiv_mut_seq_inp_csv_file_name}')
        chain_combo_nm = indiv_mut_seq_inp_csv_file_name.split('/')[-1].replace(f'af2inp_cmpl_{dim_prot_complx_nm}_', '').replace('.csv', '')
        print(f'\n {dim_prot_complx_tag} chain_combo_nm: {chain_combo_nm}')

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
        
        # Read the indiv_mut_seq_inp_csv_file_name in a df
        indiv_mut_seq_inp_df = pd.read_csv(indiv_mut_seq_inp_csv_file_name)
        # Prepend fixed_chain_seq to each entry of the 'sequence' column
        print(f'{dim_prot_complx_tag} Prepending fixed_chain_seq to each entry of the "sequence" column.')
        indiv_mut_seq_inp_df['sequence'] = str(fixed_chain_seq) + ':' + indiv_mut_seq_inp_df['sequence'].astype(str)
        # save this dataframe as it will be used by AlphaFold2 for mutated complex structure prediction
        csv_fl_name_parts_lst = indiv_mut_seq_inp_csv_file_name.split('/')[-1].split('_')  # e.g. af2inp,cmpl,2I25,fpi,L,mpi,N.csv for 'af2inp_cmpl_2I25_fpi_L_mpi_N.csv'
        mut_complx_struct_pred_inp_csv_fl_nm = os.path.join(inp_dir_mut_complx_struct_pred_af2, f"{csv_fl_name_parts_lst[0]}_mut_{'_'.join(csv_fl_name_parts_lst[1:])}")
        indiv_mut_seq_inp_df.to_csv(mut_complx_struct_pred_inp_csv_fl_nm, index=False)

        # Create chain-combo specific output directory
        crnt_chain_combo_mut_complx_struct_pred_result_dir = os.path.join(curnt_dim_complx_mut_complx_struct_pred_result_dir, chain_combo_nm)
        PPIPUtils.createFolder(crnt_chain_combo_mut_complx_struct_pred_result_dir, recreate_if_exists=False)

        print(f'{dim_prot_complx_tag} ########################################## Calling AlphaFold2 for chain combo: {chain_combo_nm} -Start')
        # Build command for the AlphaFold2 invocation -Start
        af2_invoc_cmd = ["colabfold_batch"]
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
        # Build command for the AlphaFold2 invocation -End

        # Set environment variable CUDA_VISIBLE_DEVICES
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_index)

        # Execute the af2_invoc_cmd programmatically
        result = subprocess.run(af2_invoc_cmd, capture_output=True, text=True, env=env)
        
        # print the output of the AlphaFold2 invocation
        print("\nOutput of the AlphaFold2 invocation :")
        print(result.stdout)
        print(f'{dim_prot_complx_tag} ########################################## Calling AlphaFold2 for chain combo: {chain_combo_nm} -End')
    # end of for loop: for indiv_mut_seq_inp_csv_file_name in mut_seq_inp_csv_files_lst_for_chain_combo:
    # save tracking info records
    mut_complx_struct_pred_postproc_5_info_df = pd.DataFrame({'misc_info': mut_complx_struct_pred_postproc_5_info_lst, 'misc_info_val': mut_complx_struct_pred_postproc_5_val_lst})
    mut_complx_struct_pred_postproc_5_info_df.to_csv(os.path.join(curnt_dim_complx_mut_complx_struct_pred_result_dir, 'mut_complx_struct_pred_postproc_5.csv'), index=False)
    print('inside run_postproc_5_mut_complx_struct_pred_by_af2() method - End')
    
