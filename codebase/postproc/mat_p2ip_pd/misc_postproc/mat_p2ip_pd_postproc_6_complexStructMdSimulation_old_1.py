import os, sys

from pathlib import Path
path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
import glob
import pandas as pd
import shutil
from utils import prot_design_util, MD_Simulation


def run_postproc_6_complx_struct_mdSim(**kwargs):
    """Postprocessing stage where Molecular Dynamics (MD) simulation is carried out on the protein complex structure.

    This method is called from a triggering method like mat_p2ip_pd_postproc_trigger.trigger_pd_postproc().
    """
    print('inside run_postproc_6_complx_struct_mdSim() method - Start')

    # ######### IMPORTANT: The following unix commands must be executed in the runtime environment (bash) before invoking MD simulation related python script - Start #########
    # ### module load apps/gromacs/2022/gpu  # For Paramvidya: module load apps/gromacs/16.6.2022/intel
    # ### source /home/apps/gromacs/gromacs-2022.2/installGPUIMPI/bin/GMXRC  # For Paramvidya: source /home/apps/gromacs/gromacs-2022.2/installGPUIOMPI/bin/GMXRC
    # ### export PATH=/home/pralaycs/miniconda3/envs/py3114_torch_gpu_param/bin:$PATH
    # ######### IMPORTANT: The following unix commands must be executed in the runtime environment (bash) before invoking MD simulation related python script - End #########

    print('####################################')
    # Iterate over kwargs and raise ValueError if any of the input arguments (except a few) is None. Also print each keyword argument name and respective value.
    for arg_name, arg_value in kwargs.items():
        if(arg_value is None):
            raise ValueError(f"Argument '{arg_name}' must be provided with a value.")
        print(f"'{arg_name}': {arg_value}")
    # end of for loop: for arg_name, arg_value in kwargs.items():

    # retrieve all the keyword arguments
    root_path = kwargs.get('root_path'); dim_prot_complx_nm = kwargs.get('dim_prot_complx_nm')
    cuda_index_mdSim = kwargs.get('cuda_index_mdSim')
    postproc_result_dir = kwargs.get('postproc_result_dir'); pdb_file_location = kwargs.get('pdb_file_location')
    af2_use_amber = kwargs.get('af2_use_amber')
    inp_dir_mut_complx_struct_pred_af2 = kwargs.get('inp_dir_mut_complx_struct_pred_af2'); out_dir_mut_complx_struct_pred_af2 = kwargs.get('out_dir_mut_complx_struct_pred_af2')
    mdSim_overwrite_existing_results = kwargs.get('mdSim_overwrite_existing_results');  forcefield_mdSim = kwargs.get('forcefield_mdSim')
    max_cadidate_count_mdSim = kwargs.get('max_cadidate_count_mdSim')
    mdSim_result_dir = kwargs.get('mdSim_result_dir') 
    print('####################################')

    # creating few lists to track the MD simulation
    md_sim_postproc_6_info_lst, md_sim_postproc_6_val_lst = [], []
    md_sim_avg_rmsd_rmsf_rg_dict_lst = []
    prot_complx_tag = f' Postproc_6: complex_{dim_prot_complx_nm} :: '

    # ######################## Check the availability of eligible MD Simulation candidate(s) for the original dimer -Start ########################
    overall_accept_complx_struct_csv = os.path.join(postproc_result_dir, 'overall_accept_complx_struct_comp_res_postproc_5_part2.csv')
    overall_accept_complx_struct_df = pd.read_csv(overall_accept_complx_struct_csv)

    # First, check whether there is any available MD Simulation candidate(s) for the original input protein complex
    print(f"\n!!! {prot_complx_tag} First, check whether there is any available MD Simulation candidate(s) for the original input protein complex.")
    # Search in all the CSV files with the specific name pattern 'af2inp_mut_cmpl_{dim_prot_complx_nm}_*.csv' in the inp_dir_mut_complx_struct_pred_af2
    af2_specific_name_pattern = os.path.join(inp_dir_mut_complx_struct_pred_af2, f'af2inp_mut_cmpl_{dim_prot_complx_nm}_*.csv')
    mut_complx_inp_csv_files_lst = glob.glob(af2_specific_name_pattern, recursive=False)
    # Find the number of resulting csv files
    num_mut_complx_inp_csv_files = len(mut_complx_inp_csv_files_lst)
    # If the number of resulting csv files is zero, return
    if(num_mut_complx_inp_csv_files == 0):
        print(f"\n!!! {prot_complx_tag} Found no CSV files with the specific name pattern 'af2inp_mut_cmpl_{dim_prot_complx_nm}_*.csv' in the {inp_dir_mut_complx_struct_pred_af2}\
              \nSo, NOT proceeding further for the current dimeric complex: {dim_prot_complx_nm}\n")
        md_sim_postproc_6_info_lst.append(f'{dim_prot_complx_nm}'); md_sim_postproc_6_val_lst.append(f'Error: Found no CSV files with the specific name pattern "af2inp_mut_cmpl_{dim_prot_complx_nm}_*.csv" in the {inp_dir_mut_complx_struct_pred_af2}')
        delta_change_df = pd.DataFrame({'complx_id': md_sim_postproc_6_info_lst, 'md_sim_status': md_sim_postproc_6_val_lst})
        update_md_sim_info_postproc_6_csv(mdSim_result_dir, delta_change_df)
        return  # return back
    # end of if block: if(num_mut_complx_inp_csv_files == 0):

    print(f"\n{prot_complx_tag} There are {num_mut_complx_inp_csv_files} CSV file(s) with the specific name pattern 'af2inp_mut_cmpl_{dim_prot_complx_nm}_*.csv' in the {inp_dir_mut_complx_struct_pred_af2} \
           \n Parsing every CSV file with the specific name pattern 'af2inp_mut_cmpl_{dim_prot_complx_nm}_*.csv' inside it in a loop ...")
    # The following flag indicates whether there is any MD Simulation eligible mutated protein complex candidate for the given dimer
    no_eligbl_md_sim_cand = True
    eligbl_mut_complx_inp_csv_file_nm_lst = []
    for indiv_mut_complx_inp_csv_file_name in mut_complx_inp_csv_files_lst:
        print(f'\n ###### {prot_complx_tag} indiv_mut_complx_inp_csv_file_name: {indiv_mut_complx_inp_csv_file_name}')
        chain_combo_nm = indiv_mut_complx_inp_csv_file_name.split('/')[-1].replace(f'af2inp_mut_cmpl_{dim_prot_complx_nm}_', '').replace('.csv', '')
        print(f'\n {prot_complx_tag} chain_combo_nm: {chain_combo_nm}')
        fpi = chain_combo_nm.split('_')[1]; mpi = chain_combo_nm.split('_')[-1]
        print(f'\n {prot_complx_tag} Checking whether there is any entry in "overall_accept_complx_struct_comp_res_postproc_5_part2.csv" file for the given protein complex: {dim_prot_complx_nm} and fpi: {fpi} and mpi: {mpi}')
        filtered_df = overall_accept_complx_struct_df.loc[(overall_accept_complx_struct_df['cmplx'] == dim_prot_complx_nm)
                                                        & (overall_accept_complx_struct_df['fpi'] == fpi)
                                                        & (overall_accept_complx_struct_df['mpi'] == mpi)]
        if(filtered_df.shape[0] == 0):
            # Number of rows is zero.
            print(f'\n {prot_complx_tag} There is no entry for this chain_combo ({chain_combo_nm}). Hence skipping this chain_combo.')
        else:
            # Number of rows > 0.
            no_eligbl_md_sim_cand = False
            print(f'\n {prot_complx_tag} There is/are {filtered_df.shape[0]} entry/entries for this chain_combo ({chain_combo_nm}).')
            eligbl_mut_complx_inp_csv_file_nm_lst.append(indiv_mut_complx_inp_csv_file_name)
    # end of for loop: for indiv_mut_complx_inp_csv_file_name in mut_complx_inp_csv_files_lst:

    # If there is not any MD Simulation eligible mutated protein complex candidate for the given dimer, then just skip this dimer
    if(no_eligbl_md_sim_cand):
        print(f'\n {prot_complx_tag} There is not any MD Simulation eligible mutated protein complex candidate for the given dimer.\
              \nSo, NOT proceeding further for the current dimeric complex: {dim_prot_complx_nm}\n')
        md_sim_postproc_6_info_lst.append(f'{dim_prot_complx_nm}'); md_sim_postproc_6_val_lst.append(f'Error: Found no MD Simulation eligible mutated protein complex candidate for the given dimer')
        delta_change_df = pd.DataFrame({'complx_id': md_sim_postproc_6_info_lst, 'md_sim_status': md_sim_postproc_6_val_lst})
        update_md_sim_info_postproc_6_csv(mdSim_result_dir, delta_change_df)
        return  # return back
    # end of if block: if(no_eligbl_md_sim_cand):
    
    print(f'\n {prot_complx_tag} #################### There is at least one MD Simulation eligible mutated protein complex candidate for the given dimer: {dim_prot_complx_nm}.\
          \nSo, proceeding further for the current dimeric complex: {dim_prot_complx_nm}\n')
    # ######################## Check the availability of eligible MD Simulation candidate(s) for the original dimer -End ########################

    # Next, execute MD simulation for the original protein complex
    print(f'First, execute MD simulation for the original protein complex: {dim_prot_complx_nm}')
    orig_prot_mdSim_result_dir = os.path.join(mdSim_result_dir, f'{dim_prot_complx_nm}', 'orig')
    mdSim_status = prot_design_util.create_mdSim_res_dir_and_return_status(orig_prot_mdSim_result_dir, recreate_if_exists = mdSim_overwrite_existing_results)
    if(mdSim_status == 'mdSim_res_already_exists'):
        print(f'################## Postproc_6: As the MD simulation result already exists for the original protein complex: {dim_prot_complx_nm} at \n {orig_prot_mdSim_result_dir}\n hence, skipping MD simulation for it...')
        md_sim_postproc_6_info_lst.append(f'{dim_prot_complx_nm} (orig)'); md_sim_postproc_6_val_lst.append('mdSim_res_already_exists')
    else:
        print(f'\nPostproc_6: Executing MD simulation for the original protein complex: {dim_prot_complx_nm}...')
        prot_complx_tag = f' Postproc_6: complex_{dim_prot_complx_nm} :orig:: '
        print(f'prot_complx_tag: {prot_complx_tag}')
        try:
            # call execute_md_simulation() method with some more input arguments
            kwargs['prot_complx_tag'] = prot_complx_tag
            kwargs['specific_inp_pdb_file'] = os.path.join(pdb_file_location, f'{dim_prot_complx_nm}.pdb')
            kwargs['specific_mdSim_result_dir'] = orig_prot_mdSim_result_dir
            avg_rmsd_nm, avg_rg_nm, avg_rmsf_nm = MD_Simulation.execute_md_simulation(**kwargs)
            # store avg_rmsd_nm, avg_rg_nm, avg_rmsf_nm into md_sim_avg_rmsd_rmsf_rg_dict
            md_sim_avg_rmsd_rmsf_rg_dict = {'complx_id': f'{dim_prot_complx_nm} : (orig)', 'avg_rmsd_nm': avg_rmsd_nm, 'avg_rg_nm': avg_rg_nm, 'avg_rmsf_nm': avg_rmsf_nm}
            # append the dict into md_sim_avg_rmsd_rmsf_rg_dict_lst
            md_sim_avg_rmsd_rmsf_rg_dict_lst.append(md_sim_avg_rmsd_rmsf_rg_dict)
            # update MD simulation status
            md_sim_postproc_6_info_lst.append(f'{dim_prot_complx_nm} : (orig)'); md_sim_postproc_6_val_lst.append(f'Success')
        except Exception as ex:
            print(f'\n{prot_complx_tag} ################## Error!! Error in MD simulation: {ex}')
            print(f'{prot_complx_tag} ################## Removing the result directory: {orig_prot_mdSim_result_dir}')
            shutil.rmtree(orig_prot_mdSim_result_dir, ignore_errors=False, onerror=None)
            md_sim_postproc_6_info_lst.append(f'{dim_prot_complx_nm} (orig)'); md_sim_postproc_6_val_lst.append(f'Error: {ex}')
        # end of try-catch block
    # end of if-else block: if(mdSim_status == 'mdSim_res_already_exists'):

    # Next, execute MD simulation for all the mutated protein complex candidates corresponding to the original one
    print(f'\n Next, execute MD simulation for all the eligible mutated protein complex candidates corresponding to the original: {dim_prot_complx_nm}')
    # As already it has been confirmed that there is at least one MD Simulation eligible mutated protein complex candidate for the given dimer, so skipping 
    # the validations.
    # Iterate over eligbl_mut_complx_inp_csv_file_nm_lst
    for indiv_eligbl_mut_complx_inp_csv_file_nm in eligbl_mut_complx_inp_csv_file_nm_lst:
        print(f'\n ###### {prot_complx_tag} indiv_eligbl_mut_complx_inp_csv_file_nm: {indiv_eligbl_mut_complx_inp_csv_file_nm}')
        chain_combo_nm = indiv_eligbl_mut_complx_inp_csv_file_nm.split('/')[-1].replace(f'af2inp_mut_cmpl_{dim_prot_complx_nm}_', '').replace('.csv', '')
        print(f'\n {prot_complx_tag} chain_combo_nm: {chain_combo_nm}')
        fpi = chain_combo_nm.split('_')[1]; mpi = chain_combo_nm.split('_')[-1]
        filtered_df = overall_accept_complx_struct_df.loc[(overall_accept_complx_struct_df['cmplx'] == dim_prot_complx_nm)
                                                        & (overall_accept_complx_struct_df['fpi'] == fpi)
                                                        & (overall_accept_complx_struct_df['mpi'] == mpi)]
        filtered_df = filtered_df.reset_index(drop=True)
        # Find the number of MD Simulation eligible mutated protein complex candidate(s) for the given csv file.
        num_of_md_sim_eligbl_cand = filtered_df.shape[0]
        if(num_of_md_sim_eligbl_cand > max_cadidate_count_mdSim):
            # if the number of eligible candidates is more than the max threshold, clip the count.
            num_of_md_sim_eligbl_cand = max_cadidate_count_mdSim
        print(f'\n {prot_complx_tag} num_of_md_sim_eligbl_cand: {num_of_md_sim_eligbl_cand}')
        filtered_df = filtered_df.head(num_of_md_sim_eligbl_cand)

        # As overall_accept_complx_struct_df is already sorted by 'bb_rmsd_superimp' column in ascending order, so is filtered_df.
        # Hence, no further sorting is required on filtered_df.
        print(f'\n\n ##### Iterating over the MD Simulation eligible mutated protein complex candidate(s) one by one.')
        for index, row in filtered_df.iterrows():
            mut_complx_id = row['id']; seq = row['sequence']
            print(f'\n {prot_complx_tag} iteration index: {index} :: mut_complx_id: {mut_complx_id} :: seq: {seq}')
            # sample mut_complx_id: cmplx_2I25_fpi_L_mpi_N_batch_idx_1001_simuln_idx_3005
            # create mut_complx_id specific MD simulation output directory
            mut_complx_mdSim_result_dir = os.path.join(mdSim_result_dir, dim_prot_complx_nm, chain_combo_nm, f'{mut_complx_id}')
            mdSim_status = prot_design_util.create_mdSim_res_dir_and_return_status(mut_complx_mdSim_result_dir, recreate_if_exists = mdSim_overwrite_existing_results)
            if(mdSim_status == 'mdSim_res_already_exists'):
                print(f'################## Postproc_6: As the MD simulation result already exists for the mutated protein complex with id: {mut_complx_id} at \n {mut_complx_mdSim_result_dir}\n hence, skipping MD simulation for it...')
                md_sim_postproc_6_info_lst.append(f'{dim_prot_complx_nm} : {mut_complx_id}'); md_sim_postproc_6_val_lst.append('mdSim_res_already_exists')
            else:
                print(f'\nPostproc_6: Executing MD simulation for the mutated protein complex with id: {mut_complx_id}...')
                prot_complx_tag = f' Postproc_6: complex_{dim_prot_complx_nm} :{mut_complx_id}:: '
                print(f'prot_complx_tag: {prot_complx_tag}')
                try:
                    # call execute_md_simulation() method with some more input arguments
                    kwargs['prot_complx_tag'] = prot_complx_tag
                    # find the specific location of the pdb file generated by af2 for this mutated protein complex
                    specific_inp_pdb_file_pattern = os.path.join(out_dir_mut_complx_struct_pred_af2, dim_prot_complx_nm, chain_combo_nm, f'{mut_complx_id}_unrelaxed_rank_001_alphafold2_*.pdb')
                    if(af2_use_amber):
                        specific_inp_pdb_file_pattern = specific_inp_pdb_file_pattern.replace('_unrelaxed_', '_relaxed_')
                    # end of if block:  if(af2_use_amber):
                    specific_inp_pdb_file = glob.glob(specific_inp_pdb_file_pattern, recursive=False)[0]
                    kwargs['specific_inp_pdb_file'] = specific_inp_pdb_file
                    kwargs['specific_mdSim_result_dir'] = mut_complx_mdSim_result_dir
                    avg_rmsd_nm, avg_rg_nm, avg_rmsf_nm = MD_Simulation.execute_md_simulation(**kwargs)
                    # store avg_rmsd_nm, avg_rg_nm, avg_rmsf_nm into md_sim_avg_rmsd_rmsf_rg_dict
                    md_sim_avg_rmsd_rmsf_rg_dict = {'complx_id': f'{dim_prot_complx_nm} : {mut_complx_id}', 'avg_rmsd_nm': avg_rmsd_nm, 'avg_rg_nm': avg_rg_nm, 'avg_rmsf_nm': avg_rmsf_nm}
                    # append the dict into md_sim_avg_rmsd_rmsf_rg_dict_lst
                    md_sim_avg_rmsd_rmsf_rg_dict_lst.append(md_sim_avg_rmsd_rmsf_rg_dict)
                    # update MD simulation status
                    md_sim_postproc_6_info_lst.append(f'{dim_prot_complx_nm} : {mut_complx_id}'); md_sim_postproc_6_val_lst.append(f'Success')
                except Exception as ex:
                    print(f'\n{prot_complx_tag} ################## Error!! Error in MD simulation: {ex}')
                    print(f'{prot_complx_tag} ################## Removing the result directory: {mut_complx_mdSim_result_dir}')
                    shutil.rmtree(mut_complx_mdSim_result_dir, ignore_errors=False, onerror=None)
                    md_sim_postproc_6_info_lst.append(f'{dim_prot_complx_nm} : {mut_complx_id}'); md_sim_postproc_6_val_lst.append(f'Error: {ex}')
                # end of try-catch block
            # end of if-else block: if(mdSim_status == 'mdSim_res_already_exists'):
        # end of for loop: for index, row in indiv_mut_complx_inp_df.iterrows():
    # end of for loop: for indiv_eligbl_mut_complx_inp_csv_file_nm in mut_complx_inp_csv_files_lst:

    # check and update md_sim_avg_rmsd_rmsf_rg_postproc_6.csv
    print(f'\nPostproc_6: Checking and updating md_sim_avg_rmsd_rmsf_rg_postproc_6.csv...')
    delta_change_df = pd.DataFrame(md_sim_avg_rmsd_rmsf_rg_dict_lst)
    update_md_sim_avg_rmsd_rmsf_rg_postproc_6_csv(mdSim_result_dir, delta_change_df)
    
    # check and update md_sim_info_postproc_6.csv
    print(f'\nPostproc_6: Checking and updating md_sim_info_postproc_6.csv...')
    delta_change_df = pd.DataFrame({'complx_id': md_sim_postproc_6_info_lst, 'md_sim_status': md_sim_postproc_6_val_lst})
    update_md_sim_info_postproc_6_csv(mdSim_result_dir, delta_change_df)

    print('inside run_postproc_6_complx_struct_mdSim() method - End')


def update_md_sim_info_postproc_6_csv(mdSim_result_dir, delta_change_df):
    print('Inside update_md_sim_info_postproc_6_csv() method - Start')
    md_sim_info_postproc_6_csv_path = os.path.join(mdSim_result_dir, 'md_sim_info_postproc_6.csv')

    # check whether 'md_sim_info_postproc_6.csv' file already exists at mdSim_result_dir
    csv_file_path = os.path.join(mdSim_result_dir, 'md_sim_info_postproc_6.csv')
    if not os.path.exists(csv_file_path):
        print(f'"md_sim_info_postproc_6.csv" file does NOT exist at {mdSim_result_dir}. Hence creating it...')
        delta_change_df.to_csv(csv_file_path, index=False)
    else:
        print(f'"md_sim_info_postproc_6.csv" file already exists at {mdSim_result_dir}. Hence updating or inserting rows...')
        con_df = pd.read_csv(csv_file_path)
        identifier_col_nm_lst = ['complx_id']
        modifiable_col_nm_lst = ['md_sim_status']
        
        # Case 1: Update existing rows
        con_df = con_df.set_index(identifier_col_nm_lst)
        delta_change_df = delta_change_df.set_index(identifier_col_nm_lst)
        update_mask_1 = con_df.index.isin(delta_change_df.index)
        update_mask_2 = delta_change_df.index.isin(con_df.index)
        con_df.loc[update_mask_1, modifiable_col_nm_lst] = delta_change_df.loc[update_mask_2, modifiable_col_nm_lst]

        # Case 2: Insert new rows
        insert_mask = ~delta_change_df.index.isin(con_df.index)
        con_df = pd.concat([con_df, delta_change_df.loc[insert_mask]])
        print("Rows updated or inserted successfully.")

        # reset index of con_df after update or insert
        con_df = con_df.reset_index()
        
        # overwrite the existing csv file with new content
        con_df.to_csv(csv_file_path, index=False)
    # end of if-else block: if not os.path.exists(csv_file_path):
    print('Inside update_md_sim_info_postproc_6_csv() method - End')


def update_md_sim_avg_rmsd_rmsf_rg_postproc_6_csv(mdSim_result_dir, delta_change_df):
    print('Inside update_md_sim_avg_rmsd_rmsf_rg_postproc_6_csv() method - Start')
    # md_sim_avg_rmsd_rmsf_rg_postproc_6_csv_path = os.path.join(mdSim_result_dir, 'md_sim_avg_rmsd_rmsf_rg_postproc_6.csv')

    # check whether 'md_sim_avg_rmsd_rmsf_rg_postproc_6.csv' file already exists at mdSim_result_dir
    csv_file_path = os.path.join(mdSim_result_dir, 'md_sim_avg_rmsd_rmsf_rg_postproc_6.csv')
    if not os.path.exists(csv_file_path):
        print(f'"md_sim_avg_rmsd_rmsf_rg_postproc_6.csv" file does NOT exist at {mdSim_result_dir}. Hence creating it...')
        delta_change_df.to_csv(csv_file_path, index=False)
    else:
        print(f'"md_sim_avg_rmsd_rmsf_rg_postproc_6.csv" file already exists at {mdSim_result_dir}. Hence updating or inserting rows...')
        con_df = pd.read_csv(csv_file_path)
        identifier_col_nm_lst = ['complx_id']
        modifiable_col_nm_lst = ['avg_rmsd_nm', 'avg_rg_nm', 'avg_rmsf_nm']
        
        # Case 1: Update existing rows
        con_df = con_df.set_index(identifier_col_nm_lst)
        delta_change_df = delta_change_df.set_index(identifier_col_nm_lst)
        update_mask_1 = con_df.index.isin(delta_change_df.index)
        update_mask_2 = delta_change_df.index.isin(con_df.index)
        con_df.loc[update_mask_1, modifiable_col_nm_lst] = delta_change_df.loc[update_mask_2, modifiable_col_nm_lst]

        # Case 2: Insert new rows
        insert_mask = ~delta_change_df.index.isin(con_df.index)
        con_df = pd.concat([con_df, delta_change_df.loc[insert_mask]])
        print("Rows updated or inserted successfully.")

        # reset index of con_df after update or insert
        con_df = con_df.reset_index()
        
        # overwrite the existing csv file with new content
        con_df.to_csv(csv_file_path, index=False)
    # end of if-else block: if not os.path.exists(csv_file_path):
    print('Inside update_md_sim_avg_rmsd_rmsf_rg_postproc_6_csv() method - End')


