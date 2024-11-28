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
##  from utils import prot_design_util, MD_Simulation_Porul as MD_Simulation


def run_postproc_6_complx_struct_mdSim(**kwargs):
    """Postprocessing stage where Molecular Dynamics (MD) simulation is carried out on the protein complex structure.

    This method is called from a triggering method like mat_p2ip_pd_postproc_trigger.trigger_pd_postproc().
    """
    print('Inside run_postproc_6_complx_struct_mdSim() method - Start')

    # ######### IMPORTANT: The following unix commands must be executed in the runtime environment (bash) before invoking MD simulation related python script - Start #########
    # ### module load apps/gromacs/2022/gpu  # For Paramvidya: module load apps/gromacs/16.6.2022/intel
    # ### source /home/apps/gromacs/gromacs-2022.2/installGPUIMPI/bin/GMXRC  # For Paramvidya: source /home/apps/gromacs/gromacs-2022.2/installGPUIOMPI/bin/GMXRC
    # ### export GMX_ENABLE_DIRECT_GPU_COMM=1
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
    out_dir_mut_complx_struct_pred_af2 = kwargs.get('out_dir_mut_complx_struct_pred_af2')
    mdSim_overwrite_existing_results = kwargs.get('mdSim_overwrite_existing_results');  forcefield_mdSim = kwargs.get('forcefield_mdSim')
    max_cadidate_count_mdSim = kwargs.get('max_cadidate_count_mdSim')
    mdSim_result_dir = kwargs.get('mdSim_result_dir') 
    print('####################################')

    # creating few lists to track the MD simulation
    md_sim_postproc_6_info_lst, md_sim_postproc_6_val_lst = [], []
    md_sim_avg_rmsd_rmsf_rg_dict_lst = []
    prot_complx_tag = f' Postproc_6: complex_{dim_prot_complx_nm} :: '

    # ######################## Check the availability of eligible MD Simulation candidate(s) for the original dimer -Start ########################
    # First, check whether there is any available MD Simulation candidate(s) for the original input protein complex
    print(f"\n!!! {prot_complx_tag} First, check whether there is any available MD Simulation candidate(s) for the original input protein complex.")
    print(f'For that check whether the dimeric protein complex name ({dim_prot_complx_nm}) is present in the "overall_accept_complx_struct_comp_res_postproc_5_part2.csv" file')
    overall_accept_complx_struct_csv = os.path.join(postproc_result_dir, 'overall_accept_complx_struct_comp_res_postproc_5_part2.csv')
    overall_accept_complx_struct_df = pd.read_csv(overall_accept_complx_struct_csv)
    complx_nm_filtered_df = overall_accept_complx_struct_df[overall_accept_complx_struct_df['cmplx'] == dim_prot_complx_nm]
    complx_nm_filtered_df = complx_nm_filtered_df.reset_index(drop=True)
    
    # Find the number of entries in the filtered dataframe
    num_of_rows_complx_nm_filtered_df = complx_nm_filtered_df.shape[0]
    # If the number of entries is zero, return
    if(num_of_rows_complx_nm_filtered_df == 0):
        print(f"\n!!! {prot_complx_tag} Found no entry for the dimeric complex name {dim_prot_complx_nm} in the {overall_accept_complx_struct_csv}.\
              \nSo, NOT proceeding further for the current dimeric complex: {dim_prot_complx_nm}\n")
        # save tracking info records and continue with the next dimeric complex
        md_sim_postproc_6_info_lst.append(f'{dim_prot_complx_nm}'); md_sim_postproc_6_val_lst.append(f'Error: Found no entry for the dimeric complex name {dim_prot_complx_nm} in the {overall_accept_complx_struct_csv}')
        delta_change_df = pd.DataFrame({'complx_id': md_sim_postproc_6_info_lst, 'md_sim_status': md_sim_postproc_6_val_lst})
        update_md_sim_info_postproc_6_csv(mdSim_result_dir, delta_change_df)
        return  # return back
    # end of if block: if(num_of_rows_complx_nm_filtered_df == 0):
    
    print(f"\n{prot_complx_tag} There is/are {num_of_rows_complx_nm_filtered_df} entry/entries for the dimeric complex name {dim_prot_complx_nm} in the {overall_accept_complx_struct_csv}. \
           \n So, proceeding further for the current dimeric complex: {dim_prot_complx_nm}...")
    # ######################## Check the availability of eligible MD Simulation candidate(s) for the original dimer -End ########################
    
    # Next, execute MD simulation for the original protein complex
    print(f'Next, execute MD simulation for the original protein complex: {dim_prot_complx_nm}')
    orig_prot_mdSim_result_dir = os.path.join(mdSim_result_dir, f'{dim_prot_complx_nm}', 'orig')
    mdSim_status = prot_design_util.create_mdSim_res_dir_and_return_status(orig_prot_mdSim_result_dir, recreate_if_exists = mdSim_overwrite_existing_results)
    ## if(mdSim_status == 'mdSim_res_already_exists'):
    ##     print(f'################## Postproc_6: As the MD simulation result already exists for the original protein complex: {dim_prot_complx_nm} at \n {orig_prot_mdSim_result_dir}\n hence, skipping MD simulation for it...')
    ##     md_sim_postproc_6_info_lst.append(f'{dim_prot_complx_nm} (orig)'); md_sim_postproc_6_val_lst.append('mdSim_res_already_exists')
    ## else:
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
        # shutil.rmtree(orig_prot_mdSim_result_dir, ignore_errors=True, onerror=None)
        md_sim_postproc_6_info_lst.append(f'{dim_prot_complx_nm} (orig)'); md_sim_postproc_6_val_lst.append(f'Error: {ex}')
    # end of try-catch block
    ## end of if-else block: if(mdSim_status == 'mdSim_res_already_exists'):


    # Next, execute MD simulation for all the mutated protein complex candidates corresponding to the original one
    print(f'\n Next, execute MD simulation for all the eligible mutated protein complex candidates corresponding to the original: {dim_prot_complx_nm}')
    # As already it has been confirmed that there is at least one MD Simulation eligible mutated protein complex candidate for the given dimer, so skipping 
    # the validations.

    # Restore back 'prot_complx_tag' to its value before executing MD simulation for the original protein complex
    prot_complx_tag = f' Postproc_6: complex_{dim_prot_complx_nm} :: '

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
        print(f'\n ###### {prot_complx_tag} fpi: {fpi} :: mpi: {mpi}')
        chain_combo_nm = f'fpi_{fpi}_mpi_{mpi}'
        print(f'\n {prot_complx_tag} chain_combo_nm: {chain_combo_nm}')
        
        # Find the number of MD Simulation eligible mutated protein complex candidate(s) for the given chain-combo
        num_of_md_sim_eligbl_cand = cmplx_fpi_mpi_grouped_df.shape[0]
        if(num_of_md_sim_eligbl_cand > max_cadidate_count_mdSim):
            # if the number of eligible candidates is more than the max threshold, clip the count.
            num_of_md_sim_eligbl_cand = max_cadidate_count_mdSim
        print(f'\n {prot_complx_tag} num_of_md_sim_eligbl_cand: {num_of_md_sim_eligbl_cand}')
        cmplx_fpi_mpi_grouped_df = cmplx_fpi_mpi_grouped_df.head(num_of_md_sim_eligbl_cand)

        # As cmplx_fpi_mpi_grouped_df is already sorted by 'bb_rmsd_superimp' column in ascending order, 
        # hence, no further sorting is required on cmplx_fpi_mpi_grouped_df.
        print(f'\n\n ##### {prot_complx_tag} Iterating over the MD Simulation eligible mutated protein complex candidate(s) one by one.')
        for index, row in cmplx_fpi_mpi_grouped_df.iterrows():
            mut_complx_id = row['id']; seq = row['sequence']
            print(f'\n {prot_complx_tag} iteration index: {index} :: mut_complx_id: {mut_complx_id} :: seq: {seq}')
            # sample mut_complx_id: cmplx_2I25_fpi_L_mpi_N_batch_idx_1001_simuln_idx_3005
            # create mut_complx_id specific MD simulation output directory
            mut_complx_mdSim_result_dir = os.path.join(mdSim_result_dir, dim_prot_complx_nm, chain_combo_nm, f'{mut_complx_id}')
            mdSim_status = prot_design_util.create_mdSim_res_dir_and_return_status(mut_complx_mdSim_result_dir, recreate_if_exists = mdSim_overwrite_existing_results)
            ## if(mdSim_status == 'mdSim_res_already_exists'):
            ##     print(f'################## Postproc_6: As the MD simulation result already exists for the mutated protein complex with id: {mut_complx_id} at \n {mut_complx_mdSim_result_dir}\n hence, skipping MD simulation for it...')
            ##     md_sim_postproc_6_info_lst.append(f'{dim_prot_complx_nm} : {mut_complx_id}'); md_sim_postproc_6_val_lst.append('mdSim_res_already_exists')
            ## else:
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
                # shutil.rmtree(mut_complx_mdSim_result_dir, ignore_errors=True, onerror=None)
                md_sim_postproc_6_info_lst.append(f'{dim_prot_complx_nm} : {mut_complx_id}'); md_sim_postproc_6_val_lst.append(f'Error: {ex}')
            # end of try-catch block
            ## end of if-else block: if(mdSim_status == 'mdSim_res_already_exists'):
        # end of for loop: for index, row in cmplx_fpi_mpi_grouped_df.iterrows():
    # end of for loop: for cmplx_fpi_mpi_grouped_df in cmplx_fpi_mpi_grouped_df_list:

    # check and update md_sim_avg_rmsd_rmsf_rg_postproc_6.csv
    print(f'\nPostproc_6: Checking and updating md_sim_avg_rmsd_rmsf_rg_postproc_6.csv...')
    if(len(md_sim_avg_rmsd_rmsf_rg_dict_lst) == 0):
        print(f'\nPostproc_6: md_sim_avg_rmsd_rmsf_rg_dict_lst is empty and hence NOT updating md_sim_avg_rmsd_rmsf_rg_postproc_6.csv')
    else:
        delta_change_df = pd.DataFrame(md_sim_avg_rmsd_rmsf_rg_dict_lst)
        update_md_sim_avg_rmsd_rmsf_rg_postproc_6_csv(mdSim_result_dir, delta_change_df)
    
    # check and update md_sim_info_postproc_6.csv
    print(f'\nPostproc_6: Checking and updating md_sim_info_postproc_6.csv...')
    if(len(md_sim_postproc_6_info_lst) == 0):
        print(f'\nPostproc_6: md_sim_postproc_6_info_lst is empty and hence NOT updating md_sim_info_postproc_6.csv')
    else:
        delta_change_df = pd.DataFrame({'complx_id': md_sim_postproc_6_info_lst, 'md_sim_status': md_sim_postproc_6_val_lst})
        update_md_sim_info_postproc_6_csv(mdSim_result_dir, delta_change_df)
    print('Inside run_postproc_6_complx_struct_mdSim() method - End')


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


def prep_mds_overall_analysis_result(root_path='./', itr_tag=None, update=False):
    print('inside prep_mds_overall_analysis_result() method - Start')
    postproc_mds_res_path = os.path.join(root_path, 'dataset/postproc_data/mdSim_result', f'{itr_tag}')
    # postproc_mds_res_path = os.path.join(root_path, 'dataset/postproc_data/mdSim_result_final', f'{itr_tag}')

    # Find all the locations of the folder '9_analysis_out' inside postproc_mds_res_path
    analysis_out_fldr_loc_lst = glob.glob(os.path.join(postproc_mds_res_path, '**/9_analysis_out'), recursive=True)

    md_sim_avg_rmsd_rmsf_rg_dict_lst = []  # It would be used to create 'overall_md_sim_avg_rmsd_rmsf_rg_postproc_6.csv'
    # Iterate through each '9_analysis_out' folder in analysis_out_fldr_loc_lst
    for ana_out_itr, analysis_out_fldr_loc in enumerate(analysis_out_fldr_loc_lst):
        # '9_analysis_out' folder location will be either {postproc_mds_res_path}/7CEI/orig/9_analysis_out 
        # or {postproc_mds_res_path}/7CEI/fpi_B_mpi_A/cmplx_7CEI_fpi_B_mpi_A_batch_idx_4394_simuln_idx_21973/9_analysis_out
        analysis_out_fldr_loc_split_lst = analysis_out_fldr_loc.split('/')
        if('orig' in analysis_out_fldr_loc_split_lst):
            # For 'orig' folder
            complx_id = f'{analysis_out_fldr_loc_split_lst[-3]} : (orig)'
        else:
            complx_id = f'{analysis_out_fldr_loc_split_lst[-4]} : {analysis_out_fldr_loc_split_lst[-2]}'
        # End of if-else block
        print(f'\n\n complx_id is {complx_id} :: Iteration {ana_out_itr} out of {len(analysis_out_fldr_loc_lst)}\n')

        # Find avg_rmsd_nm
        avg_rmsd_nm = None
        rmsd_csv_path = os.path.join(analysis_out_fldr_loc, 'rmsd/rmsd.csv')
        if os.path.exists(rmsd_csv_path):
            # If rmsd.csv exists, then calculate avg. rmsd
            rmsd_df = pd.read_csv(rmsd_csv_path)
            avg_rmsd_nm = round(rmsd_df['RMSD(nm)'].mean(), ndigits=3)
        else:
            print(f'!!! Attention !!! Attention !!! {rmsd_csv_path} does NOT exist !!!')

        # Find avg_rg_nm
        avg_rg_nm = None
        rg_csv_path = os.path.join(analysis_out_fldr_loc, 'rg/rg.csv')
        if os.path.exists(rg_csv_path):
            # If rg.csv exists, then calculate avg. rg
            rg_df = pd.read_csv(rg_csv_path)
            avg_rg_nm = round(rg_df['rg(nm)'].mean(), ndigits=3)
        else:
            print(f'!!! Attention !!! Attention !!! {rg_csv_path} does NOT exist !!!')
        
        # Find avg_rmsf_nm
        avg_rmsf_nm = None
        rmsf_csv_path = os.path.join(analysis_out_fldr_loc, 'rmsf/rmsf.csv')
        if os.path.exists(rmsf_csv_path):
            # If rmsf.csv exists, then calculate avg. rmsf
            rmsf_df = pd.read_csv(rmsf_csv_path)
            avg_rmsf_nm = round(rmsf_df['rmsf(nm)'].mean(), ndigits=3)
        else:
            print(f'!!! Attention !!! Attention !!! {rmsf_csv_path} does NOT exist !!!')
        
        # Create and populate md_sim_avg_rmsd_rmsf_rg_dict
        md_sim_avg_rmsd_rmsf_rg_dict = {}
        md_sim_avg_rmsd_rmsf_rg_dict['complx_id'] = complx_id
        md_sim_avg_rmsd_rmsf_rg_dict['avg_rmsd_nm'] = avg_rmsd_nm
        md_sim_avg_rmsd_rmsf_rg_dict['avg_rg_nm'] = avg_rg_nm
        md_sim_avg_rmsd_rmsf_rg_dict['avg_rmsf_nm'] = avg_rmsf_nm
        # Append md_sim_avg_rmsd_rmsf_rg_dict at md_sim_avg_rmsd_rmsf_rg_dict_lst
        md_sim_avg_rmsd_rmsf_rg_dict_lst.append(md_sim_avg_rmsd_rmsf_rg_dict)
    # End of for loop: for ana_out_itr, analysis_out_fldr_loc in enumerate(analysis_out_fldr_loc_lst):

    # Create a dataframe from md_sim_avg_rmsd_rmsf_rg_dict_lst
    md_sim_avg_rmsd_rmsf_rg_df = pd.DataFrame(md_sim_avg_rmsd_rmsf_rg_dict_lst)
    # Depneding on 'update' flag, either create a new 'overall_md_sim_avg_rmsd_rmsf_rg_postproc_6.csv' file OR update an already existing one.
    avg_val_csv_file_loc = os.path.join(root_path, 'dataset/postproc_data/result_dump', itr_tag)
    avg_val_csv_file_nm_loc = os.path.join(avg_val_csv_file_loc, 'overall_md_sim_avg_rmsd_rmsf_rg_postproc_6.csv')
    if(not update):
        # Create new csv file
        print(f'\n#### "update" flag = False. Hence, creating a new "overall_md_sim_avg_rmsd_rmsf_rg_postproc_6.csv" file at ({avg_val_csv_file_loc})')
        md_sim_avg_rmsd_rmsf_rg_df.to_csv(avg_val_csv_file_nm_loc, index=False)
    elif(update):
        # Update the csv (if already exists)
        print(f'\n### "update" flag = True. Hence, updating the "overall_md_sim_avg_rmsd_rmsf_rg_postproc_6.csv" file at {avg_val_csv_file_loc} (if it exists).')

        if not os.path.exists(avg_val_csv_file_nm_loc):
            print(f'\n "overall_md_sim_avg_rmsd_rmsf_rg_postproc_6.csv" file does NOT exist at {avg_val_csv_file_loc}. Hence creating it...')
            md_sim_avg_rmsd_rmsf_rg_df.to_csv(avg_val_csv_file_nm_loc, index=False)
        else:
            print(f'\n "overall_md_sim_avg_rmsd_rmsf_rg_postproc_6.csv" file already exists at {avg_val_csv_file_loc}. Hence updating or inserting rows...')
            consolidated_df = pd.read_csv(avg_val_csv_file_nm_loc)
            delta_change_df = md_sim_avg_rmsd_rmsf_rg_df
            identifier_col_nm_lst = ['complx_id']
            modifiable_col_nm_lst = ['avg_rmsd_nm', 'avg_rg_nm', 'avg_rmsf_nm']
            
            # Case 1: Update existing rows
            consolidated_df = consolidated_df.set_index(identifier_col_nm_lst)
            delta_change_df = delta_change_df.set_index(identifier_col_nm_lst)
            update_mask_1 = consolidated_df.index.isin(delta_change_df.index)
            update_mask_2 = delta_change_df.index.isin(consolidated_df.index)
            consolidated_df.loc[update_mask_1, modifiable_col_nm_lst] = delta_change_df.loc[update_mask_2, modifiable_col_nm_lst]

            # Case 2: Insert new rows
            insert_mask = ~delta_change_df.index.isin(consolidated_df.index)
            consolidated_df = pd.concat([consolidated_df, delta_change_df.loc[insert_mask]])
            print("Rows updated or inserted successfully.")

            # reset index of consolidated_df after update or insert
            consolidated_df = consolidated_df.reset_index()
            
            # overwrite the existing csv file with new content
            consolidated_df.to_csv(avg_val_csv_file_nm_loc, index=False)
        # end of if-else block: if not os.path.exists(avg_val_csv_file_nm_loc):
    # End of if-elif block: if(not update):
    print('inside prep_mds_overall_analysis_result() method - End')



if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/matpip_pd_prj')

    itr_tag = 'mcmc_fullLen_puFalse_batch5_mutPrcntLen10'
    # itr_tag = 'fullLen_puFalse_batch5_thorough'

    # Prepare overall analysis result for the MD-Simulation
    prep_mds_overall_analysis_result(root_path=root_path, itr_tag=itr_tag, update=True)
