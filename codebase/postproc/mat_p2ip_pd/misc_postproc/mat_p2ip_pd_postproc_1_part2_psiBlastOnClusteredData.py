import os, sys

from pathlib import Path
path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
from Bio.Blast import NCBIXML
import glob
import pandas as pd
import subprocess
from utils import PPIPUtils, postproc_clustering


def run_postproc_1_part2_psiBlastOnClusteredData(**kwargs):
    """Optional postprocessing step to run PSI-Blast on the clustered data 

    This method is called from a triggering method like mat_p2ip_pd_postproc_trigger.trigger_pd_postproc().
    """
    print('inside run_postproc_1_part2_psiBlastOnClusteredData() method - Start')
    print('####################################')
    # Iterate over kwargs and raise ValueError if any of the input arguments (except a few) is None. Also print each keyword argument name and respective value.
    for arg_name, arg_value in kwargs.items():
        if(arg_value is None):
            raise ValueError(f"Argument '{arg_name}' must be provided with a value.")
        print(f"'{arg_name}': {arg_value}")
    # end of for loop: for arg_name, arg_value in kwargs.items():

    # retrieve all the keyword arguments
    root_path = kwargs.get('root_path'); dim_prot_complx_nm = kwargs.get('dim_prot_complx_nm')
    simulation_exec_mode = kwargs.get('simulation_exec_mode'); cuda_index = kwargs.get('cuda_index')
    sim_result_dir = kwargs.get('sim_result_dir'); postproc_result_dir = kwargs.get('postproc_result_dir')
    pdb_file_location = kwargs.get('pdb_file_location'); ppi_score_threshold = kwargs.get('ppi_score_threshold')
    postproc_psiblast_enabled = kwargs.get('postproc_psiblast_enabled')
    postproc_psiblast_exec_path = kwargs.get('postproc_psiblast_exec_path'); postproc_psiblast_uniprot_db_path = kwargs.get('postproc_psiblast_uniprot_db_path')
    postproc_psiblast_result_dir = kwargs.get('postproc_psiblast_result_dir'); psiBlast_percent_similarity_score_threshold = kwargs.get('psiBlast_percent_similarity_score_threshold')
    max_num_of_seq_below_psiBlast_sim_score_threshold = kwargs.get('max_num_of_seq_below_psiBlast_sim_score_threshold')
    max_num_of_seq_for_af2_chain_struct_pred = kwargs.get('max_num_of_seq_for_af2_chain_struct_pred')
    inp_dir_mut_seq_struct_pred_af2 = kwargs.get('inp_dir_mut_seq_struct_pred_af2')
    print('####################################')

    # Set the environment variable 'BLASTDB' for the PSI-Blast execution (python equivalent of 'export BLASTDB=/psiblast/uniprot_db/path/here')
    os.environ['BLASTDB'] = postproc_psiblast_uniprot_db_path
    # create the postprocess result directory for the given dimeric protein complex
    print(f'\nPostproc_1: Creating the postprocess result directory for the given dimeric protein complex: {dim_prot_complx_nm}')
    complx_postproc_result_dir = os.path.join(postproc_result_dir, f'complex_{dim_prot_complx_nm}')
    PPIPUtils.createFolder(complx_postproc_result_dir, recreate_if_exists=False)
    
    # creating few lists to track the postprocessing
    misc_info_complx_postproc_1_lst, misc_info_val_complx_postproc_1_lst = [], []

    # Search the simulation result directory for the given dimeric protein complex name
    print(f'\nPostproc_1: Searching the simulation result directory for the given dimeric protein complex name: {dim_prot_complx_nm}')
    simuln_res_dir = os.path.join(root_path, sim_result_dir, f'complex_{dim_prot_complx_nm}')
    if(not os.path.exists(simuln_res_dir)):
        print("!!!! Postproc_1: The directory: " + str(simuln_res_dir) + " does not exist.... NOT proceeding further... !!!!\n")
        misc_info_complx_postproc_1_lst.append('sim_res_dir_search_result'); misc_info_val_complx_postproc_1_lst.append(f'{simuln_res_dir} NOT FOUND')
        misc_info_complx_postproc_1_lst.append('status'); misc_info_val_complx_postproc_1_lst.append('ERROR')
        # save misc info records and return
        misc_info_complx_postproc_1_df = pd.DataFrame({'misc_info': misc_info_complx_postproc_1_lst, 'misc_info_val': misc_info_val_complx_postproc_1_lst})
        misc_info_complx_postproc_1_df.to_csv(os.path.join(complx_postproc_result_dir, 'misc_info_complex_postproc_1.csv'), index=False)
        return  # return to the triggering method like mat_p2ip_pd_postproc_trigger.trigger_pd_postproc().
    #end of if block: if(not os.path.exists(simuln_res_dir)):
    
    print("Simulation result directory: " + str(simuln_res_dir) + " exists...Proceeding further...\n")
    # set the search-result as success
    misc_info_complx_postproc_1_lst.append('sim_res_dir_search_result'); misc_info_val_complx_postproc_1_lst.append(f'{simuln_res_dir} FOUND')

    # Finds the chain combinations (fixed, mut) inside the simulation result directory
    print('Postproc_1: Finding the chain combination (fixed, mut) directories inside the simulation result directory...')
    chain_combo_lst = PPIPUtils.get_top_level_directories(simuln_res_dir)
    if(len(chain_combo_lst) == 0):
        print(f'!!! Postproc_1: No chain combination (fixed, mut) is found inside the simulation result directory: {simuln_res_dir}...NOT proceeding further... !!!\n')
        misc_info_complx_postproc_1_lst.append('chain_combo_inside_sim_res_dir'); misc_info_val_complx_postproc_1_lst.append(f'NOT FOUND')
        misc_info_complx_postproc_1_lst.append('status'); misc_info_val_complx_postproc_1_lst.append('ERROR')
        # save misc info records and return
        misc_info_complx_postproc_1_df = pd.DataFrame({'misc_info': misc_info_complx_postproc_1_lst, 'misc_info_val': misc_info_val_complx_postproc_1_lst})
        misc_info_complx_postproc_1_df.to_csv(os.path.join(complx_postproc_result_dir, 'misc_info_complex_postproc_1.csv'), index=False)
        return  # return to the triggering method like mat_p2ip_pd_postproc_trigger.trigger_pd_postproc().
    # end of if block: if(len(chain_combo_lst) == 0):
    
    print(f'{len(chain_combo_lst)} chain combination (fixed, mut) found with name(s): {chain_combo_lst}')
    misc_info_complx_postproc_1_lst.append('chain_combo_inside_sim_res_dir'); misc_info_val_complx_postproc_1_lst.append(f'{chain_combo_lst}')
    # save misc info records
    misc_info_complx_postproc_1_df = pd.DataFrame({'misc_info': misc_info_complx_postproc_1_lst, 'misc_info_val': misc_info_val_complx_postproc_1_lst})
    misc_info_complx_postproc_1_df.to_csv(os.path.join(complx_postproc_result_dir, 'misc_info_complex_postproc_1.csv'), index=False)
    
    # iterating over chain combination (fixed, mut) found
    print('Iterating over chain combination (fixed, mut) ...')
    for chain_combo_itr in range(len(chain_combo_lst)):
        curnt_chain_combo = chain_combo_lst[chain_combo_itr]
        print(f'curnt_chain_combo: {curnt_chain_combo}')
        curnt_chain_combo_splitted_lst = curnt_chain_combo.split('_')
        fixed_prot_id, mut_prot_id = curnt_chain_combo_splitted_lst[1], curnt_chain_combo_splitted_lst[-1]
        # fix_mut_prot_id_tag is a string to be used in the subsequent print statements to keep track of the specific iteration
        fix_mut_prot_id_tag = f' Postproc_1: complex_{dim_prot_complx_nm}_fixed_prot_id_{fixed_prot_id}_mut_prot_id_{mut_prot_id}:: '
        print(f'fix_mut_prot_id_tag: {fix_mut_prot_id_tag}')

        # create the postprocess result directory for the curnt_chain_combo
        print(f'\n{fix_mut_prot_id_tag} Creating the postprocess result directory for the curnt_chain_combo')
        curnt_chain_combo_postproc_result_dir = os.path.join(complx_postproc_result_dir, f'{curnt_chain_combo}')
        PPIPUtils.createFolder(curnt_chain_combo_postproc_result_dir, recreate_if_exists=False)
        # creating few lists to track the postprocessing of curnt_chain_combo
        curnt_chain_combo_postproc_1_info_lst, curnt_chain_combo_postproc_1_val_lst = [], []

        # Search for all folders with the specific name pattern 'res_totItr*' in the curnt_chain_combo directory
        res_totItr_folders = glob.glob(os.path.join(simuln_res_dir, curnt_chain_combo, 'res_totItr*'), recursive=False)
        # Find the number of resulting directories
        num_res_totItr_folders = len(res_totItr_folders)
        curnt_chain_combo_postproc_1_info_lst.append('num_res_totItr_dir_inside_chain_combo_dir'); curnt_chain_combo_postproc_1_val_lst.append(f'{num_res_totItr_folders}')
        # If the number of resulting directories is not equal to one, return
        if(num_res_totItr_folders != 1):
            print(f"\n!!!{fix_mut_prot_id_tag} Found multiple ({num_res_totItr_folders}) directories with the pattern 'res_totItr*'. So, NOT proceeding further for the curnt_chain_combo: {curnt_chain_combo}\n")
            curnt_chain_combo_postproc_1_info_lst.append('status'); curnt_chain_combo_postproc_1_val_lst.append('ERROR')
            # save curnt_chain_combo info records and continue with the next chain combination
            curnt_chain_combo_postproc_1_info_df = pd.DataFrame({'misc_info': curnt_chain_combo_postproc_1_info_lst, 'misc_info_val': curnt_chain_combo_postproc_1_val_lst})
            curnt_chain_combo_postproc_1_info_df.to_csv(os.path.join(curnt_chain_combo_postproc_result_dir, 'chain_combo_postproc_1.csv'), index=False)
            continue  # continue with the next chain combination
        # end of if block: if(num_res_totItr_folders != 1):

        # If there is exactly one resulting directory, parse every CSV file with the specific name pattern 'wrapperStepIdx_*_*_localStepSize_*.csv' inside it using an inner loop
        print("\n As there is exactly one resulting directory, parsing every CSV file with the specific name pattern 'wrapperStepIdx_*_*_localStepSize_*.csv' inside it using an inner loop ...")
        res_totItr_dir = res_totItr_folders[0]

        batch_res_csv_files = None
        if(simulation_exec_mode == 'remc'):
            batch_res_csv_files = glob.glob(os.path.join(res_totItr_dir, 'wrapperStepIdx_*_*_localStepSize_*.csv'))
        else:
            # For mcmc
            batch_res_csv_files = glob.glob(os.path.join(res_totItr_dir, 'batchIdx_*_*_batchSize_*.csv'))

        prob_thresholded_df_lst = []
        for indiv_batch_res_csv_file in batch_res_csv_files:
            indiv_batch_res_df = pd.read_csv(indiv_batch_res_csv_file)
            print(f"\n{fix_mut_prot_id_tag}Parsed {indiv_batch_res_csv_file} into pandas DataFrame:")
            # Filter indiv_batch_res_df using ppi_score_threshold and also exclude the original mutating chain 
            indiv_prob_thresholded_df = indiv_batch_res_df[(indiv_batch_res_df['ppi_score'] > ppi_score_threshold) & (~((indiv_batch_res_df['batch_idx'] == 0) & (indiv_batch_res_df['simuln_idx'] == 0)))]
            # indiv_prob_thresholded_df = indiv_batch_res_df[(indiv_batch_res_df['ppi_score'] > ppi_score_threshold) & (indiv_batch_res_df['ppi_score'] < 0.99)]
            # Remove any duplicate rows for the identical mutated sequences. This may happen if the randomly selected amino-acid is same as that is 
            # already present in the respective replacement position.
            indiv_prob_thresholded_df = indiv_prob_thresholded_df.drop_duplicates(subset=['prot_seq'], keep='first')
            # Reset index of indiv_prob_thresholded_df
            indiv_prob_thresholded_df = indiv_prob_thresholded_df.reset_index(drop=True)
            # Check if the filtered df is not empty. If not empty, then append it in a list (prob_thresholded_df_lst) so that 
            # the listed dataframes can be concatenated after exiting from this inner for loop
            if(indiv_prob_thresholded_df.shape[0] > 0):
                prob_thresholded_df_lst.append(indiv_prob_thresholded_df)
        # end of for loop: for indiv_batch_res_csv_file in batch_res_csv_files:
        
        curnt_chain_combo_postproc_1_info_lst.append('len(prob_thresholded_df_lst)'); curnt_chain_combo_postproc_1_val_lst.append(f'{len(prob_thresholded_df_lst)}')
        # Check whether prob_thresholded_df_lst is empty
        if(len(prob_thresholded_df_lst) == 0):
            # There is not a single mutated sequence in the simulation which yielded ppi score greater than the 'ppi_score_threshold'.
            # So, continue with the next chain combination
            print(f"\n!!!{fix_mut_prot_id_tag} There is not a single mutated sequence in the simulation which yielded ppi score greater than the 'ppi_score_threshold' ({ppi_score_threshold}). \
                  \n So, NOT proceeding further for the curnt_chain_combo.\n")
            curnt_chain_combo_postproc_1_info_lst.append('status'); curnt_chain_combo_postproc_1_val_lst.append('ERROR')
            # save curnt_chain_combo info records and continue with the next chain combination
            curnt_chain_combo_postproc_1_info_df = pd.DataFrame({'misc_info': curnt_chain_combo_postproc_1_info_lst, 'misc_info_val': curnt_chain_combo_postproc_1_val_lst})
            curnt_chain_combo_postproc_1_info_df.to_csv(os.path.join(curnt_chain_combo_postproc_result_dir, 'chain_combo_postproc_1.csv'), index=False)
            continue  # continue with the next chain combination
        # end of if block: if(len(indiv_prob_thresholded_df) == 0):
        
        # Concate the dataframes present in prob_thresholded_df_lst
        print(f'\n{fix_mut_prot_id_tag} Creating prob_thresholded_df and sorting it with respect to "ppi_score" value in the descending order.')
        prob_thresholded_df = pd.concat(prob_thresholded_df_lst, ignore_index=True, sort=False)
        # Remove any duplicate rows for the identical mutated sequences from prob_thresholded_df
        prob_thresholded_df = prob_thresholded_df.drop_duplicates(subset=['prot_seq'], keep='first')
        # Reset index of prob_thresholded_df
        prob_thresholded_df = prob_thresholded_df.reset_index(drop=True)
        print(f"\n{fix_mut_prot_id_tag} There is/are {prob_thresholded_df.shape[0]} mutated sequence(s) in the simulation which yielded ppi score greater than the 'ppi_score_threshold' ({ppi_score_threshold}).")
        # Sort prob_thresholded_df with respect to 'ppi_score' value in descending order and save the sorted df as csv file
        prob_thresholded_df.sort_values(by=['ppi_score'], ascending=False, inplace=True, ignore_index=True)
        prob_thresholded_df.to_csv(os.path.join(curnt_chain_combo_postproc_result_dir, 'prob_thresholded_df.csv'), index=False)

        # ####### CODE SNIPPET IS COMMENTED OUT AS PSI-BLAST BASED FILTERING IS NOW APPLIED -START
        ## print(f"{fix_mut_prot_id_tag}Selecting the mutated sequences corresponding to the top-k rows of the sorted dataframe where k = 'max_num_of_seq_above_ppi_score_threshold' = {max_num_of_seq_above_ppi_score_threshold}. \
        ##       \n If there is not enough rows in the sorted dataframe (i.e. when number of rows < k), then select all the rows from the dataframe.")
        ## num_rows_prob_thresholded_df = prob_thresholded_df.shape[0]
        ## k = max_num_of_seq_above_ppi_score_threshold
        ## top_k_seq_df = None
        ## if(num_rows_prob_thresholded_df > k):
        ##     # select top-k rows
        ##     top_k_seq_df = prob_thresholded_df.head(k)
        ## else:
        ##     # select all the rows
        ##     top_k_seq_df = prob_thresholded_df
        ##     k = num_rows_prob_thresholded_df
        ## # end of if-else block
        ## # save top_k_seq_df
        ## top_k_seq_df.to_csv(os.path.join(curnt_chain_combo_postproc_result_dir,f'top_{k}_seq.csv'), index=False)
        # ####### CODEinp_dir_mut_seq_struct_pred_af2 SNIPPET IS COMMENTED OUT AS PSI-BLAST BASED FILTERING IS NOW APPLIED -END

        # Apply PSI-Blast based filtering on each sequence of the prob_thresholded_df. The acceptance criteria:
        #  'evalue' (input argument to the blast) = 0.001 and output 'Total Score' < 70%. Also, there will be a maximum limit (say, 100) of 
        # these sequence satifying PSI-Blasr criterion.
        print(f"{fix_mut_prot_id_tag} Applying PSI-Blast based filtering on each sequence of the prob_thresholded_df to obtain psi_blast_thresholded_df.\
              \n psi_blast_thresholded_df will be sorted in the descending order of ppi_score as it is filtered from prob_thresholded_df, sorted by ppi_score.\
              \n Also, there will be a maximum limit (say, 100) of these sequences satifying PSI-Blasr criterion\n")
        PPIPUtils.createFolder(postproc_psiblast_result_dir, recreate_if_exists=False)
        kwargs['fix_mut_prot_id_tag'] = fix_mut_prot_id_tag; kwargs['curnt_chain_combo'] = curnt_chain_combo; kwargs['prob_thresholded_df'] = prob_thresholded_df
        # Check whether PSI-Blast based sequence similarity check is enabled as a part of the postprocessing.
        # If not enabled, then directly use prob_thresholded_df as psi_blast_thresholded_df
        psi_blast_thresholded_df = None
        if(not postproc_psiblast_enabled):
            print(f"\n{fix_mut_prot_id_tag} Since, 'postproc_psiblast_enabled' = {postproc_psiblast_enabled}, directly using prob_thresholded_df as psi_blast_thresholded_df")
            psi_blast_thresholded_df = prob_thresholded_df
        else:
            print(f"\n{fix_mut_prot_id_tag} Since, 'postproc_psiblast_enabled' = {postproc_psiblast_enabled}, actually applying PSI-Blast based filtering to obtain psi_blast_thresholded_df")
            psi_blast_thresholded_df = apply_postproc_psi_blast(**kwargs)
        # end of if-else block: if(not postproc_psiblast_enabled):
        if(psi_blast_thresholded_df is None):
            curnt_chain_combo_postproc_1_info_lst.append('psi_blast_thresholded_df.shape[0]'); curnt_chain_combo_postproc_1_val_lst.append(str(0))
            print(f"\n!!! {fix_mut_prot_id_tag} There is not a single entry in prob_thresholded_df satisfying PSI-Blast based criterion. So, skipping this curnt_chain_combo: {curnt_chain_combo}...\n")
            curnt_chain_combo_postproc_1_info_lst.append('status'); curnt_chain_combo_postproc_1_val_lst.append('ERROR')
            # save curnt_chain_combo info records and continue with the next chain combination
            curnt_chain_combo_postproc_1_info_df = pd.DataFrame({'misc_info': curnt_chain_combo_postproc_1_info_lst, 'misc_info_val': curnt_chain_combo_postproc_1_val_lst})
            curnt_chain_combo_postproc_1_info_df.to_csv(os.path.join(curnt_chain_combo_postproc_result_dir, 'chain_combo_postproc_1.csv'), index=False)
            continue  # continue with the next chain combination

        num_rows_psi_blast_thresholded_df = psi_blast_thresholded_df.shape[0]
        print(f"\n{fix_mut_prot_id_tag} There are {num_rows_psi_blast_thresholded_df} entries in psi_blast_thresholded_df satisfying PSI-Blast based criterion.")
        curnt_chain_combo_postproc_1_info_lst.append('psi_blast_thresholded_df.shape[0]'); curnt_chain_combo_postproc_1_val_lst.append(f'{num_rows_psi_blast_thresholded_df}')
        # save psi_blast_thresholded_df as csv file
        psi_blast_thresholded_df.to_csv(os.path.join(curnt_chain_combo_postproc_result_dir, 'psi_blast_thresholded_df.csv'), index=False)

        print(f"{fix_mut_prot_id_tag}Checking whether psi_blast_thresholded_df has sufficient number of entries to perform clustering.\
              \n If the entry count ({num_rows_psi_blast_thresholded_df}) > max_num_of_seq_for_af2_chain_struct_pred ({max_num_of_seq_for_af2_chain_struct_pred}), then go for clustering and select {max_num_of_seq_for_af2_chain_struct_pred} mutated sequences for AF2 prediction. Otherwise, do not go for clustering and select those sequences directly for AF2 prediction.")
        k = max_num_of_seq_for_af2_chain_struct_pred
        sel_k_seq_psiBlast_thresholded_df_for_af2_pred = None
        if(num_rows_psi_blast_thresholded_df <= k):
            sel_k_seq_psiBlast_thresholded_df_for_af2_pred = psi_blast_thresholded_df
        else:
            # perform clustering
            kwargs['psi_blast_thresholded_df'] = psi_blast_thresholded_df
            sel_k_seq_psiBlast_thresholded_df_for_af2_pred = postproc_clustering.perform_clustering(**kwargs)
        # end of if-else block
        # save sel_k_seq_psiBlast_thresholded_df_for_af2_pred
        sel_k_seq_psiBlast_thresholded_df_for_af2_pred.to_csv(os.path.join(curnt_chain_combo_postproc_result_dir,f'sel_{k}_seq_psiBlast_thresholded.csv'), index=False)

        # save curnt_chain_combo info records
        curnt_chain_combo_postproc_1_info_df = pd.DataFrame({'misc_info': curnt_chain_combo_postproc_1_info_lst, 'misc_info_val': curnt_chain_combo_postproc_1_val_lst})
        curnt_chain_combo_postproc_1_info_df.to_csv(os.path.join(curnt_chain_combo_postproc_result_dir, 'chain_combo_postproc_1.csv'), index=False)

        print(f"\n\n##### {fix_mut_prot_id_tag} Creating input csv for the mutated sequence structure prediction by AlphaFold2 #####")
        # Create input csv for the mutated sequence structure prediction by AlphaFold2
        mut_seq_struct_af2_inp_df = pd.DataFrame()
        mut_seq_struct_af2_inp_df['id'] = f'cmplx_{dim_prot_complx_nm}_fpi_{fixed_prot_id}_mpi_{mut_prot_id}_batch_idx_' + sel_k_seq_psiBlast_thresholded_df_for_af2_pred['batch_idx'].astype(str) + '_simuln_idx_' + sel_k_seq_psiBlast_thresholded_df_for_af2_pred['simuln_idx'].astype(str)
        mut_seq_struct_af2_inp_df['sequence'] = sel_k_seq_psiBlast_thresholded_df_for_af2_pred['prot_seq']
        # Save the df as csv file in curnt_chain_combo_postproc_result_dir
        mut_seq_struct_af2_inp_df.to_csv(os.path.join(curnt_chain_combo_postproc_result_dir, f'af2inp_cmpl_{dim_prot_complx_nm}_fpi_{fixed_prot_id}_mpi_{mut_prot_id}.csv'), index=False)
        # Also, save the df in the input directory of the AlphaFold2.
        mut_seq_struct_af2_inp_df.to_csv(os.path.join(inp_dir_mut_seq_struct_pred_af2, f'af2inp_cmpl_{dim_prot_complx_nm}_fpi_{fixed_prot_id}_mpi_{mut_prot_id}.csv'), index=False)
        print(f"##### {fix_mut_prot_id_tag} Please call AlphaFold2 for the structure prediction of mutated seuqnce(s). #####")
        print('\n#########################################################')
        print(f"           NEXT CALL postproc_2_mut_seq_struct_pred_by_af2.")
        print('#########################################################\n')
    # end of for loop: for chain_combo_itr in range(len(chain_combo_lst)):   
    print('inside run_postproc_1_part2_psiBlastOnClusteredData() method - End')
# End of run_postproc_1_part2_psiBlastOnClusteredData() method


def apply_postproc_psi_blast(**kwargs):
    simulation_exec_mode = kwargs.get('simulation_exec_mode')
    postproc_psiblast_exec_path = kwargs.get('postproc_psiblast_exec_path'); postproc_psiblast_uniprot_db_path = kwargs.get('postproc_psiblast_uniprot_db_path')
    postproc_psiblast_result_dir = kwargs.get('postproc_psiblast_result_dir')
    fix_mut_prot_id_tag = kwargs.get('fix_mut_prot_id_tag'); prob_thresholded_df = kwargs.get('prob_thresholded_df')
    dim_prot_complx_nm = kwargs.get('dim_prot_complx_nm'); curnt_chain_combo = kwargs.get('curnt_chain_combo')
    psiBlast_percent_similarity_score_threshold = kwargs.get('psiBlast_percent_similarity_score_threshold')
    max_num_of_seq_below_psiBlast_sim_score_threshold = kwargs.get('max_num_of_seq_below_psiBlast_sim_score_threshold')

    psi_blast_thresholded_df = None
    psi_blast_thresholded_dict_lst = []
    # The below counter will track the running number of mutated sequecnces satisfying PSI-Blast criterion, so that the loop iteration can be stopped when 'max_num_of_seq_below_psiBlast_sim_score_threshold' is reached.
    crnt_successful_seq_count = 0
    # Iterate through prob_thresholded_df and carry out PSI-Blast
    print(f"\n{fix_mut_prot_id_tag} Iterate through prob_thresholded_df and carry out PSI-Blast).")
    for index, row in prob_thresholded_df.iterrows():
        row_dict = row.to_dict()
        print(f"\n{fix_mut_prot_id_tag} row_dict: {row_dict}")
        sequence = row_dict['prot_seq']
        # Declare a flag to check whether all iterations result need to be processed because if in any 
        # intermediate iteration the 'percent_similarity_score' is greater than the threshold for any high-scoring-segment-pair (hsp), then
        # there is no need to proceed further for that row of prob_thresholded_df
        skip_psi_blast_for_crnt_row = False
        # create psi-blast result xml file name
        psi_blast_result_xml_file_nm = None
        if(simulation_exec_mode == 'remc'):
            psi_blast_result_xml_file_nm = os.path.join(postproc_psiblast_result_dir
                                        , f'cmplx_{dim_prot_complx_nm}_{curnt_chain_combo}_wrapper_step_idx_{row_dict["wrapper_step_idx"]}_replica_index_{row_dict["replica_index"]}_batch_idx_{row_dict["batch_idx"]}_simuln_idx_{row_dict["simuln_idx"]}_blast_res.xml')
        else:
            # For mcmc
            psi_blast_result_xml_file_nm = os.path.join(postproc_psiblast_result_dir
                                        , f'cmplx_{dim_prot_complx_nm}_{curnt_chain_combo}_batch_idx_{row_dict["batch_idx"]}_simuln_idx_{row_dict["simuln_idx"]}_blast_res.xml')
        # end of if-else block
        query_sequence = f">query_sequence\n{sequence}"
        psiblast_cmd = [
            f'{postproc_psiblast_exec_path}psiblast',
            '-db', 'uniprotSprotFull_v2',
            '-evalue', str(0.001),
            '-num_iterations', str(4),  # Number of iterations for PSI-BLAST
            '-outfmt', str(5),  # Output format (XML)
            '-out', psi_blast_result_xml_file_nm,  # Output file for results
            '-num_threads', str(4)
        ]
        try:
            # Run PSI-BLAST with input from the query_sequence and redirect the output to a file
            # result = subprocess.run(psiblast_cmd, input=query_sequence, text=True, capture_output=True)
            result = subprocess.run(psiblast_cmd, input=query_sequence, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Check for errors in the PSI-BLAST execution
            if result.returncode != 0:
                # print(f"Error running PSI-BLAST: {result.stderr}")
                raise Exception(f"!!! Error running PSI-BLAST: {result.stderr}")
            
            # Parse the PSI-BLAST results from the XML output file
            with open(psi_blast_result_xml_file_nm) as result_handle:
                blast_records = NCBIXML.parse(result_handle)
                # Process each blast record
                for record in blast_records:
                    # Check for hits in the output
                    if not record.alignments:
                        print(f"{fix_mut_prot_id_tag} *** No hits found for the given query sequence.")
                        # ### raise Exception(f"No hits found")
                    # Process each alignment
                    for alignment in record.alignments:
                        for hsp in alignment.hsps:
                            # Calculate percent similarity score and expectation value
                            # The percent similarity score is calculated by dividing the number of identical matches (hsp.identities) by 
                            # the alignment length (hsp.align_length) and multiplying by 100
                            percent_similarity_score = (hsp.identities / hsp.align_length) * 100.0
                            # print(f'\n{fix_mut_prot_id_tag} PSI-Blast: percent_similarity_score = {percent_similarity_score}')
                            # check if the 'percent_similarity_score' is greater than the threshold for any high-scoring-segment-pair (hsp) 
                            if(percent_similarity_score > psiBlast_percent_similarity_score_threshold):
                                # Then there is no need to proceed further for that row of prob_thresholded_df
                                skip_psi_blast_for_crnt_row = True
                                print(f'\n{fix_mut_prot_id_tag} ******* Skipping this row of prob_thresholded_df as \
                                    \npercent_similarity_score = {percent_similarity_score} ( > {psiBlast_percent_similarity_score_threshold}) for one of the alignments.\n')
                                break                            
                        # end of for loop: for hsp in alignment.hsps:
                        if(skip_psi_blast_for_crnt_row): break
                    # end of for loop: for alignment in record.alignments:
                    if(skip_psi_blast_for_crnt_row): break
                # end of for loop: for record in blast_records:
            # end of with block
            if(not skip_psi_blast_for_crnt_row):
                # Accept this mutated sequence for further processing
                psi_blast_thresholded_dict_lst.append(row_dict)
                # Increase successful sequence count
                crnt_successful_seq_count += 1
                # Check whether the current running count of successful mutated sequences reaches the maximum limit
                if(crnt_successful_seq_count == max_num_of_seq_below_psiBlast_sim_score_threshold):
                    # skip rest of the sequence processing
                    print(f'\n{fix_mut_prot_id_tag} As crnt_successful_seq_count reaches "max_num_of_seq_below_psiBlast_sim_score_threshold" {max_num_of_seq_below_psiBlast_sim_score_threshold}, skipping the rest of the mutated sequences for PSI-Blast based similarity score checking.')
                    break
        except Exception as ex:
            print(f"*********** !!!! apply_postproc_psi_blast: An exception occurred: {ex}")
        # end of try-except block
    # end of for loop: for index, row in prob_thresholded_df.iterrows():

    # If psi_blast_thresholded_dict_lst contains at least one entry, then create dataframe and return it.
    # Otherwise return None.
    if(len(psi_blast_thresholded_dict_lst) > 0):
        psi_blast_thresholded_df = pd.DataFrame(psi_blast_thresholded_dict_lst)
    return psi_blast_thresholded_df
# End of apply_postproc_psi_blast() method


