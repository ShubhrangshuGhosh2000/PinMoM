import os, sys

from pathlib import Path
path_root = Path(__file__).parents[3]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

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
from sword2 import sword2
import torch.multiprocessing as mp


def run_prot_design_simuln(**kwargs):
    """The actual experiemnt for the protein design through PPI.

    This method is called from a triggering method like mat_p2ip_pd_trigger.trigger_experiment().
    """
    # ## Set the start method to 'spawn'
    # ### mp.set_start_method('spawn')
    
    print('inside run_prot_design_simuln() method - Start')
    print('####################################')
    # Iterate over kwargs and raise ValueError if any of the input arguments (except a few) is None. Also print each keyword argument name and respective value.
    for arg_name, arg_value in kwargs.items():
        if((arg_value is None) and (arg_name not in ['temperature_scheduler', 'resource_monitor', 'mutation_number_adjuster'])):
            raise ValueError(f"Argument '{arg_name}' must be provided with a value.")
        print(f"'{arg_name}': {arg_value}")
    # end of for loop: for arg_name, arg_value in kwargs.items():

    # retrieve all the keyword arguments
    root_path = kwargs.get('root_path'); dim_prot_complx_nm = kwargs.get('dim_prot_complx_nm'); use_prot_unit=kwargs.get('use_prot_unit'); num_of_itr = kwargs.get('num_of_itr')
    percent_len_for_calc_mut_pts = kwargs.get('percent_len_for_calc_mut_pts'); exec_mode_type = kwargs.get('exec_mode_type'); plm_file_location = kwargs.get('plm_file_location')
    plm_name = kwargs.get('plm_name'); batch_size = kwargs.get('batch_size'); batch_dump_interval = kwargs.get('batch_dump_interval')
    result_dump_dir = kwargs.get('result_dump_dir'); cuda_index = kwargs.get('cuda_index')
    use_psiblast_for_pssm = kwargs.get('use_psiblast_for_pssm'); psiblast_exec_path = kwargs.get('psiblast_exec_path')
    pdb_file_location = kwargs.get('pdb_file_location')
    mut_only_at_intrfc_resid_idx = kwargs.get('mut_only_at_intrfc_resid_idx'); naccess_path = kwargs.get('naccess_path')
    earlyStoppingCriteria_inst = kwargs.get('early_stop_checkpoint')
    fixed_temp_mcmc = kwargs.get('fixed_temp_mcmc')
    temperatureScheduler_inst = kwargs.get('temperature_scheduler')
    resourceMonitor_inst = kwargs.get('resource_monitor')
    mutationNumberAdjuster_inst = kwargs.get('mutation_number_adjuster')
    max_thrshld_for_num_of_mut_pts = kwargs.get('max_thrshld_for_num_of_mut_pts')
    temp_min_remc = kwargs.get('temp_min_remc'); temp_max_remc = kwargs.get('temp_max_remc'); num_of_replica_remc = kwargs.get('num_of_replica_remc')
    result_save_interval_remc = kwargs.get('result_save_interval_remc')
    print('####################################')

    # use del x1, x2 and  gc.collect() as and when required
    aa_lst = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    # Determine the number of available CPU cores
    num_cpu_cores = multiprocessing.cpu_count()
    # Store original value of 'percent_len_for_calc_mut_pts' as it will be used to reset while starting the second chain combo processing
    percent_len_for_calc_mut_pts_orig = percent_len_for_calc_mut_pts

    print('\n################# Processing dimeric protein complex- Start\n')
    # Extracts chain names and sequences from a protein complex PDB file.
    chain_sequences_dict = prot_design_util.extract_chain_sequences(dim_prot_complx_nm, pdb_file_location)
    pdb_chain_nm_lst = list(chain_sequences_dict.keys())
    # generate list of tuples containing all possible combinations of chain names from the chain_sequences_dict where
    # the tuple containing the same chain names is not allowed
    unique_chain_name_tuples_lst = prot_design_util.gen_unique_tuples_from_keys(chain_sequences_dict)
    print('\n################# Processing dimeric protein complex- End\n')

    chain_nm_pu_dict = {}
    if(use_prot_unit and (not mut_only_at_intrfc_resid_idx)):
        print(f'As "use_prot_unit" is True and "mut_only_at_intrfc_resid_idx" is False, calculating PU for chains ...')
        # estimate num_pu
        inp_file = os.path.join(pdb_file_location, f"{dim_prot_complx_nm}.pdb")
        output_dir = os.path.join(root_path, "dataset/sword2_result")
        # chain_nm_pu_dict contains chain name as key and correponding PU list as value
        for pdb_chain_nm in pdb_chain_nm_lst:
            print(f'\n########## Calculating PU for chain {pdb_chain_nm} - Start')
            # invoking customized SWORD2 method for only PU calculation
            last_lvl_PUs_lst = sword2.main(inp_file, pdb_chain_nm, output_dir)
            chain_nm_pu_dict[pdb_chain_nm] = last_lvl_PUs_lst
            print(f'########## Calculating PU for chain {pdb_chain_nm} - End')
        # end of for loop: for pdb_chain_nm in pdb_chain_nm_lst:
        print(f"\n chain_nm_pu_dict:\n {chain_nm_pu_dict}")

    intrfc_resid_idx_dict = {}
    if(mut_only_at_intrfc_resid_idx):
        print(f'As "mut_only_at_intrfc_resid_idx" is True, finding interface residue index positions for dimer chains ...')
        inp_pdb_file = os.path.join(pdb_file_location, f"{dim_prot_complx_nm}.pdb")
        intrfc_resid_idx_dict = interface_residue.find_interface_residue_indices_for_dimer(root_path=root_path, pdb_file=inp_pdb_file
                                                                                  , chain_nm_lst=pdb_chain_nm_lst, naccess_path=naccess_path)
        print(f"intrfc_resid_idx_dict:\n {intrfc_resid_idx_dict}")

    print('\n################# One time model loading - Start\n')
    # ## RANDOM SEED IS ALREADY SET IN dl_reproducible_result_util import
    # seed_value = 456
    # # Set the seed for reproducibility
    # random.seed(seed_value)

    # initialize device
    device = torch.device(f'cuda:{cuda_index}')
    # load the protTrans PLM model -Start
    print('loading the protTrans PLM model -Start') 
    protTrans_model, tokenizer = preproc_plm_util.load_protTrans_model(protTrans_model_path=plm_file_location
                        , protTrans_model_name=plm_name, device=device)
    print('loading the protTrans PLM model -End') 
    # load the protTrans PLM model -End

    # set model_type based on exec_mode_type
    model_type = 'matds_hybrid' if(exec_mode_type == 'thorough') else 'matpip'

    # load MaTPIP model
    matpip_model = None
    matpip_model = matpip_RunTests.load_matpip_model(root_path)
    print('loaded MaTPIP model') 

    # if model_type is 'matds_hybrid', also load D-SCRIPT model and pre-trained Bepler-Berger model
    dscript_model = None
    if(model_type == 'matds_hybrid'):
        dscript_model = matpip_RunTests.load_dscript_model(root_path)
        pretrained_bb_model = pretrained.get_pretrained("lm_v1")
        pretrained_bb_model = pretrained_bb_model.cuda()
        pretrained_bb_model.eval()
        alphabet = alphabets.Uniprot21()
        print('loaded D-SCRIPT model and pre-trained Bepler-Berger model') 
    print('\n################# One time model loading - End\n')

    # Iterate over unique_chain_name_tuples_lst for each pair of (fixed_prot_id, mut_prot_id)
    print('\n################# Iterating over unique_chain_name_tuples_lst for each pair of (fixed_prot_id, mut_prot_id) - Start')
    for unique_chain_name_tuple_idx in range(len(unique_chain_name_tuples_lst)):
    # ################ for unique_chain_name_tuple_idx in range(1,len(unique_chain_name_tuples_lst)): ################ TEMP CODE FOR AD-HOC TESTING
        unique_chain_name_tuple = unique_chain_name_tuples_lst[unique_chain_name_tuple_idx]
        fixed_prot_id, mut_prot_id = unique_chain_name_tuple
        # fix_mut_prot_id_tag is a string to be used in the subsequent print statements to keep track of the specific iteration
        fix_mut_prot_id_tag = f' complex_{dim_prot_complx_nm}_fixed_prot_id_{fixed_prot_id}_mut_prot_id_{mut_prot_id}:: '
        print(fix_mut_prot_id_tag)
        # fix_mut_prot_id is used in case of PSSM feature (2D-manual-feature) extraction through extract_prot_seq_2D_manual_feat() method
        fix_mut_prot_id = f'cmplx_{dim_prot_complx_nm}_fixed_{fixed_prot_id}_mut_{mut_prot_id}'
        # Reset percent_len_for_calc_mut_pts
        percent_len_for_calc_mut_pts = percent_len_for_calc_mut_pts_orig

        print(f'\n{fix_mut_prot_id_tag}################# Processing batch size- Start\n')
        # Sets batch_size as it depends on the value of use_prot_unit (True/False) and mut_only_at_intrfc_resid_idx. If 'use_prot_unit' is True and 
        # 'mut_only_at_intrfc_resid_idx' is false, then 'batch_size' input argument value is ignored and batch_size is set to the number of PU.
        if(use_prot_unit and (not mut_only_at_intrfc_resid_idx)):
            print(f"{fix_mut_prot_id_tag}**** As use_prot_unit is True and mut_only_at_intrfc_resid_idx is False, 'batch_size' input argument value is ignored and batch_size is set to the number of PU.")
            # set batch_size = num_pu for mut_prot_id
            batch_size = len(chain_nm_pu_dict[mut_prot_id])
            print(f'{fix_mut_prot_id_tag}latest batch_size = num_pu for mut_prot_id ({mut_prot_id}) = {batch_size}')

        # check for num_of_itr to be a multiple of batch_size
        if(num_of_itr % batch_size != 0):
            print(f'{fix_mut_prot_id_tag}**** Number of iterations ({num_of_itr}) is not a multiple of batch size ({batch_size}), so\
                  \n reducing the number of iterations to be exact multiple of batch size')
            num_of_itr = batch_size * int(num_of_itr / batch_size)
            print(f'{fix_mut_prot_id_tag}modified num_of_itr = {num_of_itr}')
        # derive the number of batches
        # num_of_batches = int(num_of_itr / batch_size)  # ############# Not needed at this stage in REMC
        print(f'\n{fix_mut_prot_id_tag}################# Processing batch size- End\n')

        print(f'\n{fix_mut_prot_id_tag}################# Creating fresh result_dump_dir- Start\n')
        # Create result_dump_dir for this iteration
        iter_spec_result_dump_dir = os.path.join(result_dump_dir, f'complex_{dim_prot_complx_nm}/fixed_{fixed_prot_id}_mut_{mut_prot_id}'
                                                 , f'res_totItr{num_of_itr}_batchSz{batch_size}_percLnForMutPts{percent_len_for_calc_mut_pts}_pu{use_prot_unit}_mutIntrfc{mut_only_at_intrfc_resid_idx}')
        
        # Check whether result_dump_dir already exists. If exists, then skip this iteration
        if os.path.exists(iter_spec_result_dump_dir):
            print(f'{fix_mut_prot_id_tag}The specific result_dump_dir :\n {result_dump_dir} \n already exists. Hence skipping this pair of (fixed_prot_id, mut_prot_id) simulation')
            continue  # skipping to next (fixed_prot_id, mut_prot_id) simulation

        # if the result_dump_dir does not already exist, create it
        PPIPUtils.createFolder(iter_spec_result_dump_dir)
        print('\n################# Creating fresh result_dump_dir- End\n')
    
        print(f'\n{fix_mut_prot_id_tag}################# One time task- Start\n')
        # fetch the protein sequence for the given protein id
        fixed_prot_seq = chain_sequences_dict[fixed_prot_id]
        fixed_prot_seq = fixed_prot_seq.replace(" ", "").upper()
        mut_prot_seq = chain_sequences_dict[mut_prot_id]
        mut_prot_seq = mut_prot_seq.replace(" ", "").upper()
        mut_prot_len = len(mut_prot_seq)
        print(f'mut_prot_seq:{mut_prot_seq}')

        # # ############### extract the features for the fixed_prot_seq -Start ###############
        # ### extract PLM-based features for fixed_prot_seqTHC
        fixed_plm_1d_embedd_tensor_lst = preproc_plm_util.extract_protTrans_plm_feat(prot_seq_lst=[fixed_prot_seq], model=protTrans_model
                                                                                    , tokenizer=tokenizer, device=device)
        # ### extract manual features for fixed_prot_seq
        featureDir = os.path.join(root_path, 'dataset/preproc_data/derived_feat/')
        fixed_prot_fastas = [(fixed_prot_id, fixed_prot_seq)]  # ### approach can be taken to parallelly caculate manual 2d features for protein sequences in fastas
        # extract manual 2d features for fixed_prot_seq
        # read and cache blosum62.txt file content as it would be required for Blosum62 and PSSM(fallback option)feature
        # calculation (no need for the repeated read)
        blosumMatrix = PreprocessUtils.loadBlosum62()
        PPIPUtils.createFolder(os.path.join(featureDir, 'PSSM'))
        man_2d_feat_dict_prot_fixed = feat_engg_manual_main_pd.extract_prot_seq_2D_manual_feat(fix_mut_prot_id=fix_mut_prot_id, folderName=featureDir, fastas=fixed_prot_fastas
                                                                , skipgrm_lookup_prsnt = False, use_psiblast_for_pssm=use_psiblast_for_pssm, psiblast_exec_path=psiblast_exec_path
                                                                , labelEncode_lookup_prsnt = False, blosumMatrix=blosumMatrix)
        # extract manual 1d features for fixed_prot_seq
        man_1d_feat_dict_prot_fixed = feat_engg_manual_main_pd.extract_prot_seq_1D_manual_feat(fastas=fixed_prot_fastas)
        # ############### extract the features for the fixed_prot_seq -End ###############
        
        # # initialize a list to store batch-specific output dictionaries and
        # # it will be saved to disk as per result_save_interval_remc
        # batch_out_dict_lst = []  # ############# Not needed at this stage in REMC
        print(f'\n{fix_mut_prot_id_tag}################# One time task - End')

        # start the simulation
        print(f'\n{fix_mut_prot_id_tag}################# Starting the simulation')

        print(f'\n{fix_mut_prot_id_tag}################# Replica Exchange MC - Start')
        local_step_size = batch_dump_interval
        total_wrapper_steps = num_of_itr // local_step_size
        print(f'\n{fix_mut_prot_id_tag} REMC: Assigning temperature, initial sequence, probability and input arguments before starting parallel replica execution- Start')
        # Assign temperature to each replica
        replica_temp_lst = [0.0] * num_of_replica_remc
        for repl_indx in range(num_of_replica_remc):
            replica_temp_lst[repl_indx] = temp_min_remc + (repl_indx * ((temp_max_remc -temp_min_remc) / (num_of_replica_remc -1)))
        # end of for loop: for repl_indx in range(num_of_replica_remc):

        # Initially assign mutating sequence to each replica
        replica_seq_lst = [''] * num_of_replica_remc
        for repl_indx in range(num_of_replica_remc):
            replica_seq_lst[repl_indx] = mut_prot_seq
        # end of for loop: for repl_indx in range(num_of_replica_remc):

        # Initially assign stable probability to each replica
        replica_prob_lst = [0.0] * num_of_replica_remc
        for repl_indx in range(num_of_replica_remc):
            replica_prob_lst[repl_indx] = 0.0
        # end of for loop: for repl_indx in range(num_of_replica_remc):

        # Before parallel replica execution, prepare input argument dictionary for each replica.
        # Create a list of dictionaries for all the replicas.
        remc_inp_arg_dict_lst = [{}] * num_of_replica_remc
        for repl_indx in range(num_of_replica_remc):
            # First populate the fixed part of the dictioanry
            indiv_replica_arg_dict = copy.deepcopy(kwargs)
            # ####################################################################################
            indiv_replica_arg_dict['num_of_batches'] = local_step_size // batch_size
            # ####################################################################################
            indiv_replica_arg_dict['fixed_prot_id'] = fixed_prot_id
            indiv_replica_arg_dict['mut_prot_id'] = mut_prot_id
            indiv_replica_arg_dict['mut_prot_len'] = mut_prot_len
            indiv_replica_arg_dict['chain_nm_pu_dict'] = chain_nm_pu_dict
            indiv_replica_arg_dict['intrfc_resid_idx_dict'] = intrfc_resid_idx_dict
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

            # Next, populate the variable part of the dictioanry
            indiv_replica_arg_dict['replica_index_remc'] = repl_indx
            indiv_replica_arg_dict['replica_temp'] = replica_temp_lst[repl_indx]
            indiv_replica_arg_dict['replica_mut_seq'] = replica_seq_lst[repl_indx]
            indiv_replica_arg_dict['replica_prob'] = replica_prob_lst[repl_indx]

            # Append indiv_replica_arg_dict to the remc_inp_arg_dict_lst
            remc_inp_arg_dict_lst[repl_indx] = indiv_replica_arg_dict
        # end of for loop: for repl_indx in range(num_of_replica_remc):
        print(f'\n{fix_mut_prot_id_tag} REMC: Assigning temperature, initial sequence, probability and input arguments before starting parallel replica execution- End')

        print(f'\n #############################################################################################')
        print(f'{fix_mut_prot_id_tag} ##### For REMC, in our implementation, local_step_size = batch_dump_interval = {local_step_size} and num_of_batches = local_step_size ({local_step_size}) / batch_size ({batch_size}) = {local_step_size // batch_size}')
        print(f'#############################################################################################\n')
        
        print(f'\n{fix_mut_prot_id_tag} In Replica Exchange MC, executing {total_wrapper_steps} wrapper steps with {local_step_size} local steps in each ...')
        # Initialize a few lists required to save the result
        batch_out_df_lst, misc_info_df_lst, time_df_lst = [], [], []
        for wrapper_step_idx in range(total_wrapper_steps):
            print(f'\n{fix_mut_prot_id_tag} wrapper_step: {wrapper_step_idx} -Start')

            # Add wrapper_step_idx in each of the indiv_replica_arg_dict of remc_inp_arg_dict_lst
            for indiv_replica_arg_dict in remc_inp_arg_dict_lst: 
                indiv_replica_arg_dict['wrapper_step_idx'] = wrapper_step_idx

            # Call indiv_replica_work() method on all replicas in parallel
            # Use a Pool to run the function in parallel
            with mp.get_context('spawn').Pool(processes=num_of_replica_remc) as pool:
                # Map the arguments to the function
                result_dict_lst = pool.map(indiv_replica_work, remc_inp_arg_dict_lst)
            # end of with block
            print(f'\n{fix_mut_prot_id_tag} wrapper_step: {wrapper_step_idx}: After completion of all the replica execution in parallel, processing the result.')
            
            # Check result_dict_lst for any exception
            print(f'{fix_mut_prot_id_tag} wrapper_step: {wrapper_step_idx}: Checking multiprocessing execution result_dict_lst for any exception.' )
            for res_idx, result_dict in enumerate(result_dict_lst):
                if 'error' in result_dict:
                    print(f"{result_dict['fix_mut_prot_id_tag']} ###@@@ Error !!! Exception occurred: {result_dict['error']}")
                    print(f"{fix_mut_prot_id_tag} wrapper_step: {wrapper_step_idx}: So, not proceeding further in REMC process." )
                    return
                # end of if block: if 'error' in result_dict:
            # end of for loop: for res_idx, result_dict in enumerate(result_dict_lst):

            # If no error occurred in the multiprocessing execution, proceed further in REMC process
            print(f'{fix_mut_prot_id_tag} wrapper_step: {wrapper_step_idx}: No exception occurred in the multiprocessing execution. So, proceeding further in REMC process.' )
            print(f'{fix_mut_prot_id_tag} wrapper_step: {wrapper_step_idx}: Extract the result and check for Replica Exchange (if required) and prepare input for the next wrapper step.' )
            # Arrange result_dict_lst as per replica_index_remc because for multiprocessing workflow, the result_dict_lst may not be ordered by replica index
            ordered_result_dict_lst = [{}] * num_of_replica_remc  # Initialize with empty dictionaries
            for result_dict in result_dict_lst:
                replica_idx = result_dict['replica_index_remc']
                ordered_result_dict_lst[replica_idx] = result_dict

            # Check for Replica Exchange (if required) and prepare input for the next wrapper step
            # Note: Replicas are arranged in such a way so that R-1 has the lowest temperature and R-10 has the highest. So, R-1 has the lowest acceptance rate and 
            # R-10 has the highest acceptance rate. Ideally, R-1 should belong to the lower probability region and R-10 in the higher probability region. 
            # So, for any pair of consecutive replicas, if this expected probability behavior is not followed after the local steps (i.e. say, Prob(R-5) < Prob(R-4)),
            #  then replica exchange takes place. For our case stable_prob and last stable sequence are exchanged.
            # This can be achieved at the end of each wrapper_step, by sorting the list of (replica_prob, replica_mut_seq) tuples based on replica_prob in 
            # ascending order and set them in the replicas in the order from R-1 to R-10.

            # Create the list of (replica_prob, replica_mut_seq) tuples from the ordered_result_dict_lst
            repl_prob_seq_tupl_lst = [(0.0, '')] * num_of_replica_remc
            for repl_indx in range(num_of_replica_remc):
                repl_prob_seq_tupl_lst[repl_indx] = (ordered_result_dict_lst[repl_indx]['replica_prob'], ordered_result_dict_lst[repl_indx]['replica_mut_seq'])
            # end of for loop: for repl_indx in range(num_of_replica_remc):

            # Add the original indices to the tuples
            indexed_lst = [(orig_idx, tup) for orig_idx, tup in enumerate(repl_prob_seq_tupl_lst)]
            # Sort the list of tuples in ascending order based on the first element (replica_prob)
            sorted_indexed_lst = sorted(indexed_lst, key=lambda x: x[1][0])
            # Extract the new order and compare it with the original indices
            new_order = [original_index for original_index, _ in sorted_indexed_lst]
            # Find the positions where the original index has changed
            # 'changes' is a list of (original_index, new_index) tuples where original_index != new_index
            changes = [(new_order[i], i) for i in range(len(sorted_indexed_lst)) if new_order[i] != i]
            # Extract the sorted list
            sorted_repl_prob_seq_tupl_lst = [t for _, t in sorted_indexed_lst]
            # Set the (replica_prob, replica_mut_seq) tuples from the sorted list in the replicas in the order from R-1 to R-10
            for repl_indx in range(num_of_replica_remc):
                remc_inp_arg_dict_lst[repl_indx]['replica_prob'] = sorted_repl_prob_seq_tupl_lst[repl_indx][0]  # 0-th element of the tuple is replica_prob
                remc_inp_arg_dict_lst[repl_indx]['replica_mut_seq'] = sorted_repl_prob_seq_tupl_lst[repl_indx][1]  # 1-th element of the tuple is replica_mut_seq
            # End of for loop: for repl_indx in range(num_of_replica_remc):
            # If there is any replica exchange, then print the log
            if(len(changes) > 0):
                print(f'\n{fix_mut_prot_id_tag} wrapper_step: {wrapper_step_idx}: ##################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
                print(f'{fix_mut_prot_id_tag} original probabilities after wrapper step: {[(f"Repl-{entry[0]}", f"{entry[1][0]:.4f}") for entry in indexed_lst]}')
                print(f'{fix_mut_prot_id_tag} sorted probabilities after wrapper step:   {[("Repl-" + str(repl_indx), str(round(remc_inp_arg_dict_lst[repl_indx]["replica_prob"], ndigits=4))) for repl_indx in range(num_of_replica_remc)]}')
                print(f'{fix_mut_prot_id_tag} wrapper_step: {wrapper_step_idx}: The total number of replicas where final (replica_prob, replica_mut_seq) are swapped: {len(changes)}. \
                      \n Changes in positions (original index, new index): {changes}.')
                print(f'{fix_mut_prot_id_tag} wrapper_step: {wrapper_step_idx}: ##################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n')

            # ########### THE FOLLOWING CODE IS COMMENTED OUT IN FAVOR OF SORTING -START
            # for repl_indx in range(num_of_replica_remc - 1):
            #     # Retrieve probability associated with the current replica
            #     prob_crnt_replica = ordered_result_dict_lst[repl_indx]['replica_prob']
            #     # Retrieve probability associated with the next replica
            #     prob_next_replica = ordered_result_dict_lst[repl_indx + 1]['replica_prob']

            #     if(prob_next_replica < prob_crnt_replica):
            #         # Replica exchange is required
            #         print(f'\n {fix_mut_prot_id_tag} wrapper_step: {wrapper_step_idx}: ########@@@@@@@')
            #         print(f'{fix_mut_prot_id_tag} wrapper_step: {wrapper_step_idx}: prob of replica-{repl_indx + 1} ({prob_next_replica}) < prob of replica-{repl_indx} ({prob_crnt_replica}). ***** Replica exchange is required.')
            #         print(f'{fix_mut_prot_id_tag} wrapper_step: {wrapper_step_idx}: ########@@@@@@@\n')
            #         # Perform replica exchange
            #         mut_seq_crnt_replica = ordered_result_dict_lst[repl_indx]['replica_mut_seq']
            #         temp_prob = prob_crnt_replica; temp_seq = mut_seq_crnt_replica
            #         # Swap probabilities
            #         remc_inp_arg_dict_lst[repl_indx]['replica_prob'] = ordered_result_dict_lst[repl_indx + 1]['replica_prob']
            #         remc_inp_arg_dict_lst[repl_indx + 1]['replica_prob'] = temp_prob
            #         # Swap sequences
            #         remc_inp_arg_dict_lst[repl_indx]['replica_mut_seq'] = ordered_result_dict_lst[repl_indx + 1]['replica_mut_seq']
            #         remc_inp_arg_dict_lst[repl_indx + 1]['replica_mut_seq'] = temp_seq
            #     else:
            #         # Replica exchange is NOT required
            #         print(f'{fix_mut_prot_id_tag} wrapper_step: {wrapper_step_idx}: prob of replica-{repl_indx + 1} ({prob_next_replica}) >= prob of replica-{repl_indx} ({prob_crnt_replica}). Replica exchange is NOT required.')
            #         # Save probabilities and sequences as it is for the next wrapper step
            #         remc_inp_arg_dict_lst[repl_indx]['replica_prob'] = ordered_result_dict_lst[repl_indx]['replica_prob']
            #         remc_inp_arg_dict_lst[repl_indx + 1]['replica_prob'] = ordered_result_dict_lst[repl_indx + 1]['replica_prob']
            #         remc_inp_arg_dict_lst[repl_indx]['replica_mut_seq'] = ordered_result_dict_lst[repl_indx]['replica_mut_seq']
            #         remc_inp_arg_dict_lst[repl_indx + 1]['replica_mut_seq'] = ordered_result_dict_lst[repl_indx + 1]['replica_mut_seq']
            #     # End of if-else block: if(prob_next_replica < prob_crnt_replica):
            # # End of for loop: for repl_indx in range(num_of_replica_remc - 1):
            # ########### THE FOLLOWING CODE IS COMMENTED OUT IN FAVOR OF SORTING -END

            # Prepare the wrapper_step result for saving at 'result_save_interval_remc'
            for repl_indx in range(num_of_replica_remc):
                result_dict = ordered_result_dict_lst[repl_indx]
                if('batch_out_df' in result_dict):
                    batch_out_df_lst.append(result_dict['batch_out_df'])
                else:
                    print(f'\n{fix_mut_prot_id_tag} wrapper_step: {wrapper_step_idx}: NOT a single mutated sequence is accepted in wrapper_step: {wrapper_step_idx} for replica:{repl_indx}')

                misc_info_df_lst.append(result_dict['misc_info_df'])
                time_df_lst.append(result_dict['time_df'])

            # check to save the batch_out_df_lst to disk as per result_save_interval_remc
            if((wrapper_step_idx + 1) % result_save_interval_remc == 0):
                if(len(batch_out_df_lst) > 0):  # check to see if batch_out_df_lst is not empty, then only save it
                    t9_1 = time.time()
                    wrapper_out_df = pd.concat(batch_out_df_lst)
                    wrapper_out_csv_nm = os.path.join(iter_spec_result_dump_dir, f'wrapperStepIdx_{wrapper_step_idx + 1 - result_save_interval_remc}_{wrapper_step_idx}_localStepSize_{local_step_size}.csv')
                    wrapper_out_df.to_csv(wrapper_out_csv_nm, index=False)
                    # reset batch_out_df_lst
                    batch_out_df_lst = []
                    t9_2 = time.time()
                    print(f'\n{fix_mut_prot_id_tag} wrapper_step: {wrapper_step_idx}: :: Saving batch_out_df_lst to disk :: time (in sec) for execution = {round(t9_2-t9_1, 3)}\n')
                # end of if block: if(len(batch_out_df_lst) > 0)
            # end of if block
            print(f'\n{fix_mut_prot_id_tag} wrapper_step: {wrapper_step_idx} -End')
        # end of for loop: for wrapper_step_idx in range(total_wrapper_steps):

        # Save misc_info_df_lst to disk at the end of all wrapper steps
        con_misc_info_df = pd.concat(misc_info_df_lst)
        con_misc_info_csv_nm = os.path.join(iter_spec_result_dump_dir, f'misc_info.csv')
        con_misc_info_df.to_csv(con_misc_info_csv_nm, index=False)
        
        # Save time_df_lst to disk at the end of all wrapper steps
        con_time_df = pd.concat(time_df_lst)
        con_time_csv_nm = os.path.join(iter_spec_result_dump_dir, f'time_records.csv')
        con_time_df.to_csv(con_time_csv_nm, index=False)
        
        print(f'\n{fix_mut_prot_id_tag}################# Replica Exchange MC - End')

    # end of for loop: for unique_chain_name_tuple_idx in range(len(unique_chain_name_tuples_lst)):
    print('\n################# Iterating over unique_chain_name_tuples_lst for each pair of (fixed_prot_id, mut_prot_id) - End')
    print('inside run_prot_design_simuln() method - End')



def indiv_replica_work(indiv_replica_arg_dict):
    # Iterate over indiv_replica_arg_dict and print each keyword argument name and respective value.
    for arg_name, arg_value in indiv_replica_arg_dict.items():
        if(arg_name in ['dim_prot_complx_nm', 'num_of_batches', 'fixed_prot_id', 'mut_prot_id', 'mut_prot_len', 'model_type'
                        , 'iter_spec_result_dump_dir', 'replica_index_remc', 'replica_temp', 'replica_prob', 'wrapper_step_idx']):
            print(f"'{arg_name}': {arg_value}")
    # end of for loop: for arg_name, arg_value in indiv_replica_arg_dict.items():

    # retrieve all the keyword arguments
    root_path = indiv_replica_arg_dict.get('root_path'); dim_prot_complx_nm = indiv_replica_arg_dict.get('dim_prot_complx_nm'); use_prot_unit=indiv_replica_arg_dict.get('use_prot_unit'); num_of_itr = indiv_replica_arg_dict.get('num_of_itr')
    percent_len_for_calc_mut_pts = indiv_replica_arg_dict.get('percent_len_for_calc_mut_pts'); exec_mode_type = indiv_replica_arg_dict.get('exec_mode_type'); plm_file_location = indiv_replica_arg_dict.get('plm_file_location')
    plm_name = indiv_replica_arg_dict.get('plm_name'); batch_size = indiv_replica_arg_dict.get('batch_size'); batch_dump_interval = indiv_replica_arg_dict.get('batch_dump_interval')
    result_dump_dir = indiv_replica_arg_dict.get('result_dump_dir'); cuda_index = indiv_replica_arg_dict.get('cuda_index')
    use_psiblast_for_pssm = indiv_replica_arg_dict.get('use_psiblast_for_pssm'); psiblast_exec_path = indiv_replica_arg_dict.get('psiblast_exec_path')
    pdb_file_location = indiv_replica_arg_dict.get('pdb_file_location')
    mut_only_at_intrfc_resid_idx = indiv_replica_arg_dict.get('mut_only_at_intrfc_resid_idx'); naccess_path = indiv_replica_arg_dict.get('naccess_path')
    earlyStoppingCriteria_inst = indiv_replica_arg_dict.get('early_stop_checkpoint')
    fixed_temp_mcmc = indiv_replica_arg_dict.get('fixed_temp_mcmc')
    temperatureScheduler_inst = indiv_replica_arg_dict.get('temperature_scheduler')
    resourceMonitor_inst = indiv_replica_arg_dict.get('resource_monitor')
    mutationNumberAdjuster_inst = indiv_replica_arg_dict.get('mutation_number_adjuster')
    max_thrshld_for_num_of_mut_pts = indiv_replica_arg_dict.get('max_thrshld_for_num_of_mut_pts')

    # Retrive additional arguments stored in indiv_replica_arg_dict
    num_of_batches = indiv_replica_arg_dict['num_of_batches']
    fixed_prot_id = indiv_replica_arg_dict['fixed_prot_id']
    mut_prot_id = indiv_replica_arg_dict['mut_prot_id']
    mut_prot_len = indiv_replica_arg_dict['mut_prot_len']
    chain_nm_pu_dict = indiv_replica_arg_dict['chain_nm_pu_dict']
    intrfc_resid_idx_dict = indiv_replica_arg_dict['intrfc_resid_idx_dict']
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
    iter_spec_result_dump_dir = indiv_replica_arg_dict['iter_spec_result_dump_dir']


    # Next, populate the variable part of the dictioanry
    replica_index_remc = indiv_replica_arg_dict['replica_index_remc']
    replica_temp = indiv_replica_arg_dict['replica_temp']
    replica_mut_seq = indiv_replica_arg_dict['replica_mut_seq']
    replica_prob = indiv_replica_arg_dict['replica_prob']
    
    wrapper_step_idx = indiv_replica_arg_dict['wrapper_step_idx']
    # modify fix_mut_prot_id_tag
    fix_mut_prot_id_tag = f' complex_{dim_prot_complx_nm}_fixed_prot_id_{fixed_prot_id}_mut_prot_id_{mut_prot_id}_replica_{replica_index_remc}_wrapper_step_{wrapper_step_idx}:: '
    
    # Enclose the main processing in a try-catch block
    try:
        print(f'\n{fix_mut_prot_id_tag} ##### For REMC, in our implementation, local_step_size = batch_dump_interval = {batch_dump_interval} and num_of_batches = local_step_size ({batch_dump_interval}) / batch_size ({batch_size}) = {batch_dump_interval // batch_size}')
        print(f'\n{fix_mut_prot_id_tag} ####################  Starting individual replica execution through indiv_replica_work() method - Start\n')
        # initialize a dictionary to store the output from this method which will be returned back
        out_replica_work_dict = {}
        # Store replica_index_remc, replica_temp, replica_mut_seq, replica_prob, wrapper_step_idx in the out_replica_work_dict
        out_replica_work_dict['replica_index_remc'] = replica_index_remc; out_replica_work_dict['replica_temp'] = replica_temp; out_replica_work_dict['replica_mut_seq'] = replica_mut_seq
        out_replica_work_dict['replica_prob'] = replica_prob; out_replica_work_dict['wrapper_step_idx'] = wrapper_step_idx; 

        # initialize a list to store batch-specific output dictionaries
        batch_out_dict_lst = []

        t_entire_init = time.time()
        # for tracking time of execution
        crit_lst, exec_time_lst = [], []
        misc_info_lst, misc_info_val_lst = [], []
        max_cpu_cores_used = 1; max_gpu_cores_used = 1
        
        mut_idx = 0
        # initialize prv_best_mutated_prot_seq (This sequence can be either the previous mutated sequence with maximum PPI probability or the last MCMC satifying mutated sequence).
        mut_prot_seq = replica_mut_seq
        prv_best_mutated_prot_seq = mut_prot_seq
        # initialize p_stable
        # p_stable = -1.0
        p_stable = replica_prob
        # temperature used in the MCMC (Monte Carlo simulation with Metropolis Criteria)
        # It is initiated with 'fixed_temp_mcmc' value and later is updated by temperature scheduler (if temperatureScheduler_inst is not None).
        # t_mcmc = fixed_temp_mcmc
        t_mcmc = replica_temp
        # Flags related to early stopping criteria of batch iterations
        early_stop = False; early_stop_crit = None
        # Flags related to mutation number adjuster of batch iterations
        trigger_fixed_mut_num_mna = False;  enable_early_stopping_check = True
        
        # deep-copy original instance of earlyStoppingCriteria_inst, temperatureScheduler_inst, resourceMonitor_inst to
        # use for each iteration of (fixed_prot_id, mut_prot_id) simulation
        earlyStoppingCriteria_inst_dcopy, temperatureScheduler_inst_dcopy, resourceMonitor_inst_dcopy = None, None, None
        mutationNumberAdjuster_inst_dcopy = None

        earlyStoppingCriteria_inst_dcopy = copy.deepcopy(earlyStoppingCriteria_inst)
        if(temperatureScheduler_inst is not None):
            temperatureScheduler_inst_dcopy = copy.deepcopy(temperatureScheduler_inst)
        if(resourceMonitor_inst is not None):
            resourceMonitor_inst_dcopy = copy.deepcopy(resourceMonitor_inst)
        if(mutationNumberAdjuster_inst is not None):
            mutationNumberAdjuster_inst_dcopy = copy.deepcopy(mutationNumberAdjuster_inst)

        t_batch_init = time.time()
        for batch_idx in range(num_of_batches):
            t1 = time.time()
            # assume the PPI pair is (p1, p2) where p1=fixed protein and p2=second or mutated second protein
            # mut_prot_fastas contains p2 mutated 'batch_size' number of times

            # The following are reset after every batch - Start
            ppi_score_arr = np.empty((0,), dtype=float)
            mut_prot_fastas, mut_pos_lst, aa_idx_lst,  = [], [], []
            # The following are reset after every batch - End

            mod_batch_size = batch_size
            if(batch_idx == 0):
                # special handling for the 0th batch as the original p2 needs to be accommodated first.
                mut_prot_fastas.append((mut_prot_id, mut_prot_seq))
                mod_batch_size = batch_size -1
                # store the relevant information for the original p2
                mut_idx += 1
                mut_pos_lst.append([-1])  # special value for the original p2
                aa_idx_lst.append([-1])  # special value for the original p2

            # perform mutation 'mod_batch_size' number of times on previous best mutated sequence as obtained from the previous batch
            # Note: All the mutations will take place on prv_best_mutated_prot_seq even if mod_batch_size > 1
            for itr in range(mod_batch_size):
                # Decide about the range within the prv_best_mutated_prot_seq which will be considered for the mutation and this decision
                # will be based on the 'use_prot_unit' flag and 'mut_only_at_intrfc_resid_idx' flag.

                # Consider all possible combinations of use_prot_unit (True/False), mut_only_at_intrfc_resid_idx (True/False), batch_size, num_of_mut_pts as follows:
                # Case-1: use_prot_unit = False; mut_only_at_intrfc_resid_idx = False; batch_size = 1; num_of_mut_pts = n (n> =1) ==> 
                # In each iteration, 1 cadidate mutated protein sequence will be generated from the prv_best_mutated_prot_seq and
                # the mutated candidate will have n (>=1) mutation(s) within the range of entire prv_best_mutated_prot_seq.
                # 
                # Case-2: use_prot_unit = False; mut_only_at_intrfc_resid_idx = False; batch_size = b (b >1); num_of_mut_pts = n (n >=1) ==> 
                # In each iteration, b (>1) cadidate mutated protein sequences will be generated (may be in parallel) from the prv_best_mutated_prot_seq and 
                # each mutated candidate will have n (>=1) mutation(s) within the range of entire prv_best_mutated_prot_seq.
                # 
                # Case-3: use_prot_unit = True; mut_only_at_intrfc_resid_idx = False; batch_size = b (b >=1); num_of_mut_pts = n (n>=1) ==> In this case, user given batch_size 
                # is overwritten by the number of PU (u>=1) for mut_prot_id i.e. modified batch_size = u.
                # In each iteration, u (>=1) cadidate mutated protein sequences will be generated (may be in parallel) from the prv_best_mutated_prot_seq and 
                # each mutated candidate will have n (>=1) mutation(s) within the range of corresponding PU. e.g. Let's first PU has sequence range (1, 63). Then, 
                # first mutated candidate will have n (>=1) mutation(s) within the range (1, 63) of prv_best_mutated_prot_seq.
                # 
                # Case-4: use_prot_unit = True/False; mut_only_at_intrfc_resid_idx = True; batch_size = b (b >=1); num_of_mut_pts = n (n>=1) ==> 
                # In this case, 'use_prot_unit' flag will be ignored in favor of 'mut_only_at_intrfc_resid_idx' flag.
                # In each iteration, b (>=1) cadidate mutated protein sequences will be generated (may be in parallel) from the prv_best_mutated_prot_seq and 
                # each mutated candidate will have n (>=1) mutation(s) at the interface residue index positions.
                
                # For the ease of manipulation of mut_prot_seq which is a string, convert the string to a list of characters
                # Note: All the mutations will take place on prv_best_mutated_prot_seq even if mod_batch_size > 1
                prv_best_mutated_prot_seq_as_lst = list(prv_best_mutated_prot_seq)
                
                # set default range for the mutation as the entire mutating protein length
                mut_seq_range = range(mut_prot_len)
                if(use_prot_unit and (not mut_only_at_intrfc_resid_idx)):
                    # Check for the index of the current PU out of PU list for mut_prot_id and
                    # in this case, it is equal to 'itr' value of the enclosing for loop.
                    # Then set mut_seq_range as the range of corresponding PU.
                    seq_tuple_for_crnt_pu = chain_nm_pu_dict[mut_prot_id][itr]
                    mut_seq_range = range(seq_tuple_for_crnt_pu[0] -1, seq_tuple_for_crnt_pu[1])

                if(mut_only_at_intrfc_resid_idx):
                    # Retrieve interface residue index positions for the current mutating chain 
                    intrfc_resid_idx_lst_for_crnt_mut_chain = intrfc_resid_idx_dict[mut_prot_id]
                    mut_seq_range = intrfc_resid_idx_lst_for_crnt_mut_chain
                        
                # Check if 'percent_len_for_calc_mut_pts' is negative. It can be negative only if 
                # adjust_mutation_number() method of MutationNumberAdjuster class returns a negative 'percent_len_for_calc_mut_pts' and
                # it happens only if 'fixed mutation number' strategy is active.
                if(percent_len_for_calc_mut_pts < 0):
                    num_of_mut_pts = -1 * percent_len_for_calc_mut_pts
                else:
                    # Setting random number of muation points ranging from 1 to 'x_percent'
                    # no_of_mut_points = randomly from 1 to (x% of length) for Full length 
                    #                    randomly from 1 to (x% of number of interfacing residues) for Interface
                    x_percent_of_mut_prot_len = int(percent_len_for_calc_mut_pts / 100.0 * mut_prot_len)
                    if(mut_only_at_intrfc_resid_idx):
                        x_percent_of_mut_prot_len = int(percent_len_for_calc_mut_pts / 100.0 * len(intrfc_resid_idx_lst_for_crnt_mut_chain))

                    # Handle the scenario when x_percent_of_mut_prot_len = 0 which may arise if mut_prot_len=36 (say) and percent_len_for_calc_mut_pts=2
                    if(x_percent_of_mut_prot_len < 1):
                        x_percent_of_mut_prot_len = 1
                    # Take lesser among x_percent_of_mut_prot_len and max_thrshld_for_num_of_mut_pts
                    x_percent_of_mut_prot_len = min(x_percent_of_mut_prot_len, max_thrshld_for_num_of_mut_pts)
                    num_of_mut_pts = random.sample(range(1, (x_percent_of_mut_prot_len+1)), 1)[0]
                # end of if-else block: if(percent_len_for_calc_mut_pts < 0):
                # print(f'{fix_mut_prot_id_tag} batch_idx={batch_idx} :: mut_prot_len: {mut_prot_len} :: percent_len_for_calc_mut_pts: {percent_len_for_calc_mut_pts} \n num_of_mut_pts: {num_of_mut_pts}')
                print(f'{fix_mut_prot_id_tag} num_of_mut_pts: {num_of_mut_pts}')
                
                # randomly select the mutation points within mut_seq_range of the mutable protein sequence
                # Use random.sample to get unique random positions
                sel_mut_seq_pos_lst = random.sample(mut_seq_range, num_of_mut_pts)  # e.g. sel_mut_seq_pos_lst: [19, 16, 0] for num_of_mut_pts=3

                # Iterate through the selected mutation positions one by one
                sel_aa_lst_indices = []
                for sel_pos in sel_mut_seq_pos_lst:
                    # Find the amino-acid index which is currently present at the randomly selected position of the mut sequence
                    prv_aa_idx_at_sel_pos = aa_lst.index(prv_best_mutated_prot_seq_as_lst[sel_pos])
                    # Create the modified aa index list which will not contain the index of already present amino acid at the randomly selected position of the mut sequence.
                    mod_aa_idx_lst = [aa_idx for aa_idx in range(len(aa_lst)) if aa_idx not in [prv_aa_idx_at_sel_pos]]
                    
                    # Randomly select a single amino acid from the mod_aa_idx_lst
                    # Use random.sample to get unique random position
                    new_aa_idx_at_sel_pos = random.sample(mod_aa_idx_lst, 1)[0]
                    # Append the newly selected amino acid index in sel_aa_lst_indices
                    sel_aa_lst_indices.append(new_aa_idx_at_sel_pos)
                # end of for loop: for sel_pos in sel_mut_seq_pos_lst:

                # ##### ##### COMMENTED PREVIOUS VERSION OF THE MUTATION WHERE THERE WAS NO MEASURE TAKEN TO avoid the identical mutated sequence generation which 
                # may happen if the randomly selected amino-acid is same as that is already present in the respective replacement position -START
                # # randomly select the amino acids from the aa_lst
                # # Use random.sample to get unique random positions
                # sel_aa_lst_indices = random.sample(range(20), num_of_mut_pts)  # e.g. sel_aa_lst_indices: [1, 0, 4] for num_of_mut_pts=3
            
                # # For the ease of manipulation of mut_prot_seq which is a string, convert the string to a list of characters
                # # Note: All the mutations will take place on prv_best_mutated_prot_seq even if mod_batch_size > 1
                # prv_best_mutated_prot_seq_as_lst = list(prv_best_mutated_prot_seq)
                # ##### ##### COMMENTED PREVIOUS VERSION OF THE MUTATION WHERE THERE WAS NO MEASURE TAKEN TO avoid the identical mutated sequence generation which 
                # may happen if the randomly selected amino-acid is same as that is already present in the respective replacement position -END
                
                # generate the mutated sequence
                for i in range(num_of_mut_pts):
                    prv_best_mutated_prot_seq_as_lst[sel_mut_seq_pos_lst[i]] = aa_lst[sel_aa_lst_indices[i]]
                # convert back the list into a string
                mutated_prot_seq = "".join(prv_best_mutated_prot_seq_as_lst)
                # append it to mut_prot_fastas
                mut_prot_fastas.append((f'm{mut_idx}', mutated_prot_seq))
                # store the relevant information for this iteration
                mut_pos_lst.append(sel_mut_seq_pos_lst)
                aa_idx_lst.append(sel_aa_lst_indices)
                mut_idx += 1
            # end of for loop: for itr in range(mod_batch_size):
            t2 = time.time()
            print(f'{fix_mut_prot_id_tag} batch_idx={batch_idx} :: time (in sec) for performing mutation= {round(t2-t1, 3)}')
            
            # ############### extract the features for the mut_prot_fastas -Start ###############
            mut_prot_id_lst, mut_prot_seq_lst = [], []
            for mutated_prot_id, mutated_prot_seq in mut_prot_fastas:
                mut_prot_id_lst.append(mutated_prot_id)
                mut_prot_seq_lst.append(mutated_prot_seq)
            ### extract PLM-based features for mut_prot_fastas -Start
            mut_plm_1d_embedd_tensor_lst = preproc_plm_util.extract_protTrans_plm_feat(prot_seq_lst=mut_prot_seq_lst, model=protTrans_model
                                                                                    , tokenizer=tokenizer, device=device)
            # add PLM-based features for the fixed protein alongwith the mutated proteins
            # using dict() and zip() to convert lists to dictionary
            plm_feat_dict = dict(zip([fixed_prot_id] + mut_prot_id_lst, fixed_plm_1d_embedd_tensor_lst + mut_plm_1d_embedd_tensor_lst))
            t3 = time.time()
            print(f'{fix_mut_prot_id_tag} batch_idx={batch_idx} :: time (in sec) for PLM feature extraction = {round(t3-t2, 3)}')
            ### extract PLM-based features for mut_prot_fastas -End

            ### extract manual features for mut_prot_fastas -Start
            man_2d_feat_dict_prot = man_1d_feat_dict_prot = None
            # ##### Note: FOR REMC, 'extract_manual_feat_mp()' METHOD WILL NOT WORK. SO, CALLING 'extract_manual_feat_serially()' METHOD FOR ALL TYPES OF MANUAL FEATURE EXTRACTIONS.
            if(batch_size >= 1): 
                # extract manual feature serially 
                man_2d_feat_dict_prot, man_1d_feat_dict_prot = extract_manual_feat_serially(fix_mut_prot_id, featureDir, mut_prot_fastas, use_psiblast_for_pssm, psiblast_exec_path, blosumMatrix, mut_prot_id_lst)
            else:  # THIS ELSE BLOCK WILL NEVER BE USED IN REMC
                num_process = 0
                if((batch_size > 1) and (batch_size < num_cpu_cores)):
                    num_process = batch_size
                else:
                    # batch_size >= num_cpu_cores
                    num_process = num_cpu_cores - 2
                # end of if-else block
                # update max_cpu_cores_used conditionally
                if(num_process > max_cpu_cores_used):
                    max_cpu_cores_used = num_process

                # extract manual feature in multiprocessing fashion
                man_2d_feat_dict_prot, man_1d_feat_dict_prot = extract_manual_feat_mp(fix_mut_prot_id, featureDir, mut_prot_fastas, use_psiblast_for_pssm, psiblast_exec_path, blosumMatrix, mut_prot_id_lst, num_process)
            # end of if-else block: if(batch_size == 1):

            # add manual features for the fixed protein along with the mutated proteins
            man_2d_feat_dict_prot[fixed_prot_id] = man_2d_feat_dict_prot_fixed
            man_1d_feat_dict_prot[fixed_prot_id] = man_1d_feat_dict_prot_fixed
            t4 = time.time()
            print(f'{fix_mut_prot_id_tag} batch_idx={batch_idx} :: time (in sec) for manual feature extraction = {round(t4-t3, 3)}')
            ### extract manual features for mut_prot_fastas -End
            # ############### extract the features for the mut_prot_fastas -End ###############

            # ############### execute MaTPIP prediction and get ppi_score -Start ###############
            mcp = matpip_RunTests.execute(root_path, matpip_model, featureDir, fixed_prot_id, mut_prot_id_lst
                                                , plm_feat_dict, man_2d_feat_dict_prot, man_1d_feat_dict_prot)
            t5 = time.time()
            print(f'{fix_mut_prot_id_tag}batch_idx={batch_idx} :: MaTPIP :: time (in sec) for class prob generation = {round(t5-t4, 3)}')
            # ############### execute MaTPIP prediction and get ppi_score -End ###############
            # if model_type is 'matpip', horizontally stack 'mcp' to 'ppi_score_arr'
            if(model_type == 'matpip'):
                # horizontally stack 'mcp' to 'ppi_score_arr'
                ppi_score_arr = np.hstack((ppi_score_arr, mcp))
                t7 = time.time()
            else:
                # model_type is 'matds_hybrid', so generate the prediction by D-Script and then hybrid the result
                dscript_seqs = mut_prot_fastas + fixed_prot_fastas
                dscript_pairs = [(fixed_prot_id, mut_prot_id) for mut_prot_id in mut_prot_id_lst]
                threshold = 0.5
                dcp = predict.main_pd(pairs=dscript_pairs, model=dscript_model, pre_trn_mdl=pretrained_bb_model, alphabet=alphabet
                                                        , seqs=dscript_seqs, threshold=threshold)
                t6 = time.time()
                print(f'{fix_mut_prot_id_tag}batch_idx={batch_idx} :: D-SCRIPT :: time (in sec) for class prob generation = {round(t6-t5, 3)}')

                # call matds_hybrid algorithm
                hcpa = calc_matds_hybrid_score(mcp, dcp)
                t7 = time.time()
                print(f'{fix_mut_prot_id_tag}batch_idx={batch_idx} :: Hybrid :: time (in sec) for class prob generation = {round(t7-t6, 3)}')
                # horizontally stack 'hcpa' to 'ppi_score_arr'
                ppi_score_arr = np.hstack((ppi_score_arr, hcpa))
            # end of if-else block
            
            # ############### execute MCMC (Monte Carlo simulation with Metropolis Criteria) algo -Start ###############
            # p_stable corresponds to e_stable of MCMC algo. But the main difference is thet, in case
            # of energy, lesser is more stable but in case of class-1 probability score, larger is better.
            p_stable_changed = False
            # find the min value and the corresponding index in ppi_score_arr
            min_val_index = np.argmin(ppi_score_arr)
            min_value = ppi_score_arr[min_val_index]
            # treat min_value as p_i
            p_i = min_value

            # if temperatureScheduler_inst_dcopy is not None, then use temperature scheduler to obtain the temperature to be used in Metropolis Criteria (if required).
            # Note that irrepective of whether Metropolis Criteria step in MCMC is required for the current iteration, this temperature 
            # scheduler instance needs to be called to keep track of the scheduling
            if(temperatureScheduler_inst_dcopy != None):
                t_mcmc = temperatureScheduler_inst_dcopy.adjust_temperature(p_i)

            print(f'\n\n{fix_mut_prot_id_tag} !!!! p_i = {p_i}\n\n')

            # if p_i is more stable compared to p_stable, accept p_i and make it p_stable
            if(p_i > p_stable):
                # p_i is more stable than p_stable
                p_stable = p_i
                p_stable_changed = True
            else:
                # generate a random number between 0 and 1
                rand = np.random.rand()
                del_p = -(p_i - p_stable)  # del_p must be positive
                print(f'rand: {rand} :: p_i: {p_i} :: p_stable: {p_stable} : del_p: {del_p} : exp: {math.exp(-(del_p / t_mcmc))}')

                if(rand < math.exp(-(del_p / t_mcmc))):  # Metropolis Criteria
                    print(f'{fix_mut_prot_id_tag} @@@@@@@@@@@@@@@@@@@@@@@@@ MC satisfied @@@@@@@@@@@@@@@@@@@@@')
                    # accept p_i and make it as p_stable
                    p_stable = p_i
                    p_stable_changed = True
                else:
                    # no change in p_stable
                    p_stable_changed = False
                # end of inner if-else block
            # end of outer if-else block
            t8 = time.time()
            print(f'{fix_mut_prot_id_tag}batch_idx={batch_idx} :: MCMC :: time (in sec) for execution = {round(t8-t7, 3)}')
            # ############### execute MCMC (Monte Carlo simulation with Metropolis Criteria) algo -End ###############
            
            # Call adjust_mutation_number() method whose output will be used subsequently like 'percent_len_for_calc_mut_pts' will be used in the next batch and
            # 'trigger_fixed_mut_num_mna' and 'enable_early_stopping_check' will be used below.
            if(mutationNumberAdjuster_inst_dcopy != None):
                percent_len_for_calc_mut_pts, trigger_fixed_mut_num_mna,  enable_early_stopping_check = mutationNumberAdjuster_inst_dcopy.adjust_mutation_number(crnt_prob_value=p_i, crnt_batch_indx=batch_idx)

            # if p_stable is changed then only create batch specific output
            if(p_stable_changed):
                # create new batch output
                batch_out_dict = {'batch_idx': batch_idx}
                batch_out_dict['simuln_idx'] = (batch_size * batch_idx) + min_val_index
                batch_out_dict['mut_pos_lst'] = mut_pos_lst[min_val_index]
                batch_out_dict['aa_idx_lst'] = aa_idx_lst[min_val_index]
                batch_out_dict['ppi_score'] = p_i
                # ############# THE FOLLOWING LINE CAN BE COMMENTED OUT IF RESULT SAVING IS TAKING LONG TIME ############# #
                batch_out_dict['prot_seq'] = mut_prot_fastas[min_val_index][1]
                # append batch_out_dict into batch_out_dict_lst
                batch_out_dict_lst.append(batch_out_dict)
                # update prv_best_mutated_prot_seq
                prv_best_mutated_prot_seq = mut_prot_fastas[min_val_index][1]

                if(enable_early_stopping_check):
                    # check for the early stopping criteria of batch iterations
                    early_stop, early_stop_crit = earlyStoppingCriteria_inst_dcopy.check_early_stopping(p_i)
            # end of if block: if(p_stable_changed):
            print(f'\n{fix_mut_prot_id_tag}############# batch_idx={batch_idx} :: batch_size={batch_size} :: time (in sec) for specific batch execution = {round(t8-t1, 3)}\n')
            crit_lst.append(f'batch_idx_{batch_idx}'); exec_time_lst.append(round(t8-t1, 3))

            # Check if 'trigger_fixed_mut_num_mna' flag is set by adjust_mutation_number() method called above
            if(trigger_fixed_mut_num_mna):
                print(f'{fix_mut_prot_id_tag} @@@ trigger_fixed_mut_num_mna: {trigger_fixed_mut_num_mna}')
                # Reset prv_best_mutated_prot_seq to the original mutating chain sequence
                prv_best_mutated_prot_seq = mut_prot_seq
                p_stable = -1.0
                print(f'{fix_mut_prot_id_tag} @@@ prv_best_mutated_prot_seq = {prv_best_mutated_prot_seq}')

            # Check whether early stopping criteria is already met. If yes, then stop the batch iterations.
            if(enable_early_stopping_check and early_stop):
                break
            
            # Check to save the batch_out_dict_lst at the end of each wrapper_step i.e. when all the local_steps within a wrapper_step are 
            # executed and all the local_steps are executed only when all the batches as apecified by 'num_of_batches' are completed. 
            # Also do the resource (like RAM, GPU) usage monitoring (if enabled). Calculate time for saving and monitoring.
            if((batch_idx + 1) % num_of_batches == 0):
                batch_out_df = None
                if(len(batch_out_dict_lst) > 0):  # check to see if batch_out_dict_lst is not empty, then only save it
                    t9_1 = time.time()
                    batch_out_df = pd.DataFrame(batch_out_dict_lst)
                    # Also add 2 columns for replica_index_remc and wrapper_step_index_remc
                    batch_out_df.insert(loc=0, column='replica_index', value=[replica_index_remc] * batch_out_df.shape[0])
                    batch_out_df.insert(loc=0, column='wrapper_step_idx', value=[wrapper_step_idx] * batch_out_df.shape[0])

                    # Store batch_out_df in out_replica_work_dict
                    out_replica_work_dict['batch_out_df'] = batch_out_df
                    # Also update, 'replica_mut_seq' and 'replica_prob' with the latest value
                    out_replica_work_dict['replica_mut_seq'] = batch_out_dict_lst[-1]['prot_seq']
                    out_replica_work_dict['replica_prob'] = batch_out_dict_lst[-1]['ppi_score']

                    # #### batch_out_csv_nm = os.path.join(iter_spec_result_dump_dir, f'batchIdx_{batch_idx + 1 - batch_dump_interval}_{batch_idx}_batchSize_{batch_size}.csv')
                    # #### batch_out_df.to_csv(batch_out_csv_nm, index=False)
                    
                    # reset batch_out_dict_lst
                    # batch_out_dict_lst = []
                    # t9_2 = time.time()
                    # print(f'\n{fix_mut_prot_id_tag}batch_idx={batch_idx} :: Saving batch_out_dict_lst to disk :: time (in sec) for execution = {round(t9_2-t9_1, 3)}\n')

                    # check if resource usage monitoring is enabled
                    if(resourceMonitor_inst_dcopy != None):  # resource usage monitoring is enabled
                        t10_1 = time.time()
                        resourceMonitor_inst_dcopy.monitor_peak_ram_usage()
                        resourceMonitor_inst_dcopy.monitor_peak_gpu_memory_usage()
                        t10_2 = time.time()
                        print(f'\n{fix_mut_prot_id_tag}batch_idx={batch_idx} :: Resource (like RAM, GPU) usage monitoring :: time (in sec) for monitoring = {round(t10_2-t10_1, 3)}\n')
                # end of if block: if(len(batch_out_dict_lst) > 0)
            # end of if block
        # end of for loop: for batch_idx in range(num_of_batches):
        
        # # check if any remaining result is stored in batch_out_dict_lst, if yes, then save it
        # if(len(batch_out_dict_lst) > 0):
        #     # there is remaining result, stored in batch_out_dict_lst
        #     batch_out_df = pd.DataFrame(batch_out_dict_lst)
        #     batch_out_csv_nm = os.path.join(iter_spec_result_dump_dir, f'batchIdx_{batch_dump_interval * (batch_idx // batch_dump_interval)}_{batch_idx}_batchSize_{batch_size}.csv')
        #     batch_out_df.to_csv(batch_out_csv_nm, index=False)
        # # end of if block
        
        t_final = time.time()
        print(f'{fix_mut_prot_id_tag}Total batch execution time :: time (in sec) = {round(t_final - t_batch_init, 3)}')
        crit_lst.append('batch_total_time'); exec_time_lst.append(round(t_final - t_batch_init, 3))
        print(f'\n *********\n :: {fix_mut_prot_id_tag}Total time for complete simulation execution :: time (in sec) for execution = {round(t_final - t_entire_init, 3)} \n*********\n')
        crit_lst.append('simuln_total_time'); exec_time_lst.append(round(t_final - t_entire_init, 3))

        # Check for the condition of the batch iteration termination. Possible conditions are
        # (a) Normal stopping - All batch iterations completed (b) Early stopping - Prob Not Improved and (c) Early stopping - Prob Threshold Crossed
        end_condition = f'Normal stopping - All batch iterations ({num_of_batches}) completed'
        if(early_stop):
            if(early_stop_crit == 'NIM'):
                end_condition = f'Early stopping - Prob Not Improved for {earlyStoppingCriteria_inst_dcopy.stopping_patience} batches'
            elif(early_stop_crit == 'THC'):
                end_condition = f'Early stopping - Prob Threshold ({earlyStoppingCriteria_inst_dcopy.prob_threshold}) crossed'
        # end of if block: if(early_stop):
        print(f'Iteration end condition: {end_condition}')

        # save time records
        time_df = pd.DataFrame({'criterion': crit_lst, 'time_in_sec': exec_time_lst})
        # Also add 2 columns for replica_index_remc and wrapper_step_index_remc
        time_df.insert(loc=0, column='replica_index', value=[replica_index_remc] * time_df.shape[0])
        time_df.insert(loc=0, column='wrapper_step_idx', value=[wrapper_step_idx] * time_df.shape[0])
        # Store time_df in out_replica_work_dict
        out_replica_work_dict['time_df'] = time_df
        # time_df.to_csv(os.path.join(iter_spec_result_dump_dir, 'time_records.csv'), index=False)

        # populates miscellaneous information - Start
        print(f'{fix_mut_prot_id_tag} ############# Populating miscellaneous information - Start')
        # total number of batches executed
        misc_info_lst.append('tot_num_batches_executed'); misc_info_val_lst.append(batch_idx + 1)
        # batch_size
        misc_info_lst.append('batch_size'); misc_info_val_lst.append(batch_size)
        # total batch execution time
        misc_info_lst.append('batch_total_time_in_sec'); misc_info_val_lst.append(round(t_final - t_batch_init, 3))
        # total number of iterations executed
        misc_info_lst.append('tot_num_itr_executed'); misc_info_val_lst.append((batch_idx + 1) * batch_size)
        # # iteration end-condition
        # misc_info_lst.append('batch_itr_end_cond'); misc_info_val_lst.append(end_condition)
        # # store PU related info
        # misc_info_lst.append('PU_number'); misc_info_val_lst.append(batch_size if(use_prot_unit and (not mut_only_at_intrfc_resid_idx)) else -1)
        # # store max_cpu_cores_used, max_gpu_cores_used
        # misc_info_lst.append('max_cpu_cores_used'); misc_info_val_lst.append(max_cpu_cores_used)
        # misc_info_lst.append('max_gpu_cores_used'); misc_info_val_lst.append(max_gpu_cores_used)
        # # store peak_ram_usage
        # misc_info_lst.append('peak_ram_usage_in_GB'); misc_info_val_lst.append(round(resourceMonitor_inst_dcopy.peak_ram_usage / (1024 * 1024 * 1024.0), 3))
        # # store peak_gpu_usage
        # misc_info_lst.append('peak_gpu_usage_in_GB'); misc_info_val_lst.append(round(resourceMonitor_inst_dcopy.peak_gpu_usage / 1024.0, 3))

        # save misc info records
        misc_info_df = pd.DataFrame({'misc_info': misc_info_lst, 'misc_info_val': misc_info_val_lst})
        # Also add 2 columns for replica_index_remc and wrapper_step_index_remc
        misc_info_df.insert(loc=0, column='replica_index', value=[replica_index_remc] * misc_info_df.shape[0])
        misc_info_df.insert(loc=0, column='wrapper_step_idx', value=[wrapper_step_idx] * misc_info_df.shape[0])

        # Store misc_info_df in out_replica_work_dict
        out_replica_work_dict['misc_info_df'] = misc_info_df
        # #### misc_info_df.to_csv(os.path.join(iter_spec_result_dump_dir, 'misc_info.csv'), index=False)
        # populates miscellaneous information - End
        print(f'{fix_mut_prot_id_tag} ############# Populating miscellaneous information - End')
        
        print(f'{fix_mut_prot_id_tag} ####################  Starting individual replica execution through indiv_replica_work() method - End')
        # Return out_replica_work_dict
        return out_replica_work_dict
    except Exception as ex:
        print(f'{fix_mut_prot_id_tag} ###@@@ Error !!! Exception occurred: {ex}')
        error_dict = {}
        error_dict['fix_mut_prot_id_tag'] = fix_mut_prot_id_tag
        error_dict['wrapper_step_idx'] = wrapper_step_idx
        error_dict['replica_index'] = replica_index_remc
        error_dict['error'] = str(ex)
        return error_dict


def extract_manual_feat_serially(fix_mut_prot_id, featureDir, mut_prot_fastas, use_psiblast_for_pssm, psiblast_exec_path, blosumMatrix, mut_prot_id_lst):
    # ################################ serial version for the manual feature extraction -Start ################################
    # t3_1 = time.time()
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
    # end for loop: for i in range(len(mut_prot_id_lst)):
    # t3_2 = time.time()
    # print(f'serial version for the manual feature extraction: execution time (in sec.) = {round(t3_2-t3_1, 3)}')
    # ################################ serial version for the manual feature extraction -End ################################
    return (man_2d_feat_dict_prot, man_1d_feat_dict_prot)


def extract_manual_feat_mp(fix_mut_prot_id, featureDir, mut_prot_fastas, use_psiblast_for_pssm, psiblast_exec_path, blosumMatrix, mut_prot_id_lst, num_process):
    # ################################ multprocessing version for the manual feature extraction -Start ################################
    # Shared data structures
    manager = multiprocessing.Manager()
    man_2d_feat_dict_prot = manager.dict()  # shared_data1
    man_1d_feat_dict_prot = manager.dict()  # shared_data2

    # Create a lock to protect shared data structures
    shared_data_lock = manager.Lock()

    # Fixed data structures
    # fix_mut_prot_id, featureDir, mut_prot_fastas, skipgrm_lookup_prsnt=True, use_psiblast_for_pssm, psiblast_exec_path, labelEncode_lookup_prsnt=True, blosumMatrix
    skipgrm_lookup_prsnt = labelEncode_lookup_prsnt = True
    
    # t3_1 = time.time()
    # Using Pool for multiprocessing with dynamically determined processes
    with multiprocessing.Pool(processes=num_process) as pool:
        # Use starmap to pass multiple arguments to the ext_man_feat function
        pool.starmap(ext_man_feat
                     , [(idx, fix_mut_prot_id, featureDir, mut_prot_fastas, skipgrm_lookup_prsnt, use_psiblast_for_pssm, psiblast_exec_path, labelEncode_lookup_prsnt, blosumMatrix  # Fixed data structures
                         , man_2d_feat_dict_prot, man_1d_feat_dict_prot, shared_data_lock  # Shared data structures
                         ) for idx in range(len(mut_prot_id_lst))])
    
    # t3_2 = time.time()
    # print(f'after multiprocessing.pool(): execution time (in sec.) = {round(t3_2-t3_1, 3)}')
    ################################ multprocessing version for the manual feature extraction -End ################################
    return (man_2d_feat_dict_prot, man_1d_feat_dict_prot)


def ext_man_feat(idx, fix_mut_prot_id, featureDir, mut_prot_fastas, skipgrm_lookup_prsnt, use_psiblast_for_pssm, psiblast_exec_path, labelEncode_lookup_prsnt, blosumMatrix  # Fixed data structures
                             , man_2d_feat_dict_prot, man_1d_feat_dict_prot, shared_data_lock):  # Shared data structures
    
    # t1 = time.time()
    indiv_mut_prot_fastas = [mut_prot_fastas[idx]]
    man_2d_feat_dict_prot_mut = feat_engg_manual_main_pd.extract_prot_seq_2D_manual_feat(fix_mut_prot_id=fix_mut_prot_id, folderName=featureDir, fastas=indiv_mut_prot_fastas
                                                             , skipgrm_lookup_prsnt=skipgrm_lookup_prsnt
                                                             , use_psiblast_for_pssm=use_psiblast_for_pssm, psiblast_exec_path=psiblast_exec_path
                                                             , labelEncode_lookup_prsnt=labelEncode_lookup_prsnt, blosumMatrix=blosumMatrix)
    man_1d_feat_dict_prot_mut = feat_engg_manual_main_pd.extract_prot_seq_1D_manual_feat(fastas=indiv_mut_prot_fastas)

    # Acquire a lock to safely update the shared data structures
    with shared_data_lock:
        man_2d_feat_dict_prot[indiv_mut_prot_fastas[0][0]] = man_2d_feat_dict_prot_mut
        man_1d_feat_dict_prot[indiv_mut_prot_fastas[0][0]] = man_1d_feat_dict_prot_mut
    # t2 = time.time()
    # print(f'ext_man_feat: execution time (in sec.) = {round(t2-t1, 3)}')


def calc_matds_hybrid_score(mcp, dcp):
    mccp = 1 - mcp
    dccp = 1 - dcp
    hcpa = mcp
    hcpa = np.where(
        (mcp > 0.5) & (dccp > 0.5),  # if
        mcp - (dccp - 0.5),  # then
        np.where(  # else
            (mccp > 0.5) & (dcp > 0.5),  # if
            dcp - (mccp - 0.5),  # then
            hcpa  # else
        )
    )
    return hcpa
