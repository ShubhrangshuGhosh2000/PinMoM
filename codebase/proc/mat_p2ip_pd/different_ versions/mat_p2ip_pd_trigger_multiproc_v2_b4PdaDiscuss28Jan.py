import os, sys

from pathlib import Path
path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
import gc
import math
import multiprocessing
import numpy as np
import pandas as pd
import random
import time
import torch

from proc.mat_p2ip_DS.mat_p2ip_origMan_auxTlOtherMan import matpip_RunTests
from utils import prot_design_util, PPIPUtils
from utils import preproc_plm_util, PreprocessUtils
from utils import feat_engg_manual_main_pd
from dscript.commands import predict
from dscript import pretrained
from dscript import alphabets


def run_prot_design_simuln(root_path = './', fixed_prot_id = None, mut_prot_id = None, num_of_itr=300
                           , num_of_mut_pts = 1, model_type = 'matds_hybrid', protTrans_model_name='prot_t5_xl_uniref50'
                           , batch_size=5, batch_dump_interval=5, cuda_index=0, psiblast_exec_path='./', result_dump_dir_nm='./'):
    """The actual experiemnt for the protein design through PPI

    Args:
        root_path (str, optional): Project root path. Defaults to './'.
        fixed_prot_id (str, optional): Id of the protein to remain fixed in the experiment. Defaults to None.
        mut_prot_id (str, optional): Id of the protein to be mutated in the experiment. Defaults to None.
        num_of_itr (int, optional): Number of iterations in the experiment. Defaults to 300.
        model_type (str, optional): Type of the model to be used for PPI prediction; possible values: 'matpip', 'matds_hybrid'. Defaults to 'matds_hybrid'.
        protTrans_model_name (str, optional): PLM model name to be used to derive the PLM-based features; possible values: 'prot_t5_xl_half_uniref50-enc', 'prot_t5_xl_uniref50'. Defaults to 'prot_t5_xl_uniref50'.
        batch_size (int, optional): Number of candidate mutations to be evaluated through prediction generation in a batch. Defaults to 5.
        batch_dump_interval (int, optional): Number of batches after which result would be saved to disk during sumulation. Defaults to 10.
        cuda_index (int, optional): Cuda index for GPU. Possible values are 0, 1 for two GPU devices single node. Defaults to 0.
        psiblast_exec_path (str, optional): psiblast executable path required for PSSM-based feature extraction (manual 2d features). Defaults to './'.
        result_dump_dir_nm (str, optional): Full directory path where the simulation result will be stored. Defaults to './'.

    """
    print('inside run_prot_design_simuln() method - Start')
    print('####################################')
    print(f'root_path: {root_path} \n:: fixed_prot_id: {fixed_prot_id} :: mut_prot_id: {mut_prot_id} \n num_of_itr: {num_of_itr} \
          num_of_mut_pts: {num_of_mut_pts} \n model_type: {model_type} :: protTrans_model_name: {protTrans_model_name} \
          \n batch_size: {batch_size} :: batch_dump_interval: {batch_dump_interval} :: cuda_index: {cuda_index}')
    print('####################################')
    t_entire_init = time.time()
    # use del x1, x2   and  gc.collect() as and when required
    aa_lst = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    # Determine the number of available CPU cores
    num_cpu_cores = multiprocessing.cpu_count()
    # for tracking time of execution
    crit_lst, exec_time_lst = [], []

    print('\n################# One time task- Start\n')
    # The directory to save the downloaded PDB file.
    pdb_file_location = os.path.join(root_path, "dataset/preproc_data/pdb_files")
    PPIPUtils.createFolder(pdb_file_location)
    # fetch the protein sequence for the given protein id from PDB structure (if not previously done)
    fixed_prot_seq = prot_design_util.get_protein_sequence(fixed_prot_id, pdb_file_location=pdb_file_location)
    fixed_prot_seq = fixed_prot_seq.replace(" ", "").upper()
    mut_prot_seq = prot_design_util.get_protein_sequence(mut_prot_id, pdb_file_location=pdb_file_location)
    mut_prot_seq = mut_prot_seq.replace(" ", "").upper()
    mut_prot_len = len(mut_prot_seq)

    # ## RANDOM SEED IS ALREADY SET IN dl_reproducible_result_util import
    # seed_value = 456
    # # Set the seed for reproducibility
    # random.seed(seed_value)

    # initialize device
    device = torch.device(f'cuda:{cuda_index}')
    # load the protTrans PLM model -Start
    print('loading the protTrans PLM model -Start') 
    protTrans_model_path=os.path.join(root_path, '../ProtTrans_Models/')
    protTrans_model, tokenizer = preproc_plm_util.load_protTrans_model(protTrans_model_path=protTrans_model_path
                        , protTrans_model_name=protTrans_model_name, device=device)
    print('loading the protTrans PLM model -End') 
    # load the protTrans PLM model -End 

    # # ############### extract the features for the fixed_prot_seq -Start ###############
    # ### extract PLM-based features for fixed_prot_seq
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
    man_2d_feat_dict_prot_fixed = feat_engg_manual_main_pd.extract_prot_seq_2D_manual_feat(folderName=featureDir, fastas=fixed_prot_fastas
                                                             , skipgrm_lookup_prsnt = False, psiblast_exec_path=psiblast_exec_path
                                                             , labelEncode_lookup_prsnt = False, blosumMatrix=blosumMatrix)
    # extract manualfixed_prot_fastas 1d features for fixed_prot_seq
    man_1d_feat_dict_prot_fixed = feat_engg_manual_main_pd.extract_prot_seq_1D_manual_feat(fastas=fixed_prot_fastas)
    # ############### extract the features for the fixed_prot_seq -End ###############

    # load MaTPIP model
    matpip_model = None
    matpip_model = matpip_RunTests.load_matpip_model(root_path)

    # if model_type is 'matds_hybrid', also load D-SCRIPT model and pre-trained Bepler-Berger model
    dscript_model = None
    if(model_type == 'matds_hybrid'):
        dscript_model = matpip_RunTests.load_dscript_model(root_path)
        pretrained_bb_model = pretrained.get_pretrained("lm_v1")
        pretrained_bb_model = pretrained_bb_model.cuda()
        pretrained_bb_model.eval()
        alphabet = alphabets.Uniprot21()

    # declaration of various lists to be used later to create the batch specific output
    
    # initialize a list to store batch-specific output dictionaries and
    # it will be saved to disk as per batch_dump_interval
    batch_out_dict_lst = []
    print('################# One time task - End')

    # start the simulation
    print('Starting the simulation')
    t1 = time.time()
    mut_idx = 0
    # initialize prv_best_mutated_prot_seq
    prv_best_mutated_prot_seq = mut_prot_seq
    # initialize p_stable
    p_stable = -1.0
    # constant used in the MCMC algo
    t_mcmc = 0.01
    # derive the number of batches
    num_of_batches = int(num_of_itr / batch_size)
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
        for _ in range(mod_batch_size):
            # randomly select the mutation points in the mutable protein sequence
            # Use random.sample to get unique random positions
            sel_mut_seq_pos_lst = random.sample(range(mut_prot_len), num_of_mut_pts)  # e.g. sel_mut_seq_pos_lst: [19, 16, 0] for num_of_mut_pts=3
            
            # randomly select the amino acids from the aa_lst
            # Use random.sample to get unique random positions
            sel_aa_lst_indices = random.sample(range(20), num_of_mut_pts)  # e.g. sel_aa_lst_indices: [1, 0, 4] for num_of_mut_pts=3
        
            # For the ease of manipulation of mut_prot_seq which is a string, convert the string to a list of characters
            # Note: All the mutations will take place on prv_best_mutated_prot_seq even if mod_batch_size > 1
            prv_best_mutated_prot_seq_as_lst = list(prv_best_mutated_prot_seq)
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
        # end of for loop: for _ in range(mod_batch_size):
        t2 = time.time()
        print(f'batch_idx={batch_idx} :: time (in sec) for performing mutation= {round(t2-t1, 3)}')
        
        # ############### extract the features for the mut_prot_fastas -Start ###############
        mut_prot_id_lst, mut_prot_seq_lst = [], []
        for mut_prot_id, mutated_prot_seq in mut_prot_fastas:
            mut_prot_id_lst.append(mut_prot_id)
            mut_prot_seq_lst.append(mutated_prot_seq)
        ### extract PLM-based features for mut_prot_fastas -Start
        mut_plm_1d_embedd_tensor_lst = preproc_plm_util.extract_protTrans_plm_feat(prot_seq_lst=mut_prot_seq_lst, model=protTrans_model
                                                                                  , tokenizer=tokenizer, device=device)
        # add PLM-based features for the fixed protein alongwith the mutated proteins
        # using dict() and zip() to convert lists to dictionary
        plm_feat_dict = dict(zip([fixed_prot_id] + mut_prot_id_lst, fixed_plm_1d_embedd_tensor_lst + mut_plm_1d_embedd_tensor_lst))
        t3 = time.time()
        print(f'batch_idx={batch_idx} :: time (in sec) for PLM feature extraction = {round(t3-t2, 3)}')
        ### extract PLM-based features for mut_prot_fastas -End

        ### extract manual features for mut_prot_fastas -Start
        man_2d_feat_dict_prot = man_1d_feat_dict_prot = None
        if(batch_size == 1):
            # extract manual feature serially 
            man_2d_feat_dict_prot, man_1d_feat_dict_prot = extract_manual_feat_serially(featureDir, mut_prot_fastas, psiblast_exec_path, blosumMatrix, mut_prot_id_lst)
        else:
            num_process = 0
            if((batch_size > 1) and (batch_size < num_cpu_cores)):
                num_process = batch_size
            else:
                # batch_size >= num_cpu_cores
                num_process = num_cpu_cores - 2
            # end of if-else block

            # extract manual feature in multiprocessing fashion
            man_2d_feat_dict_prot, man_1d_feat_dict_prot = extract_manual_feat_mp(featureDir, mut_prot_fastas, psiblast_exec_path, blosumMatrix, mut_prot_id_lst, num_process)

        # add manual features for the fixed protein alongwith the mutated proteins
        man_2d_feat_dict_prot[fixed_prot_id] = man_2d_feat_dict_prot_fixed
        man_1d_feat_dict_prot[fixed_prot_id] = man_1d_feat_dict_prot_fixed
        t4 = time.time()
        print(f'batch_idx={batch_idx} :: time (in sec) for manual feature extraction = {round(t4-t3, 3)}')
        ### extract manual features for mut_prot_fastas -End
        # ############### extract the features for the mut_prot_fastas -End ###############

        # ############### execute MaTPIP prediction and get ppi_score -Start ###############
        matpip_class_1_prob_arr = matpip_RunTests.execute(root_path, matpip_model, featureDir, fixed_prot_id, mut_prot_id_lst
                                               , plm_feat_dict, man_2d_feat_dict_prot, man_1d_feat_dict_prot)
        t5 = time.time()
        print(f'batch_idx={batch_idx} :: MaTPIP :: time (in sec) for class prob generation = {round(t5-t4, 3)}')
        # ############### execute MaTPIP prediction and get ppi_score -End ###############
        # if model_type is 'matpip', horizontally stack 'matpip_class_1_prob_arr' to 'ppi_score_arr'
        if(model_type == 'matpip'):
            # horizontally stack 'matpip_class_1_prob_arr' to 'ppi_score_arr'
            ppi_score_arr = np.hstack((ppi_score_arr, matpip_class_1_prob_arr))
            t7 = time.time()
        else:
            # model_type is 'matds_hybrid', so generate the prediction by D-Script and then hybrid the result
            dscript_seqs = mut_prot_fastas + fixed_prot_fastas
            dscript_pairs = [(fixed_prot_id, mut_prot_id) for mut_prot_id in mut_prot_id_lst]
            threshold = 0.5
            dscript_class_1_prob_arr = predict.main_pd(pairs=dscript_pairs, model=dscript_model, pre_trn_mdl=pretrained_bb_model, alphabet=alphabet
                                                       , seqs=dscript_seqs, threshold=threshold)
            t6 = time.time()
            print(f'batch_idx={batch_idx} :: D-SCRIPT :: time (in sec) for class prob generation = {round(t6-t5, 3)}')

            # call matds_hybrid algorithm
            hybrid_class_1_prob_arr = calc_matds_hybrid_score(matpip_class_1_prob_arr, dscript_class_1_prob_arr)
            t7 = time.time()
            print(f'batch_idx={batch_idx} :: Hybrid :: time (in sec) for class prob generation = {round(t7-t6, 3)}')
            # horizontally stack 'hybrid_class_1_prob_arr' to 'ppi_score_arr'
            ppi_score_arr = np.hstack((ppi_score_arr, hybrid_class_1_prob_arr))
        # end of if-else block
        
        # ############### execute MCMC algo -Start ###############
        # p_stable corresponds to e_stable of MCMC algo. But the main difference is thet, in case
        # of energy, lesser is more stable but in case of class-1 probability score, larger is better.
        p_stable_changed = False
        # find the max value and the corresponding index in ppi_score_arr
        max_val_index = np.argmax(ppi_score_arr)
        max_value = ppi_score_arr[max_val_index]
        # treat max_value as p_i
        p_i = max_value
        # if p_i is more stable compared to p_stable, accept p_i and make it p_stable
        if(p_i > p_stable):
            # p_i is more stable than p_stable
            p_stable = p_i
            p_stable_changed = True
        else:
            # generate a random number between 0 and 1
            rand = np.random.rand()
            del_p = -(p_i - p_stable)  # del_p must be positive
            if( rand < math.exp(-(del_p / t_mcmc))):  # Metropolis criteria
                # accept p_i and make it as p_stable
                p_stable = p_i
                p_stable_changed = True
            else:
                # no change in p_stable
                p_stable_changed = False
            # end of inner if-else block
        # end of outer if-else block
        t8 = time.time()
        print(f'batch_idx={batch_idx} :: MCMC :: time (in sec) for execution = {round(t8-t7, 3)}')
        # ############### execute MCMC algo -End ###############
        
        # if p_stable is changed then only create batch specific output
        if(p_stable_changed):
            # create new batch output
            batch_out_dict = {'batch_idx': batch_idx}
            batch_out_dict['simuln_idx'] = (batch_size * batch_idx) + max_val_index
            batch_out_dict['mut_pos_lst'] = mut_pos_lst[max_val_index]
            batch_out_dict['aa_idx_lst'] = aa_idx_lst[max_val_index]
            batch_out_dict['ppi_score'] = p_i
            # ############# THE FOLLOWING LINE CAN BE COMMENTED OUT IF RESULT SAVING IS TAKING LONG TIME ############# #
            batch_out_dict['prot_seq'] = mut_prot_fastas[max_val_index][1]
            # append batch_out_dict into batch_out_dict_lst
            batch_out_dict_lst.append(batch_out_dict)
            # update prv_best_mutated_prot_seq
            prv_best_mutated_prot_seq = mut_prot_fastas[max_val_index][1]
        # end of if block: if(p_stable_changed):
        print(f'\n############# batch_idx={batch_idx} :: batch_size={batch_size} :: time (in sec) for specific batch execution = {round(t8-t1, 3)}\n')
        crit_lst.append(f'batch_idx_{batch_idx}'); exec_time_lst.append(round(t8-t1, 3))
        
        # check to save the batch_out_dict_lst to disk as per batch_dump_interval and calculate time for saving
        if((batch_idx + 1) % batch_dump_interval == 0):
            if(len(batch_out_dict_lst) > 0):  # check to see if batch_out_dict_lst is not empty, then only save it
                t9 = time.time()
                batch_out_df = pd.DataFrame(batch_out_dict_lst)
                batch_out_csv_nm = os.path.join(result_dump_dir_nm, f'batchIdx_{batch_idx + 1 - batch_dump_interval}_{batch_idx}_batchSize_{batch_size}.csv')
                batch_out_df.to_csv(batch_out_csv_nm, index=False)
                # reset batch_out_dict_lst
                batch_out_dict_lst = []
                t10 = time.time()
                print(f'batch_idx={batch_idx} :: Saving batch_out_dict_lst to disk :: time (in sec) for execution = {round(t10-t9, 3)}\n')
            # end of if block: if(len(batch_out_dict_lst) > 0)
        # end of if block
    # end of for loop: for batch_idx in range(num_of_batches):
    
    # check if any remaining result is stored in batch_out_dict_lst, if yes, then save it
    if(len(batch_out_dict_lst) > 0):
        # there is remaining result, stored in batch_out_dict_lst
        batch_out_df = pd.DataFrame(batch_out_dict_lst)
        batch_out_csv_nm = os.path.join(result_dump_dir_nm, f'batchIdx_{batch_dump_interval * (batch_idx // batch_dump_interval)}_{batch_idx}_batchSize_{batch_size}.csv')
        batch_out_df.to_csv(batch_out_csv_nm, index=False)
    # end of if block
    
    t_final = time.time()
    print(f'Total batch execution time :: time (in sec) = {round(t_final - t_batch_init, 3)}')
    crit_lst.append('batch_total_time'); exec_time_lst.append(round(t_final - t_batch_init, 3))
    print(f'\n *********\n :: Total time for complete simulation execution :: time (in sec) for execution = {round(t_final - t_entire_init, 3)} \n*********\n')
    crit_lst.append('simuln_total_time'); exec_time_lst.append(round(t_final - t_entire_init, 3))
    # save time records
    time_df = pd.DataFrame({'criterion': crit_lst, 'time_in_sec': exec_time_lst})
    time_df.to_csv(os.path.join(result_dump_dir_nm, 'time_records.csv'), index=False)
    print('inside run_prot_design_simuln() method - End')


def extract_manual_feat_serially(featureDir, mut_prot_fastas, psiblast_exec_path, blosumMatrix, mut_prot_id_lst):
    # ################################ serial version for the manual feature extraction -Start ################################
    # t3_1 = time.time()
    man_2d_feat_dict_prot = dict()
    man_1d_feat_dict_prot = dict()
    skipgrm_lookup_prsnt = labelEncode_lookup_prsnt = True
    for i in range(len(mut_prot_id_lst)):
        indiv_mut_prot_fastas = [mut_prot_fastas[i]]
        man_2d_feat_dict_prot_mut = feat_engg_manual_main_pd.extract_prot_seq_2D_manual_feat(folderName=featureDir, fastas=indiv_mut_prot_fastas
                                                            , skipgrm_lookup_prsnt=skipgrm_lookup_prsnt, psiblast_exec_path=psiblast_exec_path
                                                            , labelEncode_lookup_prsnt=labelEncode_lookup_prsnt, blosumMatrix=blosumMatrix)
        man_1d_feat_dict_prot_mut = feat_engg_manual_main_pd.extract_prot_seq_1D_manual_feat(fastas=indiv_mut_prot_fastas)

        man_2d_feat_dict_prot[indiv_mut_prot_fastas[0][0]] = man_2d_feat_dict_prot_mut
        man_1d_feat_dict_prot[indiv_mut_prot_fastas[0][0]] = man_1d_feat_dict_prot_mut
    # end for loop: for i in range(len(mut_prot_id_lst)):
    # t3_2 = time.time()
    # print(f'serial version for the manual feature extraction: execution time (in sec.) = {round(t3_2-t3_1, 3)}')
    # ################################ serial version for the manual feature extraction -End ################################
    return (man_2d_feat_dict_prot, man_1d_feat_dict_prot)


def extract_manual_feat_mp(featureDir, mut_prot_fastas, psiblast_exec_path, blosumMatrix, mut_prot_id_lst, num_process):
    # ################################ multprocessing version for the manual feature extraction -Start ################################
    # Shared data structures
    manager = multiprocessing.Manager()
    man_2d_feat_dict_prot = manager.dict()  # shared_data1
    man_1d_feat_dict_prot = manager.dict()  # shared_data2

    # Create a lock to protect shared data structures
    shared_data_lock = manager.Lock()

    # Fixed data structures
    # featureDir, mut_prot_fastas, skipgrm_lookup_prsnt=True, psiblast_exec_path, labelEncode_lookup_prsnt=True, blosumMatrix
    skipgrm_lookup_prsnt = labelEncode_lookup_prsnt = True
    
    # t3_1 = time.time()
    # Using Pool for multiprocessing with dynamically determined processes
    with multiprocessing.Pool(processes=num_process) as pool:
        # Use starmap to pass multiple arguments to the ext_man_feat function
        pool.starmap(ext_man_feat
                     , [(idx, featureDir, mut_prot_fastas, skipgrm_lookup_prsnt, psiblast_exec_path, labelEncode_lookup_prsnt, blosumMatrix  # Fixed data structures
                         , man_2d_feat_dict_prot, man_1d_feat_dict_prot, shared_data_lock  # Shared data structures
                         ) for idx in range(len(mut_prot_id_lst))])
    
    # t3_2 = time.time()
    # print(f'after multiprocessing.pool(): execution time (in sec.) = {round(t3_2-t3_1, 3)}')
    ################################ multprocessing version for the manual feature extraction -End ################################
    return (man_2d_feat_dict_prot, man_1d_feat_dict_prot)


def ext_man_feat(idx, featureDir, mut_prot_fastas, skipgrm_lookup_prsnt, psiblast_exec_path, labelEncode_lookup_prsnt, blosumMatrix  # Fixed data structures
                             , man_2d_feat_dict_prot, man_1d_feat_dict_prot, shared_data_lock):  # Shared data structures
    
    # t1 = time.time()
    indiv_mut_prot_fastas = [mut_prot_fastas[idx]]
    man_2d_feat_dict_prot_mut = feat_engg_manual_main_pd.extract_prot_seq_2D_manual_feat(folderName=featureDir, fastas=indiv_mut_prot_fastas
                                                             , skipgrm_lookup_prsnt=skipgrm_lookup_prsnt, psiblast_exec_path=psiblast_exec_path
                                                             , labelEncode_lookup_prsnt=labelEncode_lookup_prsnt, blosumMatrix=blosumMatrix)
    man_1d_feat_dict_prot_mut = feat_engg_manual_main_pd.extract_prot_seq_1D_manual_feat(fastas=indiv_mut_prot_fastas)

    # Acquire a lock to safely update the shared data structures
    with shared_data_lock:
        man_2d_feat_dict_prot[indiv_mut_prot_fastas[0][0]] = man_2d_feat_dict_prot_mut
        man_1d_feat_dict_prot[indiv_mut_prot_fastas[0][0]] = man_1d_feat_dict_prot_mut
    # t2 = time.time()
    # print(f'ext_man_feat: execution time (in sec.) = {round(t2-t1, 3)}')


def calc_matds_hybrid_score(matpip_class_1_prob_arr, dscript_class_1_prob_arr):
    # Calculate complementary arrays
    matpip_class_0_prob_arr = 1 - matpip_class_1_prob_arr
    dscript_class_0_prob_arr = 1 - dscript_class_1_prob_arr

    # hybrid strategy:
    # Whenever matpip is predicting positive and dscript is predicting negative for the same test sample, then the 
    # adjustment is done in such a way so that the prediction which is associated with the higher confidence level (in terms 
    # of the prediction probability) wins. The same is true for the reverse case and for all the other cases, follow matpip prediction.
    # Initialize hybrid_class_1_prob_arr
    hybrid_class_1_prob_arr = matpip_class_1_prob_arr

    # Apply conditions using nested np.where
    hybrid_class_1_prob_arr = np.where(
        (matpip_class_1_prob_arr > 0.5) & (dscript_class_0_prob_arr > 0.5),  # if
        matpip_class_1_prob_arr - (dscript_class_0_prob_arr - 0.5),  # then
        np.where(  # else
            (matpip_class_0_prob_arr > 0.5) & (dscript_class_1_prob_arr > 0.5),  # if
            # if matpip predicts negative but original dscript predicts positive, then again apply hybrid strategy but in reverse way
            dscript_class_1_prob_arr - (matpip_class_0_prob_arr - 0.5),  # then
            hybrid_class_1_prob_arr  # else
        )
    )

    # Return the final hybrid_class_1_prob_arr
    return hybrid_class_1_prob_arr


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/matpip_pd_prj')

    # length('2J8B') = 78; length('1IM5') = 179
    # protein to remain fixed in the experiment
    fixed_prot_id = '1IM5'
    # protein to be mutated in the experiment
    mut_prot_id = '2J8B'
    # number of iterations in the experiment
    num_of_itr_lst = [30000]
    # number of mutation points to be used in the mutable protein in the experiment
    num_of_mut_pts_lst = [2]
    # model type to be used for PPI prediction; possible values: matpip, matds_hybrid
    model_type = 'matds_hybrid'  # 'matpip', 'matds_hybrid'30
    # PLM model name to be used to derive the PLM-based features 
    protTrans_model_name = 'prot_t5_xl_half_uniref50-enc'  # prot_t5_xl_half_uniref50-enc, prot_t5_xl_uniref50
    # number of candidate mutations to be evaluated through prediction generation in a batch
    batch_size = 1
    # Number of batches after which result would be saved to disk during sumulation
    batch_dump_interval = 2000
    # cuda index for GPU
    cuda_index = 0  # Possible values are 0, 1 for two GPU devices single node
    # psiblast executable path required for PSSM-based feature extraction (manual 2d featurt6 = time.time()es)
    psiblast_exec_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/ncbi-blast-2.13.0+/bin/')

    for num_of_itr in num_of_itr_lst:
        for num_of_mut_pts in num_of_mut_pts_lst:
            # check for num_of_itr to be a multiple of batch_size
            if(num_of_itr % batch_size != 0):
                err_msg = f'Number of iterations ({num_of_itr}) must be a multiple of batch size ({batch_size}))'
                sys.exit(err_msg)
            # create result dump directory
            result_dump_dir_nm = os.path.join(root_path, f'dataset/proc_data/result_dump/fixed_{fixed_prot_id}_mut_{mut_prot_id}', f'res_totItr{num_of_itr}_batchSz{batch_size}_mutPts{num_of_mut_pts}')
            PPIPUtils.createFolder(result_dump_dir_nm)

            # trigger the actual experiment
            run_prot_design_simuln(root_path=root_path, fixed_prot_id=fixed_prot_id, mut_prot_id=mut_prot_id
                                   , num_of_itr=num_of_itr, num_of_mut_pts=num_of_mut_pts, model_type=model_type
                                   , protTrans_model_name=protTrans_model_name, batch_size=batch_size
                                   , batch_dump_interval=batch_dump_interval, cuda_index=cuda_index, psiblast_exec_path=psiblast_exec_path
                                   , result_dump_dir_nm=result_dump_dir_nm)
            # call for the gc
            gc.collect()
        # end of for loop: for num_of_mut_pts in num_of_mut_pts_lst:
    # end of for loop: for num_of_itr in num_of_itr_lst:





