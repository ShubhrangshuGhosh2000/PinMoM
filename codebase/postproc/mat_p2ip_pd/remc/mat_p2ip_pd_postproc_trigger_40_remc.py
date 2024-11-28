import os, sys

from pathlib import Path
path_root = Path(__file__).parents[3]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

import gc

from utils import dl_reproducible_result_util
from postproc.mat_p2ip_pd.mat_p2ip_pd_postproc_1_justA4simuln import run_postproc_1_just_after_simuln
from postproc.mat_p2ip_pd.mat_p2ip_pd_postproc_2_mutSeqStructPred_byAf2 import run_postproc_2_mut_seq_struct_pred_by_af2
from postproc.mat_p2ip_pd.mat_p2ip_pd_postproc_3_mutSeqStruct_comparison import run_postproc_3_mut_seq_struct_comparison
# from postproc.mat_p2ip_pd.mat_p2ip_pd_postproc_4_mutSeqStruct_overlayForComplxFormn import run_postproc_4_mut_seq_struct_overlay_for_complx_formn
from postproc.mat_p2ip_pd.mat_p2ip_pd_postproc_5_mutComplexStructPred_byAf2 import run_postproc_5_mut_complx_struct_pred_by_af2
from postproc.mat_p2ip_pd.mat_p2ip_pd_postproc_5_part2_mutComplxStruct_comparison import run_postproc_5_part2_mut_complx_struct_comparison
from postproc.mat_p2ip_pd.mat_p2ip_pd_postproc_6_complexStructMdSimulation import run_postproc_6_complx_struct_mdSim

from utils import PPIPUtils

def trigger_pd_postproc(root_path = './', postproc_stage=None, user_input_dict=dict()):
    print('inside trigger_pd_postproc() method - Start')
    print('\n#########################################################')
    print(f'         postproc_stage = {postproc_stage}')
    print('#########################################################')
    print(f'\n ===================\n user_input_dict: \n {user_input_dict}')
    # Create input_dict from user_input_dict.
    # User may chose to provide only a few arguments, required for the experiment.
    # input_dict would contain those user overridden argument values and the rest of the arguments with
    # their default values. In other words, input_dict would contain full set of arguments required for
    # the postprocessing whereas user_input_dict would contain only a subset (which user chose to override) of it.
    input_dict = {'root_path': root_path}

    # location of the downloaded PDB files
    def_pdb_file_location = os.path.join(root_path, "dataset/preproc_data/pdb_files")  # default value
    # check if the user_input_dict contains value for the key 'pdb_file_location'; 
    # otherwise, set it to the default value
    input_dict['pdb_file_location'] = user_input_dict.get('pdb_file_location', def_pdb_file_location)

    # list of dimeric protein complex names to be used as input for the postprocessing after simulation 
    def_dim_prot_complx_nm_lst = ['2I25']  # default value
    # check if the user_input_dict contains value for the key 'dim_prot_complx_nm_lst'; 
    # otherwise, set it to the default value
    input_dict['dim_prot_complx_nm_lst'] = user_input_dict.get('dim_prot_complx_nm_lst', def_dim_prot_complx_nm_lst)

    # Location of the folder in which the simulation (processing) result is saved 
    def_sim_result_dir = os.path.join(root_path, "dataset/proc_data/result_dump")  # default value
    # check if the user_input_dict contains value for the key 'sim_result_dir'; 
    # otherwise, set it to the default value
    input_dict['sim_result_dir'] = user_input_dict.get('sim_result_dir', def_sim_result_dir)
    
    # Location of the folder in which the postprocessing result will be saved 
    def_postproc_result_dir = os.path.join(root_path, f'dataset/postproc_data/result_dump')  # default value
    # check if the user_input_dict contains value for the key 'postproc_result_dir'; 
    # otherwise, set it to the default value
    input_dict['postproc_result_dir'] = user_input_dict.get('postproc_result_dir', def_postproc_result_dir)

    # Execution mode in which the simulation process runs. It can be either 'mcmc' or 'remc'.
    def_simulation_exec_mode = 'mcmc'  # default value
    # check if the user_input_dict contains value for the key 'simulation_exec_mode'; 
    # otherwise, set it to the default value
    input_dict['simulation_exec_mode'] = user_input_dict.get('simulation_exec_mode', def_simulation_exec_mode)
    
    # Cuda index for GPU. Possible values are 0, 1 for two GPU devices in a single node.
    def_cuda_index = 0  # 0 default value
    # check if the user_input_dict contains value for the key 'cuda_index'; 
    # otherwise, set it to the default value
    input_dict['cuda_index'] = user_input_dict.get('cuda_index', def_cuda_index)

    # Threshold value for the PPI score (actually probproc_dataability of PPI). For postprocessing, mutated sequences above this threshold will be considered.
    def_ppi_score_threshold = 0.8  # default value
    # check if the user_input_dict contains value for the key 'ppi_score_threshold'; 
    # otherwise, set it to the default value
    input_dict['ppi_score_threshold'] = user_input_dict.get('ppi_score_threshold', def_ppi_score_threshold)

    # ######################################## Postprocessing PSI-Blast related parameters -Start ########################################
    # A boolean flag to indicate whether PSI-Blast based sequence similarity check will be carried out as a part of the postprocessing.
    def_postproc_psiblast_enabled = True  # default value
    # check if the user_input_dict contains value for the key 'postproc_psiblast_enabled'; 
    # otherwise, set it to the default value
    input_dict['postproc_psiblast_enabled'] = user_input_dict.get('postproc_psiblast_enabled', def_postproc_psiblast_enabled)

    # PSI-Blast executable path to be used for the PSI-Blast execution as a part of the postprocessing.
    def_postproc_psiblast_exec_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PSI_BLAST_v2/blast_executable/ncbi-blast-2.15.0+/bin/')  # default value
    # check if the user_input_dict contains value for the key 'postproc_psiblast_exec_path'; 
    # otherwise, set it to the default value
    input_dict['postproc_psiblast_exec_path'] = user_input_dict.get('postproc_psiblast_exec_path', def_postproc_psiblast_exec_path)
    
    # Uniprot DB path to be used for the PSI-Blast execution as a part of the postprocessing.
    def_postproc_psiblast_uniprot_db_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PSI_BLAST_v2/uniprot_sprot_db/')  # default value
    # check if the user_input_dict contains value for the key 'postproc_psiblast_uniprot_db_path'; 
    # otherwise, set it to the default value
    input_dict['postproc_psiblast_uniprot_db_path'] = user_input_dict.get('postproc_psiblast_uniprot_db_path', def_postproc_psiblast_uniprot_db_path)
    
    # Location of the folder in which the result for the postprocessing PSI-Blast execution will be saved 
    def_postproc_psiblast_result_dir = os.path.join(root_path, f'dataset/postproc_data/psi_blast_result/')  # default value
    # check if the user_input_dict contains value for the key 'postproc_psiblast_result_dir'; 
    # otherwise, set it to the default value
    input_dict['postproc_psiblast_result_dir'] = user_input_dict.get('postproc_psiblast_result_dir', def_postproc_psiblast_result_dir)

    # Upper threshold of the percent similarity score as obtained from the PSI-Blast. For postprocessing, mutated chaon sequence below this threshold will be considered.
    def_psiBlast_percent_similarity_score_threshold = 70.0  # default value
    # check if the user_input_dict contains value for the key 'psiBlast_percent_similarity_score_threshold'; 
    # otherwise, set it to the default value
    input_dict['psiBlast_percent_similarity_score_threshold'] = user_input_dict.get('psiBlast_percent_similarity_score_threshold', def_psiBlast_percent_similarity_score_threshold)
    # ######################################## Postprocessing PSI-Blast related parameters -End ########################################

    # Maximum mumber of mutated sequences to be selected out of those having psiBlast percent similarity_score lower than the 'psiBlast_percent_similarity_score_threshold'.
    def_max_num_of_seq_below_psiBlast_sim_score_threshold = 100  # default value
    # check if the user_input_dict contains value for the key 'max_num_of_seq_below_psiBlast_sim_score_threshold'; 
    # otherwise, set it to the default value
    input_dict['max_num_of_seq_below_psiBlast_sim_score_threshold'] = user_input_dict.get('max_num_of_seq_below_psiBlast_sim_score_threshold', def_max_num_of_seq_below_psiBlast_sim_score_threshold)
    # ######################################## Postprocessing PSI-Blast related parameters -End ########################################
    
    # Maximum mumber of mutated sequences per chain combo to be used for the AF2 based chain structure prediction. 
    # These sequences are those having psiBlast percent similarity_score lower than the 'psiBlast_percent_similarity_score_threshold' and may be selected through clustering.
    def_max_num_of_seq_for_af2_chain_struct_pred = 15  # 2
    # check if the user_input_dict contains value for the key 'max_num_of_seq_for_af2_chain_struct_pred'; 
    # otherwise, set it to the default value
    input_dict['max_num_of_seq_for_af2_chain_struct_pred'] = user_input_dict.get('max_num_of_seq_for_af2_chain_struct_pred', def_max_num_of_seq_for_af2_chain_struct_pred)
    
    # Threshold value (in Angstrom) of the Backbone RMSD for original and mutated chain structure comparison. For postprocessing, mutated sequences below this threshold will be considered.
    def_bb_rmsd_threshold_chain_struct_comp = 10.0  # default value
    # check if the user_input_dict contains value for the key 'bb_rmsd_threshold_chain_struct_comp'; 
    # otherwise, set it to the default value
    input_dict['bb_rmsd_threshold_chain_struct_comp'] = user_input_dict.get('bb_rmsd_threshold_chain_struct_comp', def_bb_rmsd_threshold_chain_struct_comp)

    # Maximum mumber of mutated sequences to be selected out of those having Backbone RMSD value lower than the 'bb_rmsd_threshold_chain_struct_comp'.
    # The selection will be done in the ascending order of the associated Backbone RMSD value, i.e. the candidate with the lowest Backbone RMSD value would be selected first and so on.
    def_max_num_of_seq_below_rmsd_thrshld_chain_struct = 15  # 2
    # check if the user_input_dict contains value for the key 'max_num_of_seq_below_rmsd_thrshld_chain_struct'; 
    # otherwise, set it to the default value
    input_dict['max_num_of_seq_below_rmsd_thrshld_chain_struct'] = user_input_dict.get('max_num_of_seq_below_rmsd_thrshld_chain_struct', def_max_num_of_seq_below_rmsd_thrshld_chain_struct)

    # Threshold value (in Angstrom) of the Backbone RMSD for original and mutated complex structure comparison. For postprocessing, mutated complexes below this threshold will be considered.
    def_bb_rmsd_threshold_complx_struct_comp = 10.0  # default value
    # check if the user_input_dict contains value for the key 'bb_rmsd_threshold_complx_struct_comp'; 
    # otherwise, set it to the default value
    input_dict['bb_rmsd_threshold_complx_struct_comp'] = user_input_dict.get('bb_rmsd_threshold_complx_struct_comp', def_bb_rmsd_threshold_complx_struct_comp)

    # ######################################## AlphaFold2 related parameters -Start ########################################
    # Whether to use SCWRL in place of AF2 for the mutated sequence structure prediction (postproc_stage = '2_mut_seq_struct_pred_by_af2') for the faster prediction at the cost of accuracy
    def_af2_use_scwrl = False
    # check if the user_input_dict contains value for the key 'af2_use_scwrl'; 
    # otherwise, set it to the default value
    input_dict['af2_use_scwrl'] = user_input_dict.get('af2_use_scwrl', def_af2_use_scwrl)

    # Scwrl executable path to be used for the Scwrl execution
    def_scwrl_exec_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/SCWRL4/')  # default value
    # check if the user_input_dict contains value for the key 'scwrl_exec_path'; 
    # otherwise, set it to the default value
    input_dict['scwrl_exec_path'] = user_input_dict.get('scwrl_exec_path', def_scwrl_exec_path)
    
    # Execution mode of Alphafold. THere are 3 modes: 
    # 1. msa_gen: In this mode, AF2 execution will produce only MSA files and not any prediction result.
    # 2. pred_a4_msa_gen: In this mode, AF2 will use pre-generated MSA files for the prediction.
    # 3. msa_pred_gen: In this mode, AF2 will first genearate MSA files and after that will do the prediction using those MSA files as an end-to-end process. 
    def_af2_exec_mode = 'msa_pred_gen'
    # check if the user_input_dict contains value for the key 'af2_exec_mode'; 
    # otherwise, set it to the default value
    input_dict['af2_exec_mode'] = user_input_dict.get('af2_exec_mode', def_af2_exec_mode) 
    
    # Number of prediction recycles. Increasing recycles can improve the prediction quality but slows down the prediction.
    def_af2_num_recycle = 10
    # check if the user_input_dict contains value for the key 'af2_num_recycle'; 
    # otherwise, set it to the default value
    input_dict['af2_num_recycle'] = user_input_dict.get('af2_num_recycle', def_af2_num_recycle)

    # Number of ensembles. The trunk of the network is run multiple times with different random choices for the MSA cluster centers. 
    # This can result in a better prediction at the cost of longer runtime. 
    def_af2_num_ensemble = 1
    # check if the user_input_dict contains value for the key 'af2_num_ensemble'; 
    # otherwise, set it to the default value
    input_dict['af2_num_ensemble'] = user_input_dict.get('af2_num_ensemble', def_af2_num_ensemble)

    # Number of seeds to try. Will iterate from range(random_seed, random_seed+num_seeds). This can result in a better/different
    # prediction at the cost of longer runtime. 
    def_af2_num_seeds = 3
    # check if the user_input_dict contains value for the key 'af2_num_seeds'; 
    # otherwise, set it to the default value
    input_dict['af2_num_seeds'] = user_input_dict.get('af2_num_seeds', def_af2_num_seeds)

    # Enable OpenMM/Amber for structure relaxation. Can improve the quality of side-chains at a cost of longer runtime. 
    def_af2_use_amber = False
    # check if the user_input_dict contains value for the key 'af2_use_amber'; 
    # otherwise, set it to the default value
    input_dict['af2_use_amber'] = user_input_dict.get('af2_use_amber', def_af2_use_amber)

    # Do not recompute results, if a query has already been predicted. 
    def_af2_overwrite_existing_results = False
    # check if the user_input_dict contains value for the key 'af2_overwrite_existing_results'; 
    # otherwise, set it to the default value
    input_dict['af2_overwrite_existing_results'] = user_input_dict.get('af2_overwrite_existing_results', def_af2_overwrite_existing_results)
    
    # Location of the I/O folder of AF2
    def_af2_io_dir = os.path.join(root_path, 'dataset/postproc_data/alphafold2_io')  # default value
    # check if the user_input_dict contains value for the key 'af2_io_dir'; 
    # otherwise, set it to the default value
    input_dict['af2_io_dir'] = user_input_dict.get('af2_io_dir', def_af2_io_dir)

    # Create the input directory for the mutated sequence structure prediction by AlphaFold2
    inp_dir_mut_seq_struct_pred_af2 = os.path.join(input_dict['af2_io_dir'], 'mut_seq_struct_pred_inp')
    PPIPUtils.createFolder(inp_dir_mut_seq_struct_pred_af2, recreate_if_exists=False)

    # Create the output directory for the mutated sequence structure prediction by AlphaFold2
    out_dir_mut_seq_struct_pred_af2 = os.path.join(input_dict['af2_io_dir'], 'mut_seq_struct_pred_out')
    PPIPUtils.createFolder(out_dir_mut_seq_struct_pred_af2, recreate_if_exists=False)

    # Create the input directory for the mutated complex structure prediction by AlphaFold2
    inp_dir_mut_complx_struct_pred_af2 = os.path.join(input_dict['af2_io_dir'], 'mut_complx_struct_pred_inp')
    PPIPUtils.createFolder(inp_dir_mut_complx_struct_pred_af2, recreate_if_exists=False)
    
    # Create the output directory for the mutated complex structure prediction by AlphaFold2
    out_dir_mut_complx_struct_pred_af2 = os.path.join(input_dict['af2_io_dir'], 'mut_complx_struct_pred_out')
    PPIPUtils.createFolder(out_dir_mut_complx_struct_pred_af2, recreate_if_exists=False)
    # ######################################## AlphaFold2 related parameters -End ########################################

    # ######################################## Molecular Dynamics (MD) simulation related parameters -Start ########################################
    # Cuda index related to MD simulation. Possible values are "0", "1", "0,1" for two GPU devices in a single node.
    def_cuda_index_mdSim = "0"  # 0 default value
    # check if the user_input_dict contains value for the key 'cuda_index_mdSim'; 
    # otherwise, set it to the default value
    input_dict['cuda_index_mdSim'] = user_input_dict.get('cuda_index_mdSim', def_cuda_index_mdSim)

    # Do not recompute results, if MD simulation is already done for a protein complex. 
    def_mdSim_overwrite_existing_results = False
    # check if the user_input_dict contains value for the key 'mdSim_overwrite_existing_results'; 
    # otherwise, set it to the default value
    input_dict['mdSim_overwrite_existing_results'] = user_input_dict.get('mdSim_overwrite_existing_results', def_mdSim_overwrite_existing_results)
    
    # Forcefield (ff) to be used in MD simulation. Possible values can be found in https://tutorials.gromacs.org/docs/md-intro-tutorial.html.
    def_forcefield_mdSim = "amber99sb-ildn"  # default value
    # check if the user_input_dict contains value for the key 'forcefield_mdSim'; 
    # otherwise, set it to the default value
    input_dict['forcefield_mdSim'] = user_input_dict.get('forcefield_mdSim', def_forcefield_mdSim)
    
    # Maximum mumber of MD Simulation eligibile mutated complex candidates to be considered per chain combo for each given protein complex.
    # The selection will be done in the ascending order of the associated Backbone RMSD value, i.e. the candidate with the lowest Backbone RMSD value would be selected first and so on.
    def_max_cadidate_count_mdSim = 1  # 1 default value
    # check if the user_input_dict contains value for the key 'max_cadidate_count_mdSim'; 
    # otherwise, set it to the default value
    input_dict['max_cadidate_count_mdSim'] = user_input_dict.get('max_cadidate_count_mdSim', def_max_cadidate_count_mdSim)
    
    # Location of the folder in which the MD simulation result is saved 
    def_mdSim_result_dir = os.path.join(root_path, 'dataset/postproc_data/mdSim_result')  # default value
    # check if the user_input_dict contains value for the key 'mdSim_result_dir'; 
    # otherwise, set it to the default value
    input_dict['mdSim_result_dir'] = user_input_dict.get('mdSim_result_dir', def_mdSim_result_dir)
    # ######################################## Molecular Dynamics (MD) simulation related parameters -End ########################################

    for dim_prot_complx_nm in input_dict['dim_prot_complx_nm_lst']:
        try:
            # trigger the actual postprocessing as per the postproc_stage
            if(postproc_stage == '1_just_after_simuln'):
                run_postproc_1_just_after_simuln(root_path = input_dict['root_path'], dim_prot_complx_nm = dim_prot_complx_nm
                                        , sim_result_dir = input_dict['sim_result_dir'], postproc_result_dir = input_dict['postproc_result_dir']
                                        , simulation_exec_mode = input_dict['simulation_exec_mode'], cuda_index = input_dict['cuda_index']
                                        , pdb_file_location = input_dict['pdb_file_location'], ppi_score_threshold = input_dict['ppi_score_threshold']
                                        , postproc_psiblast_enabled = input_dict['postproc_psiblast_enabled']
                                        , postproc_psiblast_exec_path = input_dict['postproc_psiblast_exec_path'], postproc_psiblast_uniprot_db_path = input_dict['postproc_psiblast_uniprot_db_path']
                                        , postproc_psiblast_result_dir = input_dict['postproc_psiblast_result_dir']
                                        , psiBlast_percent_similarity_score_threshold = input_dict['psiBlast_percent_similarity_score_threshold']
                                        , max_num_of_seq_below_psiBlast_sim_score_threshold = input_dict['max_num_of_seq_below_psiBlast_sim_score_threshold']
                                        , max_num_of_seq_for_af2_chain_struct_pred = input_dict['max_num_of_seq_for_af2_chain_struct_pred']
                                        , inp_dir_mut_seq_struct_pred_af2 = inp_dir_mut_seq_struct_pred_af2)
            elif(postproc_stage == '2_mut_seq_struct_pred_by_af2'):
                run_postproc_2_mut_seq_struct_pred_by_af2(root_path = input_dict['root_path'], dim_prot_complx_nm = dim_prot_complx_nm
                                        , cuda_index = input_dict['cuda_index'], af2_use_scwrl = input_dict['af2_use_scwrl'], scwrl_exec_path = input_dict['scwrl_exec_path']
                                        , af2_exec_mode = input_dict['af2_exec_mode']
                                        , af2_num_recycle = input_dict['af2_num_recycle'], af2_num_ensemble = input_dict['af2_num_ensemble']
                                        , af2_num_seeds = input_dict['af2_num_seeds'], af2_use_amber = input_dict['af2_use_amber'], af2_overwrite_existing_results = input_dict['af2_overwrite_existing_results']
                                        , inp_dir_mut_seq_struct_pred_af2 = inp_dir_mut_seq_struct_pred_af2, out_dir_mut_seq_struct_pred_af2 = out_dir_mut_seq_struct_pred_af2)
            elif(postproc_stage == '3_mut_seq_struct_compare'):
                run_postproc_3_mut_seq_struct_comparison(root_path = input_dict['root_path'], dim_prot_complx_nm = dim_prot_complx_nm
                                        , postproc_result_dir = input_dict['postproc_result_dir'] , pdb_file_location = input_dict['pdb_file_location'], af2_use_amber = input_dict['af2_use_amber']
                                        , bb_rmsd_threshold_chain_struct_comp = input_dict['bb_rmsd_threshold_chain_struct_comp']
                                        , inp_dir_mut_seq_struct_pred_af2 = inp_dir_mut_seq_struct_pred_af2, out_dir_mut_seq_struct_pred_af2 = out_dir_mut_seq_struct_pred_af2)
            # elif(postproc_stage == '4_mut_seq_struct_overlay_for_complx_formn'):
            #     run_postproc_4_mut_seq_struct_overlay_for_complx_formn(root_path = input_dict['root_path'], dim_prot_complx_nm = dim_prot_complx_nm
            #                             , postproc_result_dir = input_dict['postproc_result_dir'] , pdb_file_location = input_dict['pdb_file_location'], af2_use_amber = input_dict['af2_use_amber']
            #                             , inp_dir_mut_seq_struct_pred_af2 = inp_dir_mut_seq_struct_pred_af2, out_dir_mut_seq_struct_pred_af2 = out_dir_mut_seq_struct_pred_af2)
            elif(postproc_stage == '5_mut_complx_struct_pred_by_af2'):
                run_postproc_5_mut_complx_struct_pred_by_af2(root_path = input_dict['root_path'], dim_prot_complx_nm = dim_prot_complx_nm
                                        , cuda_index = input_dict['cuda_index']
                                        , postproc_result_dir = input_dict['postproc_result_dir'] , pdb_file_location = input_dict['pdb_file_location']
                                        , max_num_of_seq_below_rmsd_thrshld_chain_struct = input_dict['max_num_of_seq_below_rmsd_thrshld_chain_struct']
                                        , af2_exec_mode = input_dict['af2_exec_mode']
                                        , af2_num_recycle = input_dict['af2_num_recycle'], af2_num_ensemble = input_dict['af2_num_ensemble']
                                        , af2_num_seeds = input_dict['af2_num_seeds'], af2_use_amber = input_dict['af2_use_amber'], af2_overwrite_existing_results = input_dict['af2_overwrite_existing_results']
                                        , inp_dir_mut_complx_struct_pred_af2 = inp_dir_mut_complx_struct_pred_af2
                                        , out_dir_mut_complx_struct_pred_af2 = out_dir_mut_complx_struct_pred_af2)
            elif(postproc_stage == '5_part2_mut_complx_struct_compare'):
                run_postproc_5_part2_mut_complx_struct_comparison(root_path = input_dict['root_path'], dim_prot_complx_nm = dim_prot_complx_nm
                                        , postproc_result_dir = input_dict['postproc_result_dir'] , pdb_file_location = input_dict['pdb_file_location']
                                        , af2_use_amber = input_dict['af2_use_amber'], bb_rmsd_threshold_complx_struct_comp = input_dict['bb_rmsd_threshold_complx_struct_comp']
                                        , inp_dir_mut_complx_struct_pred_af2 = inp_dir_mut_complx_struct_pred_af2
                                        , out_dir_mut_complx_struct_pred_af2 = out_dir_mut_complx_struct_pred_af2)
            elif(postproc_stage == '6_complx_struct_simln_using_MD'):
                # ######### IMPORTANT: The following unix commands must be executed in the runtime environment (bash) before invoking MD simulation related python script - Start #########
                # ### module load apps/gromacs/2022/gpu  # For Paramvidya: module load apps/gromacs/16.6.2022/intel
                # ### source /home/apps/gromacs/gromacs-2022.2/installGPUIMPI/bin/GMXRC  # For Paramvidya: source /home/apps/gromacs/gromacs-2022.2/installGPUIOMPI/bin/GMXRC
                # ### export GMX_ENABLE_DIRECT_GPU_COMM=1
                # ### export PATH=/home/pralaycs/miniconda3/envs/py3114_torch_gpu_param/bin:$PATH
                # ######### IMPORTANT: The following unix commands must be executed in the runtime environment (bash) before invoking MD simulation related python script - End #########
                run_postproc_6_complx_struct_mdSim(root_path = input_dict['root_path'], dim_prot_complx_nm = dim_prot_complx_nm
                                        , cuda_index_mdSim = input_dict['cuda_index_mdSim'], postproc_result_dir = input_dict['postproc_result_dir']
                                        , pdb_file_location = input_dict['pdb_file_location']
                                        , af2_use_amber = input_dict['af2_use_amber']
                                        , out_dir_mut_complx_struct_pred_af2 = out_dir_mut_complx_struct_pred_af2
                                        , mdSim_overwrite_existing_results = input_dict['mdSim_overwrite_existing_results'], forcefield_mdSim = input_dict['forcefield_mdSim']
                                        , max_cadidate_count_mdSim = input_dict['max_cadidate_count_mdSim']
                                        , mdSim_result_dir = input_dict['mdSim_result_dir'])
        except Exception as ex:
            print(f'************ ############## Error in processing :: {dim_prot_complx_nm} \n Error is: {ex}\n')
    # end of for loop: for dim_prot_complx_nm in input_dict['dim_prot_complx_nm_lst']:
    print('inside trigger_pd_postproc() method - End')


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/matpip_pd_prj')
    # root_path = os.path.join('/home/suvra/SG_working/workspaces/matpip_pd_prj')
    

    # Different post-processing stages are:
    # 1_just_after_simuln, 2_mut_seq_struct_pred_by_af2, 3_mut_seq_struct_compare
    # 4_mut_seq_struct_overlay_for_complx_formn, 5_mut_complx_struct_pred_by_af2, 5_part2_mut_complx_struct_compare
    # 6_mut_complx_struct_simln_using_MD
    # postproc_stage_lst = ['1_just_after_simuln', '2_mut_seq_struct_pred_by_af2', '3_mut_seq_struct_compare', '4_mut_seq_struct_overlay_for_complx_formn'
    #                       , '5_mut_complx_struct_pred_by_af2', '5_part2_mut_complx_struct_compare', '6_complx_struct_simln_using_MD']

    postproc_stage_lst = ['5_mut_complx_struct_pred_by_af2']

    user_input_dict = {}

    # ###################################################### #
    iteration_tag = 'remc_fullLen_puFalse_mutPrcntLen10'
    # ###################################################### #
    # ###################################################### #
    dim_prot_complx_nm_lst_40_orig = ['1KAC', '1KTZ', '1KXP', '1PVH', '1QA9', '1S1Q', '1SBB', '1T6B', '1XD3', '1Z0K']
    dim_prot_complx_nm_lst_done =   []
    dim_prot_complx_nm_lst_notConsidered = ['5JMO', '1PPE', '1FQJ', '1OYV', '2BTF', '1H9D', '1BUH', '1QA9', '1S1Q', '1XD3', '1PVH']
    dim_prot_complx_nm_lst_chainBreak = ['1AVX', '1AY7', '1BUH', '1D6R', '1EAW', '1EFN', '1F34', '1FLE', '1GL1', '1GLA', '1GPW', '1GXD', '1H9D', '1JTG', '1KTZ', '1KXP', '1MAH', '1OC0', '1OPH', '1OYV' \
                                        ,'1PPE', '1R0R', '1S1Q', '1SBB', '1T6B', '1US7', '1XD3', '1YVB', '1Z5Y', '1ZHH', '1ZHI', '2AST', '2ABZ', '2AJF', '2AYO', '2B42', '2BTF', '2FJU', '2G77', '2HLE' \
                                        ,'2HQS', '2O8V', '2PCC', '2UUY', '2VDB', '3SGQ', '4Y7M', '5JMO', '5SV3', '6CWG', '6DBG']
    dim_prot_complx_nm_lst_excluded = dim_prot_complx_nm_lst_done + dim_prot_complx_nm_lst_notConsidered
    dim_prot_complx_nm_lst_effective = [prot_id for prot_id in dim_prot_complx_nm_lst_40_orig if prot_id not in dim_prot_complx_nm_lst_excluded]
    user_input_dict['dim_prot_complx_nm_lst'] =  dim_prot_complx_nm_lst_effective  #
    # ###################################################### #
    # ###################################################### #

    # ###################################################### #
    user_input_dict['sim_result_dir'] = os.path.join(root_path, f'dataset/proc_data/result_dump_{iteration_tag}')  #
    user_input_dict['simulation_exec_mode'] = 'remc'  #
    # ###################################################### #

    # ###################################################### #
    user_input_dict['postproc_result_dir'] = os.path.join(root_path, f'dataset/postproc_data/result_dump/{iteration_tag}')  #
    # ###################################################### #

    # ###################################################### #
    user_input_dict['postproc_psiblast_enabled'] = False  #
    user_input_dict['postproc_psiblast_result_dir'] = os.path.join(root_path, f'dataset/postproc_data/psi_blast_result/{iteration_tag}')  #
    user_input_dict['psiBlast_percent_similarity_score_threshold'] = 75  #
    user_input_dict['max_num_of_seq_below_psiBlast_sim_score_threshold'] = 100  #
    # ###################################################### #

    # ###################################################### #
    user_input_dict['af2_io_dir'] = os.path.join(root_path, f'dataset/postproc_data/alphafold2_io/{iteration_tag}')  #
    # ###################################################### #

    # ###################################################### #
    user_input_dict['mdSim_result_dir'] = os.path.join(root_path, f'dataset/postproc_data/mdSim_result/{iteration_tag}')
    # ###################################################### #

    user_input_dict['cuda_index'] = 0  #

    user_input_dict['ppi_score_threshold'] = 0.50  #
    user_input_dict['max_num_of_seq_for_af2_chain_struct_pred'] = 50  #

    # ###################################################### #
    user_input_dict['af2_use_scwrl'] = False
    user_input_dict['af2_exec_mode'] = 'msa_gen'  # msa_gen, pred_a4_msa_gen, msa_pred_gen
    
    user_input_dict['af2_num_recycle'] = 5
    user_input_dict['af2_num_seeds'] = 1
    user_input_dict['af2_use_amber'] = True

    # ###################################################### #
    user_input_dict['bb_rmsd_threshold_chain_struct_comp'] = 15  # 15, 1000
    user_input_dict['max_num_of_seq_below_rmsd_thrshld_chain_struct'] = 50
    
    user_input_dict['bb_rmsd_threshold_complx_struct_comp'] = 10  # 10, 1000

    # ###################################################### #
    user_input_dict['cuda_index_mdSim'] = "0"
    user_input_dict['mdSim_overwrite_existing_results'] = False
    user_input_dict['max_cadidate_count_mdSim'] = 2
    
    # iterate over each of the post-processing stages
    for postproc_stage in postproc_stage_lst:
        trigger_pd_postproc(root_path=root_path, postproc_stage=postproc_stage, user_input_dict=user_input_dict)

