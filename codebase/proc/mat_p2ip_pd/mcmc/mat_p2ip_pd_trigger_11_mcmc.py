import os, sys

from pathlib import Path
path_root = Path(__file__).parents[3]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

import gc
import traceback

from utils import dl_reproducible_result_util
from utils.early_stopping_criteria import EarlyStoppingCriteria
from utils.temperature_scheduler import TemperatureScheduler
from utils.resource_monitor import ResourceMonitor
from utils.mutation_number_adjuster import MutationNumberAdjuster
from proc.mat_p2ip_pd.mcmc.mat_p2ip_pd_mcmc import run_prot_design_simuln


def trigger_experiment(root_path = './', user_input_dict=dict()):
    print('inside trigger_experiment() method - Start')

    print(f'\n ===================\n user_input_dict: \n {user_input_dict}')
    # Create input_dict from user_input_dict.
    # User may chose to provide only a few arguments, required for the experiment.
    # input_dict would contain those user overridden argument values and the rest of the arguments with
    # their default values. In other words, input_dict would contain full set of arguments required for
    # the execution whereas user_input_dict would contain only a subset (which user chose to override) of it.
    input_dict = {'root_path': root_path}

    # location of the downloaded PDB files
    def_pdb_file_location = os.path.join(root_path, "dataset/preproc_data/pdb_files")  # default value
    # check if the user_input_dict contains value for the key 'pdb_file_location'; 
    # otherwise, set it to the default value
    input_dict['pdb_file_location'] = user_input_dict.get('pdb_file_location', def_pdb_file_location)

    # list of dimeric protein complex names to be used for the protein design through interaction
    def_dim_prot_complx_nm_lst = ['2I25']  # default value
    # check if the user_input_dict contains value for the key 'dim_prot_complx_nm_lst'; 
    # otherwise, set it to the default value
    input_dict['dim_prot_complx_nm_lst'] = user_input_dict.get('dim_prot_complx_nm_lst', def_dim_prot_complx_nm_lst)

    # Specify whether Protein Unit (PU) concept needs to be considered. It is a boolean (True/False) input. If 'use_prot_unit' is True, 
    # then 'batch_size' argument value is ignored and PU based mutation will take place.
    def_use_prot_unit = False  # default value
    # check if the user_input_dict contains value for the key 'use_prot_unit'; 
    # otherwise, set it to the default value
    input_dict['use_prot_unit'] = user_input_dict.get('use_prot_unit', def_use_prot_unit)

    # list for the number of iterations in the experiment
    def_num_of_itr_lst = [30000]  # [30000]  # default value
    # check if the user_input_dict contains value for the key 'num_of_itr_lst'; 
    # otherwise, set it to the default value
    input_dict['num_of_itr_lst'] = user_input_dict.get('num_of_itr_lst', def_num_of_itr_lst)

    # list for the percent length to be used to determine the number of mutation points in the experiment
    def_percent_len_for_calc_mut_pts_lst = [1]  # default value
    # check if the user_input_dict contains value for the key 'percent_len_for_calc_mut_pts_lst'; 
    # otherwise, set it to the default value
    input_dict['percent_len_for_calc_mut_pts_lst'] = user_input_dict.get('percent_len_for_calc_mut_pts_lst', def_percent_len_for_calc_mut_pts_lst)

    # execution mode type to be used for PPI prediction; Possible values: 'fast', 'thorough'
    def_exec_mode_type = 'thorough'  # default value
    # check if the user_input_dict contains value for the key 'exec_mode_type'; 
    # otherwise, set it to the default value
    input_dict['exec_mode_type'] = user_input_dict.get('exec_mode_type', def_exec_mode_type)

    # location of the PLM file
    def_plm_file_location = os.path.join(root_path, '../ProtTrans_Models/')  # default value
    # check if the user_input_dict contains value for the key 'plm_file_location'; 
    # otherwise, set it to the default value
    input_dict['plm_file_location'] = user_input_dict.get('plm_file_location', def_plm_file_location)

    # PLM name to be used to derive the PLM-based features; possible values: prot_t5_xl_half_uniref50-enc, prot_t5_xl_uniref50
    def_plm_name = 'prot_t5_xl_half_uniref50-enc'  # default value
    # check if the user_input_dict contains value for the key 'plm_name'; 
    # otherwise, set it to the default value
    input_dict['plm_name'] = user_input_dict.get('plm_name', def_plm_name)

    # batch size i.e. the number of candidate mutations to be evaluated through prediction generation in a batch.
    # As the total number of batches is the number of iterations in the experiment divided by batch size,
    # batch size ideally divide the number of iterations in the experiment.
    # If batch_size = 1 i.e. only a single candidate mutations to be evaluated at a time, the serial execution takes place but
    # if batch_size > 1 i.e. multiple candidate mutations to be evaluated at a time, the multiprocessing occurs.
    def_batch_size = 1  # default value
    # check if the user_input_dict contains value for the key 'batch_size'; 
    # otherwise, set it to the default value
    input_dict['batch_size'] = user_input_dict.get('batch_size', def_batch_size)

    # Number of batches after which result would be saved to disk during sumulation
    def_batch_dump_interval = 2000  # 2000  # default value
    # check if the user_input_dict contains value for the key 'batch_dump_interval'; 
    # otherwise, set it to the default value
    input_dict['batch_dump_interval'] = user_input_dict.get('batch_dump_interval', def_batch_dump_interval)

    # Location of the folder in which the simulation result will be dumped/saved 
    def_result_dump_dir = os.path.join(root_path, f'dataset/proc_data/result_dump')  # default value
    # check if the user_input_dict contains value for the key 'result_dump_dir'; 
    # otherwise, set it to the default value
    input_dict['result_dump_dir'] = user_input_dict.get('result_dump_dir', def_result_dump_dir)

    # Cuda index for GPU. Possible values are 0, 1 for two GPU devices in a single node.
    def_cuda_index = 0  # 0 default value
    # check if the user_input_dict contains value for the key 'cuda_index'; 
    # otherwise, set it to the de20fault value
    input_dict['cuda_index'] = user_input_dict.get('cuda_index', def_cuda_index)

    # Boolean flag indicating whether PSI-Blast be skipped in PSSM features (manually extracted 2D features) calculation for MaTPIP and is resorted to BLOSUM matrix
    def_use_psiblast_for_pssm = False  # default value
    # check if the user_input_dict contains value for the key 'use_psiblast_for_pssm'; 
    # otherwise, set it to the default value
    input_dict['use_psiblast_for_pssm'] = user_input_dict.get('use_psiblast_for_pssm', def_use_psiblast_for_pssm)

    # PSI-Blast executable path required for PSSM-based feature extraction (manual 2d features). It is relevant only if 'use_psiblast_for_pssm' flag is set to True.
    def_psiblast_exec_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/ncbi-blast-2.13.0+/bin/')  # default value
    # check if the user_input_dict contains value for the key 'psiblast_exec_path'; 
    # otherwise, set it to the default value
    input_dict['psiblast_exec_path'] = user_input_dict.get('psiblast_exec_path', def_psiblast_exec_path)

    # ############################ setting Early Stopping Criteria related inputs -Start ############################
    # Threshold value for the probability metric (one of the inputs for the EarlyStoppingCriteria instance).
    def_prob_threshold = 0.5  # default value
    # check if the user_input_dict contains value for the key 'prob_threshold'; 
    # otherwise, set it to the default value
    input_dict['prob_threshold'] = user_input_dict.get('prob_threshold', def_prob_threshold)

    # Number of batches to wait for improvement before stopping (one of the inputs for the EarlyStoppingCriteria instance).
    def_stopping_patience = 4000  # default value
    # check if the user_input_dict contains value for the key 'stopping_patience'; 
    # otherwise, set it to the default value
    input_dict['stopping_patience'] = user_input_dict.get('stopping_patience', def_stopping_patience)

    # Minimum increase in probability to qualify as improvement (one of the inputs for the EarlyStoppingCriteria instance).
    def_delta_prob_improv = 0.001  # default value
    # check if the user_input_dict contains value for the key 'delta_prob_improv'; 
    # otherwise, set it to the default value
    input_dict['delta_prob_improv'] = user_input_dict.get('delta_prob_improv', def_delta_prob_improv)

    # A boolean flag to indicate whether to check for improvement (one of the inputs for the EarlyStoppingCriteria instance).
    def_check_improvement = False  # default value
    # check if the user_input_dict contains value for the key 'check_improvement'; 
    # otherwise, set it to the default value
    input_dict['check_improvement'] = user_input_dict.get('check_improvement', def_check_improvement)

    # A boolean flag to indicate whether to check for threshold crossing. (one of the inputs for the EarlyStoppingCriteria instance).
    def_check_threshold = False  # default value
    # check if the user_input_dict contains value for the key 'check_threshold'; 
    # otherwise, set it to the default value
    input_dict['check_threshold'] = user_input_dict.get('check_threshold', def_check_threshold)
    # ############################ setting Early Stopping Criteria related inputs -End ############################

    # ############################ setting inputs for Temperature Scheduler used in MCMC algorithm  -Start ############################
    # A boolean flag to indicate whether to use temperature scheduler to control the temperature parameter as a part of Metropolis Criterion in Monte Carlo simulation (MCMC) during protein design through interaction procedure.
    # If this flag is True, then an instance of TemperatureScheduler class is used in MCMC algorithm and value for the input argument 'fixed_temp_mcmc' is ignored.
    # If this flag is False then a fixed temperature value (as specified in 'fixed_temp_mcmc' argument) is used
    def_use_temp_sch_mcmc = False  # default value
    # check if the user_input_dict contains value for the key 'use_temp_sch_mcmc'; 
    # otherwise, set it to the default value
    input_dict['use_temp_sch_mcmc'] = user_input_dict.get('use_temp_sch_mcmc', def_use_temp_sch_mcmc)

    # The fixed temperature value used as a part of Metropolis Criterion in Monte Carlo simulation (MCMC). If 'use_temp_sch_mcmc' argument is True, then this parameter value is ignored in favor of an instance of TemperatureScheduler class.
    def_fixed_temp_mcmc = 0.2  # default value
    # check if the user_input_dict contains value for the key 'fixed_temp_mcmc'; 
    # otherwise, set it to the default value
    input_dict['fixed_temp_mcmc'] = user_input_dict.get('fixed_temp_mcmc', def_fixed_temp_mcmc)

    # Initial temperature value for the scheduler (one of the inputs for the TemperatureScheduler instance).
    def_initial_temp_sch = 0.2  # default value
    # check if the user_input_dict contains value for the key 'initial_temp_sch'; 
    # otherwise, set it to the default value
    input_dict['initial_temp_sch'] = user_input_dict.get('initial_temp_sch', def_initial_temp_sch)

    # Factor by which the temperature is increased in the temperature scheduler. e.g. if temp_inc_factor_sch = 0.1, then new_temp = old_temp * 1.1 i.e. 10% increase in the current temperature. Its value will be in the range (0.0, 1.0). 
    def_temp_inc_factor_sch = 0.1  # default value
    # check if the user_input_dict contains value for the key 'temp_inc_factor_sch'; 
    # otherwise, set it to the default value
    input_dict['temp_inc_factor_sch'] = user_input_dict.get('temp_inc_factor_sch', def_temp_inc_factor_sch)

    # Number of batches with no probability improvement before increasing the temperature by scheduler (one of the inputs for the TemperatureScheduler instance).
    def_patience_for_inc_sch = 400  # default value
    # check if the user_input_dict contains value for the key 'patience_for_inc_sch'; 
    # otherwise, set it to the default value
    input_dict['patience_for_inc_sch'] = user_input_dict.get('patience_for_inc_sch', def_patience_for_inc_sch)

    # Number of batches to wait before resuming normal scheduling operation after temperature change (one of the inputs for the TemperatureScheduler instance).
    def_cooldown_sch = 100  # default value
    # check if the user_input_dict contains value for the key 'cooldown_sch'; 
    # otherwise, set it to the default value
    input_dict['cooldown_sch'] = user_input_dict.get('cooldown_sch', def_cooldown_sch)

    # Upper bound on temperature for the temperature scheduler (one of the inputs for the TemperatureScheduler instance).
    def_max_temp_sch = 1.0  # default value
    # check if the user_input_dict contains value for the key 'max_temp_sch'; 
    # otherwise, set it to the default value
    input_dict['max_temp_sch'] = user_input_dict.get('max_temp_sch', def_max_temp_sch)

    # Whether to enable verbose output from temperature scheduler (one of the inputs for the TemperatureScheduler instance).
    def_verbose_sch = False  # default value
    # check if the user_input_dict contains value for the key 'verbose_sch'; 
    # otherwise, set it to the default value
    input_dict['verbose_sch'] = user_input_dict.get('verbose_sch', def_verbose_sch)
    # ############################ setting inputs for Temperature Scheduler used in MCMC algorithm  -End ############################

    # ############################ setting Resource Monitor related inputs -Start ############################
    # A boolean flag to indicate whether to  monitor the resource (RAM, GPU) usage during protein design through interaction procedure.
    # If this flag is True, then an instance of ResourceMonitor class is created and usage of RAM, GPU, etc. is monitored periodically at each batch result dumping time. Please note that, periodic resource monitoring would cause an increase in total time of execution for the simulation.
    # If this flag is False then only maximum number of CPU cores used would be monitored during simulation.
    def_use_resource_monitor = True  # default value
    # check if the user_input_dict contains value for the key 'use_resource_monitor'; 
    # otherwise, set it to the default value
    input_dict['use_resource_monitor'] = user_input_dict.get('use_resource_monitor', def_use_resource_monitor)

    # Whether to enable verbose output from ResourceMonitor class (one of the inputs for the ResourceMonitor instance).
    def_verbose_mon = False  # default value
    # check if the user_input_dict contains value for the key 'verbose_mon'; 
    # otherwise, set it to the default value
    input_dict['verbose_mon'] = user_input_dict.get('verbose_mon', def_verbose_mon)
    # ############################ setting Resource Monitor related inputs -End ############################

    # ############################ setting Mutation Number Adjuster related inputs -Start ############################
    # A boolean flag to indicate whether to enable the Mutation Number Adjuster which perform dynamic adjustment of number of mutation points based on the probability of interaction and the stage of simulation.
    # If this flag is True, then an instance of MutationNumberAdjuster class is created and number of mutation points are adjusted after each batch.
    # If this flag is False then no dynamic adjustment of number of mutation points take place during simulation.
    def_use_mut_num_adjuster = False  # default value
    # check if the user_input_dict contains value for the key 'use_mut_num_adjuster'; 
    # otherwise, set it to the default value
    input_dict['use_mut_num_adjuster'] = user_input_dict.get('use_mut_num_adjuster', def_use_mut_num_adjuster)
    
    # Threshold value for the PPI probability metric. If this threshold is reached then 'fixed mutation number' strategy will not be applied and Early Stopping Criterion check will be enabled.
    def_prob_threshold_for_mut_num_adj_mna = 0.8  # default value
    # check if the user_input_dict contains value for the key 'prob_threshold_for_mut_num_adj_mna'; 
    # otherwise, set it to the default value
    input_dict['prob_threshold_for_mut_num_adj_mna'] = user_input_dict.get('prob_threshold_for_mut_num_adj_mna', def_prob_threshold_for_mut_num_adj_mna)
    
    # Number of stages in which the entire simulation process will be divided w.r.t. the total number of batches. 
    # For example, if total number of simulations=30k, batch_size=5 and num_of_stages_mna_mna=6 implies the entire simulation will be broken into 6 different ranges w.r.t. the total number of batches and they are [0-1k), [1k-2k), [2k -3k), [3k -4k), [4k - 5k] and [5k - 6k).
    def_num_of_stages_mna = 6  # default value
    # check if the user_input_dict contains value for the key 'num_of_stages_mna'; 
    # otherwise, set it to the default value
    input_dict['num_of_stages_mna'] = user_input_dict.get('num_of_stages_mna', def_num_of_stages_mna)

    # Stage number from which fixed number of mutations will take place. For example, if 'fixed_mut_num_trigg_stage_mna' = 4 in the above example, then from batch index 3k onwards fixed number of mutations will take place.
    def_fixed_mut_num_trigg_stage_mna = 4  # default value
    # check if the user_input_dict contains value for the key 'fixed_mut_num_trigg_stage_mna'; 
    # otherwise, set it to the default value
    input_dict['fixed_mut_num_trigg_stage_mna'] = user_input_dict.get('fixed_mut_num_trigg_stage_mna', def_fixed_mut_num_trigg_stage_mna)

    # The integer indicating the fixed number of mutations which will take place from 'fixed_mut_num_trigg_stage_mna' onwards.
    def_fixed_mut_num_mna = 1  # default value
    # check if the user_input_dict contains value for the key 'fixed_mut_num_mna'; 
    # otherwise, set it to the default value
    input_dict['fixed_mut_num_mna'] = user_input_dict.get('fixed_mut_num_mna', def_fixed_mut_num_mna)

    # The integer representing the maximum possible number of mutations in a single iteration of the simulation.
    def_max_thrshld_for_num_of_mut_pts = 10  # default value
    # check if the user_input_dict contains value for the key 'max_thrshld_for_num_of_mut_pts'; 
    # otherwise, set it to the default value
    input_dict['max_thrshld_for_num_of_mut_pts'] = user_input_dict.get('max_thrshld_for_num_of_mut_pts', def_max_thrshld_for_num_of_mut_pts)
    # ############################ setting Mutation Number Adjuster related inputs -End ############################

    # ############################ setting interface residues related inputs -Start ############################
    # A boolean flag to indicate whether to consider only the interface residue index positions of the two chains of the dimer for the mutation.
    # If this flag is True, then 'use_prot_unit' argument value is ignored and the mutation takes place at the interface residue index positions.
    def_mut_only_at_intrfc_resid_idx = False  # default value
    # check if the user_input_dict contains value for the key 'mut_only_at_intrfc_resid_idx'; 
    # otherwise, set it to the default value
    input_dict['mut_only_at_intrfc_resid_idx'] = user_input_dict.get('mut_only_at_intrfc_resid_idx', def_mut_only_at_intrfc_resid_idx)
    
    # location of the NACCESS executable
    def_naccess_path = "/scratch/pralaycs/Shubh_Working_Remote/naccess/naccess"  # default value
    # check if the user_input_dict contains value for the key 'naccess_path'; 
    # otherwise, set it to the default value
    input_dict['naccess_path'] = user_input_dict.get('naccess_path', def_naccess_path)
    # ############################ setting interface residues related inputs -End ############################

    print(f'\n ===================\n input_dict: \n {input_dict}')
    print(f'\n ===================\n Creating EarlyStoppingCriteria instance...')
    earlyStoppingCriteria_inst = EarlyStoppingCriteria(prob_threshold=input_dict['prob_threshold'], stopping_patience=input_dict['stopping_patience']
                                                       , delta_prob_improv=input_dict['delta_prob_improv'], check_improvement=input_dict['check_improvement']
                                                       , check_threshold=input_dict['check_threshold'])

    temperatureScheduler_inst = None
    if(input_dict['use_temp_sch_mcmc']):
        print(f'\n ===================\n Creating TemperatureScheduler instance...')
        temperatureScheduler_inst = TemperatureScheduler(initial_temp_sch=input_dict['initial_temp_sch'], temp_inc_factor_sch=input_dict['temp_inc_factor_sch']
                                                         , patience_for_inc_sch=input_dict['patience_for_inc_sch'], cooldown_sch=input_dict['cooldown_sch']
                                                         , max_temp_sch=input_dict['max_temp_sch'], verbose_sch=input_dict['verbose_sch'])
        
    resourceMonitor_inst = None
    if(input_dict['use_resource_monitor']):
        print(f'\n ===================\n Creating ResourceMonitor instance...')
        resourceMonitor_inst = ResourceMonitor(cuda_index=input_dict['cuda_index'], verbose_mon=input_dict['verbose_mon'])

    mutationNumberAdjuster_inst = None
    if(input_dict['use_mut_num_adjuster']):
        print(f'\n ===================\n Creating MutationNumberAdjuster instance...')
        mutationNumberAdjuster_inst = MutationNumberAdjuster(prob_threshold_for_mut_num_adj_mna=input_dict['prob_threshold_for_mut_num_adj_mna']
                                                             , num_of_itr=input_dict['num_of_itr_lst'][0], batch_size=input_dict['batch_size']
                                                             , num_of_stages_mna=input_dict['num_of_stages_mna'], fixed_mut_num_trigg_stage_mna=input_dict['fixed_mut_num_trigg_stage_mna']
                                                             , fixed_mut_num_mna=input_dict['fixed_mut_num_mna'])

    for dim_prot_complx_nm in input_dict['dim_prot_complx_nm_lst']:
        try:
            for num_of_itr in input_dict['num_of_itr_lst']:
                for percent_len_for_calc_mut_pts in input_dict['percent_len_for_calc_mut_pts_lst']:
                    # trigger the actual experiment
                    run_prot_design_simuln(root_path=input_dict['root_path'], dim_prot_complx_nm=dim_prot_complx_nm, use_prot_unit=input_dict['use_prot_unit'], num_of_itr=num_of_itr
                                        , percent_len_for_calc_mut_pts=percent_len_for_calc_mut_pts, exec_mode_type=input_dict['exec_mode_type'], plm_file_location=input_dict['plm_file_location']
                                        , plm_name=input_dict['plm_name'], batch_size=input_dict['batch_size'], batch_dump_interval=input_dict['batch_dump_interval']
                                        , result_dump_dir=input_dict['result_dump_dir'], cuda_index=input_dict['cuda_index']
                                        , use_psiblast_for_pssm=input_dict['use_psiblast_for_pssm'], psiblast_exec_path=input_dict['psiblast_exec_path']
                                        , pdb_file_location=input_dict['pdb_file_location'], mut_only_at_intrfc_resid_idx=input_dict['mut_only_at_intrfc_resid_idx']
                                        , naccess_path=input_dict['naccess_path'], early_stop_checkpoint=earlyStoppingCriteria_inst
                                        , fixed_temp_mcmc=input_dict['fixed_temp_mcmc'], temperature_scheduler=temperatureScheduler_inst
                                        , resource_monitor=resourceMonitor_inst, mutation_number_adjuster=mutationNumberAdjuster_inst
                                        , max_thrshld_for_num_of_mut_pts=input_dict['max_thrshld_for_num_of_mut_pts'])
                    # call for the gc
                    # gc.collect()
                # end of for loop: for percent_len_for_calc_mut_pts in input_dict['percent_len_for_calc_mut_pts_lst']:
            # end of for loop: for num_of_itr in input_dict['num_of_itr_lst']:
        except Exception as ex:
            # printing stack trace 
            traceback.print_exc(file=sys.stdout)
            print(f'************ ############## Error in processing :: {dim_prot_complx_nm} :: ex: {ex}')
    # end of for loop: for dim_prot_complx_nm in input_dict['dim_prot_complx_nm_lst']:
    print('inside trigger_experiment() method - End')



if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/matpip_pd_prj')

    user_input_dict = {}

    # ###################################################### #
    # ###################################################### #
    # dim_prot_complx_nm_lst_11_orig = ['1CLV']
    dim_prot_complx_nm_lst_11_orig = ['1CLV', '1D6R', '1DFJ', '1E6E', '1EAW', '1EWY', '1F34', '1FLE', '1GL1', '1GLA']
    # dim_prot_complx_nm_lst_11_orig = ['2I25', '4Y7M', '5JMO', '5SV3', '6CWG', '6DBG', '1AVX', '1AY7', '1BUH', '1BVN'] \
    # + ['1CLV', '1D6R', '1DFJ', '1E6E', '1EAW', '1EWY', '1F34', '1FLE', '1GL1', '1GLA'] \
    # + ['1GXD', '1JTG', '1MAH', '1OC0', '1OPH', '1OYV', '1PPE', '1R0R', '1TMQ'] \
    # + ['1UDI', '1US7', '1YVB', '1Z5Y', '2A9K', '2ABZ', '2AYO', '2B42', '2J0T', '2O8V'] \
    # + ['2OOB', '2OUL', '2PCC', '2SIC', '2SNI', '2UUY', '3SGQ', '4CPA', '7CEI', '1AK4'] \
    # + ['1E96', '1EFN', '1FFW', '1FQJ', '1GCQ', '1GHQ', '1GPW', '1H9D', '1HE1', '1J2J'] \
    # + ['1KAC', '1KTZ', '1KXP', '1PVH', '1QA9', '1S1Q', '1SBB', '1T6B', '1XD3', '1Z0K'] \
    # + ['1ZHH', '1ZHI', '2A5T', '2AJF', '2BTF', '2FJU', '2G77', '2HLE', '2HQS', '2VDB', '3D5S']
    dim_prot_complx_nm_lst_done =   []
    dim_prot_complx_nm_lst_notConsidered = ['5JMO', '1PPE', '1FQJ', '1OYV', '2BTF', '1H9D', '1BUH', '1QA9', '1S1Q', '1XD3', '1PVH']
    dim_prot_complx_nm_lst_chainBreak = ['1AVX', '1AY7', '1BUH', '1D6R', '1EAW', '1EFN', '1F34', '1FLE', '1GL1', '1GLA', '1GPW', '1GXD', '1H9D', '1JTG', '1KTZ', '1KXP', '1MAH', '1OC0', '1OPH', '1OYV' \
                                        ,'1PPE', '1R0R', '1S1Q', '1SBB', '1T6B', '1US7', '1XD3', '1YVB', '1Z5Y', '1ZHH', '1ZHI', '2AST', '2ABZ', '2AJF', '2AYO', '2B42', '2BTF', '2FJU', '2G77', '2HLE' \
                                        ,'2HQS', '2O8V', '2PCC', '2UUY', '2VDB', '3SGQ', '4Y7M', '5JMO', '5SV3', '6CWG', '6DBG']
    dim_prot_complx_nm_lst_excluded = dim_prot_complx_nm_lst_done + dim_prot_complx_nm_lst_notConsidered
    dim_prot_complx_nm_lst_effective = [prot_id for prot_id in dim_prot_complx_nm_lst_11_orig if prot_id not in dim_prot_complx_nm_lst_excluded]
    user_input_dict['dim_prot_complx_nm_lst'] =  dim_prot_complx_nm_lst_effective  #
    # ###################################################### #
    # ###################################################### #

    user_input_dict['cuda_index'] = 0  #
    user_input_dict['num_of_itr_lst'] = [30000]  #
    user_input_dict['batch_size'] = 5  # batch_size = 5 for Full length and 2 for interface;
    user_input_dict['batch_dump_interval'] = 1000  # 5000 for batch size of 1; 3000 for batch size of 3 (or when use_prot_unit=True); 2000 for batch_size of 5
    user_input_dict['use_prot_unit'] = False  #

    # ###################################################### #
    user_input_dict['percent_len_for_calc_mut_pts_lst'] = [40]   # 10 for fullLen; 40 for intrfc
    # ###################################################### #

    # ###################################################### #
    user_input_dict['exec_mode_type'] = 'thorough'  # 'fast', 'thorough'
    # ###################################################### #
    
    # ###################################################### #
    user_input_dict['result_dump_dir'] = os.path.join(root_path, f'dataset/proc_data/result_dump_mcmc_intrfc_puFalse_batch5_mutPrcntLen10')
    # ###################################################### #
    user_input_dict['fixed_temp_mcmc'] = 0.2
    # ###################################################### #
    user_input_dict['mut_only_at_intrfc_resid_idx'] = True
    # ###################################################### #

    # ###################################################### #
    user_input_dict['use_psiblast_for_pssm'] = False
    # ###################################################### #

    # early stopping
    user_input_dict['stopping_patience'] = 2000  # stopping_patience = 1000 for Full length and 500 for interface;
    user_input_dict['check_improvement'] = False #
    user_input_dict['delta_prob_improv'] = 0.001  #
    user_input_dict['check_threshold'] = False  #
    user_input_dict['prob_threshold'] = 0.5  #
    
    # temperature scheduler
    user_input_dict['use_temp_sch_mcmc'] = False
    user_input_dict['patience_for_inc_sch'] = 250  # 1000 for batch size of 1; 400 for batch size of 3 (or when use_prot_unit=True); 250 for batch_size of 5
    user_input_dict['cooldown_sch'] = 100  # 400 for batch size of 1; 150 for batch size of 3 (or when use_prot_unit=True); 100 for batch_size of 5
    user_input_dict['verbose_sch'] = True
    user_input_dict['verbose_mon'] = True

    # mutation number adjuster
    user_input_dict['use_mut_num_adjuster'] = False

    trigger_experiment(root_path=root_path, user_input_dict=user_input_dict)

