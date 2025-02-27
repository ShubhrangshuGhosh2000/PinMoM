import os, sys

from pathlib import Path
path_root = Path(__file__).parents[3]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

import gc
import traceback

from utils import dl_reproducible_result_util
from utils.resource_monitor import ResourceMonitor
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

    def_use_prot_unit = False
    input_dict['use_prot_unit'] = user_input_dict.get('use_prot_unit', def_use_prot_unit)

    def_batch_size = 1  # default value
    input_dict['batch_size'] = user_input_dict.get('batch_size', def_batch_size)

    def_use_psiblast_for_pssm = False  # default value
    input_dict['use_psiblast_for_pssm'] = user_input_dict.get('use_psiblast_for_pssm', def_use_psiblast_for_pssm)

    def_psiblast_exec_path = os.path.join('specify/path/to/psi/blast/executable')  # default value
    input_dict['psiblast_exec_path'] = user_input_dict.get('psiblast_exec_path', def_psiblast_exec_path)

    def_fixed_temp_mcmc = 0.2  # default value
    input_dict['fixed_temp_mcmc'] = user_input_dict.get('fixed_temp_mcmc', def_fixed_temp_mcmc)

    def_use_resource_monitor = True  # default value
    input_dict['use_resource_monitor'] = user_input_dict.get('use_resource_monitor', def_use_resource_monitor)

    def_verbose_mon = False  # default value
    input_dict['verbose_mon'] = user_input_dict.get('verbose_mon', def_verbose_mon)

    def_max_thrshld_for_num_of_mut_pts = 10  # default value
    input_dict['max_thrshld_for_num_of_mut_pts'] = user_input_dict.get('max_thrshld_for_num_of_mut_pts', def_max_thrshld_for_num_of_mut_pts)

    resourceMonitor_inst = None
    if(input_dict['use_resource_monitor']):
        resourceMonitor_inst = ResourceMonitor(cuda_index=input_dict['cuda_index'], verbose_mon=input_dict['verbose_mon'])

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
                                        , pdb_file_location=input_dict['pdb_file_location']
                                        , fixed_temp_mcmc=input_dict['fixed_temp_mcmc'], resource_monitor=resourceMonitor_inst
                                        , max_thrshld_for_num_of_mut_pts=input_dict['max_thrshld_for_num_of_mut_pts'])
        except Exception as ex:
            traceback.print_exc(file=sys.stdout)
            print(f'************ ############## Error in processing :: {dim_prot_complx_nm} :: ex: {ex}')
    print('inside trigger_experiment() method - End')




if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')

    user_input_dict = {}
    # dim_prot_complx_nm_lst_orig = ['2I25']
    dim_prot_complx_nm_lst_orig = ['2I25', '4Y7M', '5JMO', '5SV3', '6CWG', '6DBG', '1AVX', '1AY7', '1BUH', '1BVN']
    # ['2I25', '4Y7M', '5JMO', '5SV3', '6CWG', '6DBG', '1AVX', '1AY7', '1BUH', '1BVN', '1CLV', '1D6R', ..........................., etc.] 
    dim_prot_complx_nm_lst_done =   []
    dim_prot_complx_nm_lst_effective = [prot_id for prot_id in dim_prot_complx_nm_lst_orig if prot_id not in dim_prot_complx_nm_lst_done]
    user_input_dict['dim_prot_complx_nm_lst'] =  dim_prot_complx_nm_lst_effective  #

    user_input_dict['cuda_index'] = 0  #
    user_input_dict['num_of_itr_lst'] = [30000]  #
    user_input_dict['batch_size'] = 5
    user_input_dict['batch_dump_interval'] = 1000
    user_input_dict['use_prot_unit'] = False  #

    user_input_dict['percent_len_for_calc_mut_pts_lst'] = [10]
    user_input_dict['exec_mode_type'] = 'thorough'
    user_input_dict['result_dump_dir'] = os.path.join(root_path, f'dataset/proc_data/demo_result')
    user_input_dict['fixed_temp_mcmc'] = 0.2
    user_input_dict['mut_only_at_intrfc_resid_idx'] = False
    user_input_dict['use_psiblast_for_pssm'] = False

    trigger_experiment(root_path=root_path, user_input_dict=user_input_dict)

