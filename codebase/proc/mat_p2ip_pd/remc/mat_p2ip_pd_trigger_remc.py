import os, sys

from pathlib import Path
path_root = Path(__file__).parents[3]  
sys.path.insert(0, str(path_root))


import gc
import traceback

from utils import dl_reproducible_result_util
from utils.resource_monitor import ResourceMonitor
from proc.mat_p2ip_pd.mcmc.mat_p2ip_pd_mcmc import run_prot_design_simuln


def trigger_experiment(root_path = './', user_input_dict=dict()):
    input_dict = {'root_path': root_path}
    def_pdb_file_location = os.path.join(root_path, "dataset/preproc_data/pdb_files")  
    input_dict['pdb_file_location'] = user_input_dict.get('pdb_file_location', def_pdb_file_location)
    def_dim_prot_complx_nm_lst = ['2I25']  
    input_dict['dim_prot_complx_nm_lst'] = user_input_dict.get('dim_prot_complx_nm_lst', def_dim_prot_complx_nm_lst)
    def_num_of_itr_lst = [30000]  
    input_dict['num_of_itr_lst'] = user_input_dict.get('num_of_itr_lst', def_num_of_itr_lst)
    def_percent_len_for_calc_mut_pts_lst = [1]  
    input_dict['percent_len_for_calc_mut_pts_lst'] = user_input_dict.get('percent_len_for_calc_mut_pts_lst', def_percent_len_for_calc_mut_pts_lst)
    def_exec_mode_type = 'thorough'  
    input_dict['exec_mode_type'] = user_input_dict.get('exec_mode_type', def_exec_mode_type)
    def_plm_file_location = os.path.join(root_path, '../ProtTrans_Models/')  
    input_dict['plm_file_location'] = user_input_dict.get('plm_file_location', def_plm_file_location)
    def_plm_name = 'prot_t5_xl_half_uniref50-enc'  
    input_dict['plm_name'] = user_input_dict.get('plm_name', def_plm_name)
    def_local_step_size = 2000  
    input_dict['local_step_size'] = user_input_dict.get('local_step_size', def_local_step_size)
    def_result_dump_dir = os.path.join(root_path, f'dataset/proc_data/result_dump')  
    input_dict['result_dump_dir'] = user_input_dict.get('result_dump_dir', def_result_dump_dir)
    def_cuda_index = 0  
    input_dict['cuda_index'] = user_input_dict.get('cuda_index', def_cuda_index)
    def_use_prot_unit = False
    input_dict['use_prot_unit'] = user_input_dict.get('use_prot_unit', def_use_prot_unit)
    def_use_psiblast_for_pssm = False  
    input_dict['use_psiblast_for_pssm'] = user_input_dict.get('use_psiblast_for_pssm', def_use_psiblast_for_pssm)
    def_psiblast_exec_path = os.path.join('/path/to/psi/blast/executable')  
    input_dict['psiblast_exec_path'] = user_input_dict.get('psiblast_exec_path', def_psiblast_exec_path)
    def_use_resource_monitor = True  
    input_dict['use_resource_monitor'] = user_input_dict.get('use_resource_monitor', def_use_resource_monitor)
    def_verbose_mon = False  
    input_dict['verbose_mon'] = user_input_dict.get('verbose_mon', def_verbose_mon)
    def_max_thrshld_for_num_of_mut_pts = 10  
    input_dict['max_thrshld_for_num_of_mut_pts'] = user_input_dict.get('max_thrshld_for_num_of_mut_pts', def_max_thrshld_for_num_of_mut_pts)
    def_temp_min_remc = 0.02
    input_dict['temp_min_remc'] = user_input_dict.get('temp_min_remc', def_temp_min_remc)
    def_temp_max_remc = 0.2
    input_dict['temp_max_remc'] = user_input_dict.get('temp_max_remc', def_temp_max_remc)
    def_num_of_replica_remc = 10
    input_dict['num_of_replica_remc'] = user_input_dict.get('num_of_replica_remc', def_num_of_replica_remc)
    def_result_save_interval_remc = 2
    input_dict['result_save_interval_remc'] = user_input_dict.get('result_save_interval_remc', def_result_save_interval_remc)
    resourceMonitor_inst = None
    if(input_dict['use_resource_monitor']):
        resourceMonitor_inst = ResourceMonitor(cuda_index=input_dict['cuda_index'], verbose_mon=input_dict['verbose_mon'])
    for dim_prot_complx_nm in input_dict['dim_prot_complx_nm_lst']:
        try:
            for num_of_itr in input_dict['num_of_itr_lst']:
                for percent_len_for_calc_mut_pts in input_dict['percent_len_for_calc_mut_pts_lst']:
                    run_prot_design_simuln(root_path=input_dict['root_path'], dim_prot_complx_nm=dim_prot_complx_nm, use_prot_unit=input_dict['use_prot_unit'], num_of_itr=num_of_itr
                                        , percent_len_for_calc_mut_pts=percent_len_for_calc_mut_pts, exec_mode_type=input_dict['exec_mode_type'], plm_file_location=input_dict['plm_file_location']
                                        , plm_name=input_dict['plm_name'], local_step_size=input_dict['local_step_size']
                                        , result_dump_dir=input_dict['result_dump_dir'], cuda_index=input_dict['cuda_index']
                                        , use_psiblast_for_pssm=input_dict['use_psiblast_for_pssm'], psiblast_exec_path=input_dict['psiblast_exec_path']
                                        , pdb_file_location=input_dict['pdb_file_location'], mut_only_at_intrfc_resid_idx=input_dict['mut_only_at_intrfc_resid_idx']
                                        , resource_monitor=resourceMonitor_inst, max_thrshld_for_num_of_mut_pts=input_dict['max_thrshld_for_num_of_mut_pts']
                                        , temp_min_remc=input_dict['temp_min_remc'], temp_max_remc=input_dict['temp_max_remc'], num_of_replica_remc=input_dict['num_of_replica_remc']
                                        , result_save_interval_remc=input_dict['result_save_interval_remc'])
        except Exception as ex:
            traceback.print_exc(file=sys.stdout)


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')

    user_input_dict = {}
    # dim_prot_complx_nm_lst_orig = ['2I25']
    dim_prot_complx_nm_lst_orig = ['2I25', '4Y7M', '5JMO', '5SV3', '6CWG', '6DBG', '1AVX', '1AY7', '1BUH', '1BVN']
    # ['2I25', '4Y7M', '5JMO', '5SV3', '6CWG', '6DBG', '1AVX', '1AY7', '1BUH', '1BVN', '1CLV', '1D6R', ..........................., etc.] 
    dim_prot_complx_nm_lst_done =   []
    dim_prot_complx_nm_lst_effective = [prot_id for prot_id in dim_prot_complx_nm_lst_orig if prot_id not in dim_prot_complx_nm_lst_done]
    user_input_dict['dim_prot_complx_nm_lst'] =  dim_prot_complx_nm_lst_effective  #
    user_input_dict['cuda_index'] = 0
    user_input_dict['num_of_itr_lst'] = [10000]
    user_input_dict['local_step_size'] = 500
    user_input_dict['num_of_replica_remc'] = 5  
    user_input_dict['result_save_interval_remc'] = 5 
    user_input_dict['temp_min_remc'] = 0.5 
    user_input_dict['temp_max_remc'] = 1.5 
    user_input_dict['percent_len_for_calc_mut_pts_lst'] = [10] 
    user_input_dict['exec_mode_type'] = 'thorough'
    user_input_dict['result_dump_dir'] = os.path.join(root_path, f'dataset/proc_data/demo_result')
    user_input_dict['mut_only_at_intrfc_resid_idx'] = False
    user_input_dict['use_psiblast_for_pssm'] = False
    trigger_experiment(root_path=root_path, user_input_dict=user_input_dict)

