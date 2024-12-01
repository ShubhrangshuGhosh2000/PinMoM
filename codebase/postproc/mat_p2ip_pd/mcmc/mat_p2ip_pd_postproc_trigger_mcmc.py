import os, sys
from pathlib import Path
path_root = Path(__file__).parents[3]
sys.path.insert(0, str(path_root))
from codebase.postproc.mat_p2ip_pd import mat_p2ip_pd_postproc_A4simuln
from utils import PPIPUtils


def trigger_pd_postproc(root_path = './', postproc_stage=None, user_input_dict=dict()):
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
    def_ppi_score_threshold = 0.8 
    input_dict['ppi_score_threshold'] = user_input_dict.get('ppi_score_threshold', def_ppi_score_threshold)
    def_postproc_psiblast_enabled = False  
    input_dict['postproc_psiblast_enabled'] = user_input_dict.get('postproc_psiblast_enabled', def_postproc_psiblast_enabled)
    def_postproc_psiblast_exec_path = os.path.join('specify/psiblast/executable/path/')
    input_dict['postproc_psiblast_exec_path'] = user_input_dict.get('postproc_psiblast_exec_path', def_postproc_psiblast_exec_path)
    def_postproc_psiblast_uniprot_db_path = os.path.join('specify/uniprot/db/path')
    input_dict['postproc_psiblast_uniprot_db_path'] = user_input_dict.get('postproc_psiblast_uniprot_db_path', def_postproc_psiblast_uniprot_db_path)
    def_postproc_psiblast_result_dir = os.path.join(root_path, f'dataset/postproc_data/psi_blast_result/')  
    input_dict['postproc_psiblast_result_dir'] = user_input_dict.get('postproc_psiblast_result_dir', def_postproc_psiblast_result_dir)
    def_psiBlast_percent_similarity_score_threshold = 70.0  
    input_dict['psiBlast_percent_similarity_score_threshold'] = user_input_dict.get('psiBlast_percent_similarity_score_threshold', def_psiBlast_percent_similarity_score_threshold)
    inp_dir_mut_seq_struct_pred_af2 = os.path.join(input_dict['af2_io_dir'], 'mut_seq_struct_pred_inp')
    def_max_num_of_seq_for_af2_chain_struct_pred = 150
    input_dict['max_num_of_seq_for_af2_chain_struct_pred'] = user_input_dict.get('max_num_of_seq_for_af2_chain_struct_pred', def_max_num_of_seq_for_af2_chain_struct_pred)
    PPIPUtils.createFolder(inp_dir_mut_seq_struct_pred_af2, recreate_if_exists=False)

    for dim_prot_complx_nm in input_dict['dim_prot_complx_nm_lst']:
        try:
            if(postproc_stage == '1_after_simuln'):
                mat_p2ip_pd_postproc_A4simuln.run_postproc_after_simuln(root_path = input_dict['root_path'], dim_prot_complx_nm = dim_prot_complx_nm
                                        , sim_result_dir = input_dict['sim_result_dir'], postproc_result_dir = input_dict['postproc_result_dir']
                                        , simulation_exec_mode = input_dict['simulation_exec_mode'], cuda_index = input_dict['cuda_index']
                                        , pdb_file_location = input_dict['pdb_file_location'], ppi_score_threshold = input_dict['ppi_score_threshold']
                                        , postproc_psiblast_enabled = input_dict['postproc_psiblast_enabled']
                                        , postproc_psiblast_exec_path = input_dict['postproc_psiblast_exec_path'], postproc_psiblast_uniprot_db_path = input_dict['postproc_psiblast_uniprot_db_path']
                                        , postproc_psiblast_result_dir = input_dict['postproc_psiblast_result_dir']
                                        , psiBlast_percent_similarity_score_threshold = input_dict['psiBlast_percent_similarity_score_threshold']
                                        , max_num_of_seq_for_af2_chain_struct_pred = input_dict['max_num_of_seq_for_af2_chain_struct_pred']
                                        , inp_dir_mut_seq_struct_pred_af2 = inp_dir_mut_seq_struct_pred_af2)
        except Exception as ex:
            print(f'************ ############## Error in processing :: {dim_prot_complx_nm} \n Error is: {ex}\n')
    print('inside trigger_pd_postproc() method - End')


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    postproc_stage_lst = ['1_after_simuln']
    user_input_dict = {}
    iteration_tag = 'mcmc_fullLen_puFalse_mutPrcntLen10'
    dim_prot_complx_nm_lst_orig = ['2I25', '4Y7M', '5JMO', '5SV3', '6CWG', '6DBG', '1AVX', '1AY7', '1BUH', '1BVN']
    # ['2I25', '4Y7M', '5JMO', '5SV3', '6CWG', '6DBG', '1AVX', '1AY7', '1BUH', '1BVN', '1CLV', '1D6R', ..........................., etc.] 
    dim_prot_complx_nm_lst_done =   []
    dim_prot_complx_nm_lst_effective = [prot_id for prot_id in dim_prot_complx_nm_lst_orig if prot_id not in dim_prot_complx_nm_lst_done]
    user_input_dict['dim_prot_complx_nm_lst'] =  dim_prot_complx_nm_lst_effective  
    user_input_dict['sim_result_dir'] = os.path.join(root_path, f'dataset/proc_data/result_dump_{iteration_tag}')  
    user_input_dict['simulation_exec_mode'] = 'mcmc'  
    user_input_dict['postproc_result_dir'] = os.path.join(root_path, f'dataset/postproc_data/result_dump/{iteration_tag}')  
    user_input_dict['postproc_psiblast_enabled'] = False  
    user_input_dict['postproc_psiblast_result_dir'] = os.path.join(root_path, f'dataset/postproc_data/psi_blast_result/{iteration_tag}')  
    user_input_dict['psiBlast_percent_similarity_score_threshold'] = 70  
    user_input_dict['max_num_of_seq_for_af2_chain_struct_pred'] = 100 
    user_input_dict['af2_io_dir'] = os.path.join(root_path, f'dataset/postproc_data/alphafold2_io/{iteration_tag}')  
    user_input_dict['cuda_index'] = 0  
    user_input_dict['ppi_score_threshold'] = 0.50  
    user_input_dict['cuda_index_mdSim'] = "0"
    trigger_pd_postproc(root_path=root_path, postproc_stage=postproc_stage_lst[0], user_input_dict=user_input_dict)

