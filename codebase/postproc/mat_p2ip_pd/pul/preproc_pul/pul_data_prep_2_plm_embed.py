import os, sys

from pathlib import Path
path_root = Path(__file__).parents[4]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
import numpy as np
import pandas as pd
import torch
from utils import preproc_plm_util


def augment_plm_embed_for_pul_data(root_path='./', itr_tag=None, cuda_index=0):
    print('inside augment_plm_embed_for_pul_data() method - Start')
    # load the protTrans PLM model -Start
    device = torch.device(f'cuda:{cuda_index}')
    plm_file_location = os.path.join(root_path, '../ProtTrans_Models/')
    plm_name = 'prot_t5_xl_half_uniref50-enc'
    print('loading the protTrans PLM model -Start') 
    protTrans_model, tokenizer = preproc_plm_util.load_protTrans_model(protTrans_model_path=plm_file_location
                        , protTrans_model_name=plm_name, device=device)
    print('loading the protTrans PLM model -End') 
    # load the protTrans PLM model -End
    
    # Retrieve the preproc_1_pul_seq.csv, prepared at athe previous stage of data-preparation
    preproc_pul_dir_path = os.path.join(root_path, 'dataset/postproc_data/pul_result', f'{itr_tag}/preproc_pul')
    preproc_pul_seq_csv_path = os.path.join(preproc_pul_dir_path, 'preproc_1_pul_seq.csv')
    preproc_pul_seq_df = pd.read_csv(preproc_pul_seq_csv_path)

    # Split the preproc_pul_seq_df into n parts
    n_splits = preproc_pul_seq_df.shape[0]
    print(f'\nSplitting the preproc_pul_seq_df into {n_splits} parts where each part contains a single row\n')
    sub_dfs_lst = np.array_split(preproc_pul_seq_df, n_splits)  # Split into a list of n sub-dataframes
    
    processed_dfs_lst = []
    # Iterate over each sub-dataframe (df_0, df_1, ..., df_n)
    for sub_df_index in range(len(sub_dfs_lst)):
        print(f'\n\n ######### Processing sub-dataframe df_{sub_df_index} out of {len(sub_dfs_lst)} dataframes\n\n')
        sub_df = sub_dfs_lst[sub_df_index]
        
        # Process 'chain_1_seq' column and add A_col_* columns
        sub_df = process_sub_dataframe(sub_df_index=sub_df_index, sub_df=sub_df, seq_column='chain_1_seq'
                                       , prefix='A', protTrans_model=protTrans_model
                                       , tokenizer=tokenizer, device=device)
        
        # Process 'chain_2_seq' column and add B_col_* columns
        sub_df = process_sub_dataframe(sub_df_index=sub_df_index, sub_df=sub_df, seq_column='chain_2_seq'
                                       , prefix='B', protTrans_model=protTrans_model
                                       , tokenizer=tokenizer, device=device)
        
        # Save each processed sub-dataframe as a CSV
        # sub_df_csv_file_loc = os.path.join(root_path, 'dataset/postproc_data/pul_result', f'{itr_tag}/preproc_pul/plm_emdd_csv_for_pul')
        # PPIPUtils.createFolder(sub_df_csv_file_loc)
        # sub_df.to_csv(os.path.join(sub_df_csv_file_loc, f'sub_df{sub_df_index}.csv'), index=False)
        
        # Collect all processed sub-dataframes for final concatenation
        processed_dfs_lst.append(sub_df)
    # End of for loop: for sub_df_index in range(len(sub_dfs_lst)):

    # Concatenate all processed sub-dataframes into one final dataframe
    final_df = pd.concat(processed_dfs_lst, ignore_index=True)
    
    # Save the final dataframe as a CSV
    preproc_plm_embed_pul_seq_csv_file_loc = os.path.join(root_path, 'dataset/postproc_data/pul_result', f'{itr_tag}/preproc_pul')
    final_df.to_csv(os.path.join(preproc_plm_embed_pul_seq_csv_file_loc, 'preproc_2_plm_embed_pul.csv'), index=False)
    print('inside augment_plm_embed_for_pul_data() method - End')


def process_sub_dataframe(sub_df_index=0, sub_df=None, seq_column='chain_1_seq', prefix='A', protTrans_model=None
                          , tokenizer=None, device=None):
    print(f'sub_df_index: {sub_df_index}: seq_column: {seq_column}: inside process_sub_dataframe() method - Start')
    # print(f'Num. of rows: {sub_df.shape[0]}')
    
    seq_lst = sub_df[seq_column].tolist()  # Extract sequences from the column
    
    # ### extract PLM-based features
    plm_1d_embedd_tensor_lst = preproc_plm_util.extract_protTrans_plm_feat(prot_seq_lst=seq_lst, model=protTrans_model
                                                                           , tokenizer=tokenizer, device=device)
    # Convert torch tensors to numpy arrays
    plm_1d_np_arr_lst = [tensor.numpy() for tensor in plm_1d_embedd_tensor_lst]
    
    # Create a dataframe with 1024 additional columns
    embed_df = pd.DataFrame(plm_1d_np_arr_lst, columns=[f'{prefix}_col_{j}' for j in range(1024)])
    
    # Concatenate the new columns to the original dataframe
    sub_df = pd.concat([sub_df.reset_index(drop=True), embed_df.reset_index(drop=True)], axis=1)
    print(f'sub_df_index: {sub_df_index}: seq_column: {seq_column}: inside process_sub_dataframe() method - End')
    return sub_df


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/matpip_pd_prj')

    itr_tag = 'mcmc_fullLen_puFalse_batch5_mutPrcntLen10'

    cuda_index = 0
    # Prepare initial dataset required for the PU learning
    augment_plm_embed_for_pul_data(root_path=root_path, itr_tag=itr_tag, cuda_index=cuda_index)
    