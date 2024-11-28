import os, sys

from pathlib import Path
path_root = Path(__file__).parents[4]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import PPIPUtils


def apply_normalized_pca_to_embed_features(root_path='./', itr_tag=None, num_of_reduced_feat=50):
    print('inside apply_normalized_pca_to_embed_features() method - Start')
    
    # Retrieve the preproc_2_plm_embed_pul.csv, prepared at the previous stage of data-preparation
    preproc_plm_embed_pul_seq_csv_file_loc = os.path.join(root_path, 'dataset/postproc_data/pul_result', f'{itr_tag}/preproc_pul')
    preproc_plm_embed_df = pd.read_csv(os.path.join(preproc_plm_embed_pul_seq_csv_file_loc, 'preproc_2_plm_embed_pul.csv'))

    # Extract columns that start with 'A_' (A_col_0 to A_col_n)
    A_columns = [col for col in preproc_plm_embed_df.columns if col.startswith('A_col')]
    A_data = preproc_plm_embed_df[A_columns]

    # Extract columns that start with 'B_' (B_col_0 to B_col_n)
    B_columns = [col for col in preproc_plm_embed_df.columns if col.startswith('B_col')]
    B_data = preproc_plm_embed_df[B_columns]

    # Normalize A_data using StandardScaler
    scaler_A = StandardScaler()
    A_scaled = scaler_A.fit_transform(A_data)

    # Normalize B_data using StandardScaler
    scaler_B = StandardScaler()
    B_scaled = scaler_B.fit_transform(B_data)

    # Apply PCA to reduce dimensions of A_data to num_of_reduced_feat
    pca_A = PCA(n_components=num_of_reduced_feat, random_state=456)
    A_reduced = pca_A.fit_transform(A_scaled)
    
    # Convert the reduced data to a DataFrame with appropriate column names (AP_col_0, AP_col_1, ..., AP_col_49)
    A_reduced_df = pd.DataFrame(A_reduced, columns=[f'AP_col_{i}' for i in range(num_of_reduced_feat)])

    # Apply PCA to reduce dimensions of B_data to num_of_reduced_feat
    pca_B = PCA(n_components=num_of_reduced_feat, random_state=456)
    B_reduced = pca_B.fit_transform(B_scaled)

    # Convert the reduced data to a DataFrame with appropriate column names (BP_col_0, BP_col_1, ..., BP_col_49)
    B_reduced_df = pd.DataFrame(B_reduced, columns=[f'BP_col_{i}' for i in range(num_of_reduced_feat)])

    # Combine the original columns ('id', 'cseq1', 'cseq2', 'label', 'ppi_score', 'num_mut_pts') with the reduced data
    result_df = pd.concat([preproc_plm_embed_df[['prot_id', 'chain_1_seq', 'chain_2_seq', 'label', 'ppi_score', 'num_mut_pts']], A_reduced_df, B_reduced_df], axis=1)

    # Save the resulting dataframe to a CSV file
    embed_feat_pca_norm_pul_csv_file_loc = os.path.join(root_path, 'dataset/postproc_data/pul_result', f'{itr_tag}/preproc_pul')
    result_df.to_csv(os.path.join(embed_feat_pca_norm_pul_csv_file_loc, 'preproc_3_embed_pca_norm_pul.csv'), index=False)
    print('inside apply_normalized_pca_to_embed_features() method - End')




if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/matpip_pd_prj')

    itr_tag = 'mcmc_fullLen_puFalse_batch5_mutPrcntLen10'

    # Apply PCA on the normalized PLM embeddings
    num_of_reduced_feat = 50
    apply_normalized_pca_to_embed_features(root_path=root_path, itr_tag=itr_tag, num_of_reduced_feat=num_of_reduced_feat)
    