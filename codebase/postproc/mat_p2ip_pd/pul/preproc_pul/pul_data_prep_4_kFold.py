import os, sys

from pathlib import Path
path_root = Path(__file__).parents[4]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import pickle

from utils import PPIPUtils


def create_fold_dict(root_path='./', itr_tag=None, n_splits=5):
    print('inside create_fold_dict() method - Start')
    
    # Retrieve the preproc_3_embed_pca_norm_pul.csv, prepared at athe previous stage of data-preparation
    embed_feat_pca_norm_pul_csv_file_loc = os.path.join(root_path, 'dataset/postproc_data/pul_result', f'{itr_tag}/preproc_pul')
    feat_df = pd.read_csv(os.path.join(embed_feat_pca_norm_pul_csv_file_loc, 'preproc_3_embed_pca_norm_pul.csv'))

    # Step 1: Create two dataframes based on 'label' column filtering
    df_labelled = feat_df[feat_df['label'].isin([0, 1])]  # Contains rows where 'label' is 0 or 1
    df_unlabelled = feat_df[feat_df['label'] == -1]       # Contains rows where 'label' is -1
    
    # Step 2: Set up the StratifiedKFold with fixed random_state
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=456)
    
    # Store the indices of k-fold splits
    fold_dict = {}
    X = df_labelled.drop(columns=['label'])  # Features of labelled data
    y = df_labelled['label']                 # Labels of labelled data
    
    # Step 3: Iterate through each fold and populate fold_dict
    folds_dir_loc = os.path.join(root_path, 'dataset/postproc_data/pul_result', f'{itr_tag}/preproc_pul/preproc_4_pul_folds')
    PPIPUtils.createFolder(folds_dir_loc)
    for i, (train_indices, test_indices) in enumerate(skf.split(X, y)):
        fold_key_test = f'fold_{i}_test'
        fold_train_train_key = f'fold_{i}_train_train'
        fold_train_validation_key = f'fold_{i}_train_validation'
        
        # Split the training data (train_indices) further into train_train and train_validation in 90:10 ratio
        train_train_indices, train_validation_indices = train_test_split(
            train_indices, test_size=0.1, stratify=y.iloc[train_indices], random_state=456
        )
        
        # Step 4: Remove all rows from train_train_indices where label is 0 and add unlabelled indices
        label_0_indices = df_labelled[df_labelled['label'] == 0].index
        train_train_indices = np.setdiff1d(train_train_indices, label_0_indices)
        train_train_indices = np.concatenate([train_train_indices, df_unlabelled.index.values])
        # ### train_train_indices = np.concatenate([train_train_indices, df_unlabelled.index.values[:2000]])
        
        # Populate fold_dict
        fold_dict[fold_key_test] = test_indices
        fold_dict[fold_train_train_key] = train_train_indices
        fold_dict[fold_train_validation_key] = train_validation_indices

        # Create the test, train_train, and train_validation dataframes using the current fold indices
        test_df = feat_df.iloc[test_indices]
        train_train_df = feat_df.iloc[train_train_indices]
        train_val_df = feat_df.iloc[train_validation_indices]

        # Save those dataframes as CSV files
        test_df.to_csv(os.path.join(folds_dir_loc, f'{fold_key_test}.csv'), index=False)
        train_train_df.to_csv(os.path.join(folds_dir_loc, f'{fold_train_train_key}.csv'), index=False)
        train_val_df.to_csv(os.path.join(folds_dir_loc, f'{fold_train_validation_key}.csv'), index=False)
    # End of for loop: for i, (train_indices, test_indices) in enumerate(skf.split(X, y)):

    # Save the fold_dict as a pickle file
    pkl_file_nm_loc = os.path.join(folds_dir_loc, f'preproc_4_pul_fold_dict.pkl')
    with open(pkl_file_nm_loc, 'wb') as f:
        pickle.dump(fold_dict, f)
    
    print('inside create_fold_dict() method - End')
    return fold_dict


def fetch_fold_data(root_path='./', itr_tag=None, fold_index=0):
    print('inside fetch_fold_data() method - Start')
    print(f'The fold index to fetch: {fold_index}')

    # Step 1: Retrieve the preproc_3_embed_pca_norm_pul.csv, prepared at athe previous stage of data-preparation
    embed_feat_pca_norm_pul_csv_file_loc = os.path.join(root_path, 'dataset/postproc_data/pul_result', f'{itr_tag}/preproc_pul')
    feat_df = pd.read_csv(os.path.join(embed_feat_pca_norm_pul_csv_file_loc, 'preproc_3_embed_pca_norm_pul.csv'))
    
    # Step 2: Load the fold_dict from the pkl file
    fold_dict = None
    folds_dir_loc = os.path.join(root_path, 'dataset/postproc_data/pul_result', f'{itr_tag}/preproc_pul/preproc_4_pul_folds')
    pkl_file_nm_loc = os.path.join(folds_dir_loc, f'preproc_4_pul_fold_dict.pkl')
    with open(pkl_file_nm_loc, 'rb') as f:
        fold_dict = pickle.load(f)
    
    # Step 3: Check if fold_index is valid
    fold_test_key = f'fold_{fold_index}_test'
    fold_train_train_key = f'fold_{fold_index}_train_train'
    fold_train_validation_key = f'fold_{fold_index}_train_validation'
    
    if (fold_test_key not in fold_dict or 
        fold_train_train_key not in fold_dict or 
        fold_train_validation_key not in fold_dict):
        raise ValueError(f"Invalid fold_index: {fold_index}. Please provide a valid fold index.")

    # Step 4: Fetch the respective indices for the provided fold_index
    test_indices = fold_dict[fold_test_key]
    train_train_indices = fold_dict[fold_train_train_key]
    train_validation_indices = fold_dict[fold_train_validation_key]
    
    # Step 5: Create the test, train_train, and train_validation dataframes using the indices
    test_df = feat_df.iloc[test_indices]
    train_train_df = feat_df.iloc[train_train_indices]
    train_val_df = feat_df.iloc[train_validation_indices]
    
    print('inside fetch_fold_data() method - End')
    return train_train_df, train_val_df, test_df


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/matpip_pd_prj')

    itr_tag = 'mcmc_fullLen_puFalse_batch5_mutPrcntLen10'
    n_splits= 3  # 3, 5, 10

    create_fold_dict(root_path=root_path, itr_tag=itr_tag, n_splits=n_splits)

    # #### fetch_fold_data(root_path=root_path, itr_tag=itr_tag, fold_index=0)
    