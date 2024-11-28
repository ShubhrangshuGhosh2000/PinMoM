import os, sys

from pathlib import Path
path_root = Path(__file__).parents[4]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
import pandas as pd
import pickle

from postproc.mat_p2ip_pd.pul.preproc_pul import pul_data_prep_4_kFold
from postproc.mat_p2ip_pd.pul.proc_pul import pul_train_2_train_evaluate


def execute_kFold(root_path='./', itr_tag=None, fold_dict_pkl_name='fold_dict.pkl', hparams={}):
    print('inside execute_kFold() method - Start')
    
    folds_dir_loc = os.path.join(root_path, 'dataset/postproc_data/pul_result', f'{itr_tag}/preproc_pul/preproc_4_pul_folds')
    # Load fold_dict from pickle file
    pkl_file_nm_loc = os.path.join(folds_dir_loc, fold_dict_pkl_name)
    fold_dict = None
    with open(pkl_file_nm_loc, 'rb') as f:
        fold_dict = pickle.load(f)

    # Find number of folds in the fold_dict by counting unique 'fold_*_test' keys
    k_fold = len([key for key in fold_dict.keys() if 'fold_' in key and '_test' in key])
    print(f'Total number of folds = {k_fold}')

    # List to store result dictionaries for each fold
    fold_result_dict_lst = []

    # Iterate through each fold from 0 to k_fold-1
    for fold_index in range(0, k_fold):
        print(f'\n ##################################################')
        print(f'############ fold_index: {fold_index} / {k_fold-1} ############')
        print(f'##################################################\n')
        # Fetch test, train_train, and train_validation dataframes
        fold_test_key = f'fold_{fold_index}_test'
        fold_train_train_key = f'fold_{fold_index}_train_train'
        fold_train_validation_key = f'fold_{fold_index}_train_validation'
        train_train_df = pd.read_csv(os.path.join(folds_dir_loc, f'{fold_train_train_key}.csv'))
        train_val_df = pd.read_csv(os.path.join(folds_dir_loc, f'{fold_train_validation_key}.csv'))
        test_df = pd.read_csv(os.path.join(folds_dir_loc, f'{fold_test_key}.csv'))

        # Call perform_pu_learning() and get the result dictionary
        fold_result_dict = pul_train_2_train_evaluate.perform_pu_learning(root_path=root_path, itr_tag=itr_tag, fold_index=fold_index
                                                                          , train_train_df=train_train_df, train_val_df=train_val_df, test_df=test_df
                                                                          , hparams=hparams)
        # Add fold_index to the result dictionary
        fold_result_dict['fold_index'] = fold_index

        # Append the result dictionary to the list
        fold_result_dict_lst.append(fold_result_dict)
    # End of for loop: for fold_index in range(k_fold):

    # Convert the result list to a DataFrame
    fold_result_df = pd.DataFrame(fold_result_dict_lst)
    # Set the column order for fold_result_df
    fold_result_df = fold_result_df[['fold_index', 'accuracy', 'precision', 'recall', 'F1_score', 'auc_score', 'conf_matrix']]

    # Calculate average of each metric across folds
    avg_row = {
        'fold_index': 'Avg',
        'accuracy': round(fold_result_df['accuracy'].mean(), 3),
        'precision': round(fold_result_df['precision'].mean(), 3),
        'recall': round(fold_result_df['recall'].mean(), 3),
        'F1_score': round(fold_result_df['F1_score'].mean(), 3),
        'auc_score': round(fold_result_df['auc_score'].mean(), 3),
        'conf_matrix': ''
    }

    # Append the average row to the DataFrame
    fold_result_df = pd.concat([fold_result_df, pd.DataFrame([avg_row])], ignore_index=True)
    print(f'fold_result_df: \n {fold_result_df}')

    # Save the DataFrame as fold_result_{k_fold}_fold.csv
    fold_result_csv_file_loc = os.path.join(root_path, 'dataset/postproc_data/pul_result', itr_tag, 'proc_pul')
    fold_result_df.to_csv(os.path.join(fold_result_csv_file_loc, f'fold_result_{k_fold}_fold.csv'), index=False)
    print('inside execute_kFold() method - End')


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/matpip_pd_prj')

    itr_tag = 'mcmc_fullLen_puFalse_batch5_mutPrcntLen10'
    fold_dict_pkl_name = 'preproc_4_pul_fold_dict.pkl'

    # #################### Hyper-parameters -Start
    hparams = {}
    hparams['num_epochs'] = 20  # Number of epochs for training and retraining
    hparams['lr'] = 0.001  # Learning rate for the optimizer
    hparams['threshold'] = 0.5  # Threshold to identify reliable negatives
    hparams['grid_size'] = 5  # KAN specific
    hparams['spline_order'] = 3  # KAN specific
    hparams['scale_noise'] = 0.1  # KAN specific
    hparams['scale_base'] = 1.0  # KAN specific
    hparams['scale_spline'] = 1.0  # KAN specific
    # #################### Hyper-parameters -End
    execute_kFold(root_path=root_path, itr_tag=itr_tag, fold_dict_pkl_name=fold_dict_pkl_name, hparams=hparams)
    