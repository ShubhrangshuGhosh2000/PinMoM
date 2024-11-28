import sys, os
from pathlib import Path
path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

import pandas as pd


def calc_percent_seq_similarity_score(root_path='./', itr_tag=None):
    print('inside calc_percent_seq_similarity_score() method - Start')
    postproc_result_dir = os.path.join(root_path, 'dataset/postproc_data/result_dump', itr_tag)
    overall_accept_complx_struct_comp_res_postproc_5_part2_csv_path = os.path.join(postproc_result_dir, 'overall_accept_complx_struct_comp_res_postproc_5_part2.csv')
    overall_accept_complx_struct_comp_res_df = pd.read_csv(overall_accept_complx_struct_comp_res_postproc_5_part2_csv_path)

    # Iterate through overall_accept_complx_struct_comp_res_df and carry out PSI-Blast
    print(f"\nIterate through overall_accept_complx_struct_comp_res_df and carry out PSI-Blast).")
    for index, row in overall_accept_complx_struct_comp_res_df.iterrows():
        print(f'\n\n {index + 1} / {overall_accept_complx_struct_comp_res_df.shape[0]} :: id: {row['id']}\n\n')
        # TODO



    # End of for loop: for index, row in overall_accept_complx_struct_comp_res_df.iterrows():


    print('inside calc_percent_seq_similarity_score() method - End')



if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/matpip_pd_prj')

    itr_tag = 'mcmc_fullLen_puFalse_batch5_mutPrcntLen10'
    # itr_tag = 'fullLen_puFalse_batch5_thorough'

    # Prepare overall analysis result for the MD-Simulation
    calc_percent_seq_similarity_scor(root_path=root_path, itr_tag=itr_tag)

