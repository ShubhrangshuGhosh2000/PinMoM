import os
import glob
import pandas as pd
import numpy as np

def calc_accept_percent(root_path, iteration_tag):
    result_dump_folder = os.path.join(root_path, 'dataset/proc_data', 'result_dump_' + iteration_tag + '/*')
    res_totItr_folders_lst = glob.glob(os.path.join(result_dump_folder), recursive=False)

    # Define the ranges
    ranges_1 = [(0.49, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    ranges_01 = [(0.90, 0.91), (0.91, 0.92), (0.92, 0.93), (0.93, 0.94), (0.94, 0.95), (0.95, 0.96), (0.96, 0.97), (0.97, 0.98), (0.98, 0.99), (0.99, 1.0)]
    complx_chain_combo_sim_res_dict_lst = []
    for complx_sim_res in res_totItr_folders_lst:
        complx_nm = complx_sim_res.split('/')[-1]
        print(f'complx_nm: {complx_nm}')
        
        # if(complx_nm in ['complex_1OPH', 'complex_1S1Q', 'complex_1GPW']):
        #     continue
        
        chain_combo_itr_folder_lst = glob.glob(os.path.join(complx_sim_res + '/*'), recursive=False)
        for indiv_chain_combo_itr_folder in chain_combo_itr_folder_lst:
            chain_combo_name = indiv_chain_combo_itr_folder.split('/')[-1]
            print(f'chain_combo_name: {chain_combo_name}')
            
            chain_combo_itr_res_folder = glob.glob(os.path.join(indiv_chain_combo_itr_folder, 'res_totItr*'), recursive=False)[0]
            misc_info_csv_path = os.path.join(chain_combo_itr_res_folder, 'misc_info.csv')
            if(not os.path.exists(misc_info_csv_path)):
                # misc_info.csv file does not exist
                continue
            misc_info_df = pd.read_csv(misc_info_csv_path)
            tot_num_itr_executed_row = misc_info_df[misc_info_df['misc_info'] == 'tot_num_itr_executed']
            tot_num_itr_executed = int(tot_num_itr_executed_row['misc_info_val'])
            print(f'tot_num_itr_executed: {tot_num_itr_executed}')
            
            batch_sim_res_csv_file_lst = glob.glob(os.path.join(chain_combo_itr_res_folder, 'batchIdx_*.csv'), recursive=False)
            tot_aaccpt_itr = 0
            entire_ppi_score_lst = []
            min_ppi = float(-np.inf); max_ppi = float(np.inf)
            prob_range_05_06_lst, prob_range_06_07_lst, prob_range_07_08_lst, prob_range_08_09_lst, prob_range_09_1_lst =  [], [], [], [], []
            prob_range_90_91_lst, prob_range_91_92_lst, prob_range_92_93_lst, prob_range_93_94_lst, prob_range_94_95_lst \
            , prob_range_95_96_lst, prob_range_96_97_lst, prob_range_97_98_lst, prob_range_98_99_lst, prob_range_99_100_lst =  [], [], [], [], [], [], [], [], [], []

            for indiv_batch_sim_res_csv_file in batch_sim_res_csv_file_lst:
                indiv_batch_sim_res_df = pd.read_csv(indiv_batch_sim_res_csv_file)
                indiv_batch_ppi_lst = indiv_batch_sim_res_df['ppi_score'].tolist() 
                entire_ppi_score_lst += indiv_batch_ppi_lst
                tot_aaccpt_itr += indiv_batch_sim_res_df.shape[0]
            # end of for loop: for indiv_batch_sim_res_csv_file in batch_sim_res_csv_file_lst:
            print(f'tot_aaccpt_itr: {tot_aaccpt_itr}')

            accept_percent = round((tot_aaccpt_itr/(tot_num_itr_executed * 1.0)) * 100, ndigits=2)
            print(f'accept_percent: {accept_percent}')

            if(len(entire_ppi_score_lst) > 0):
                min_ppi = round(min(entire_ppi_score_lst, key=lambda x:float(x)), ndigits=3)
                max_ppi = round(max(entire_ppi_score_lst, key=lambda x:float(x)), ndigits=3)
                df = pd.DataFrame({'probabilities': entire_ppi_score_lst})
                # finding the counts in the range of 0.1
                # Count probabilities within each range
                counts_1 = [df[(df['probabilities'] >= start) & (df['probabilities'] < end)].shape[0] for start, end in ranges_1[:-1]] \
                        + [df[(df['probabilities'] >= ranges_1[-1][0]) & (df['probabilities'] <= ranges_1[-1][1])].shape[0]]

                prob_range_05_06_lst.append(counts_1[0]); prob_range_06_07_lst.append(counts_1[1]); prob_range_07_08_lst.append(counts_1[2])
                prob_range_08_09_lst.append(counts_1[3]); prob_range_09_1_lst.append(counts_1[4])

                # finding the counts in the range of 0.01
                # Count probabilities within each range
                counts_01 = [df[(df['probabilities'] >= start) & (df['probabilities'] < end)].shape[0] for start, end in ranges_01[:-1]] \
                        + [df[(df['probabilities'] >= ranges_01[-1][0]) & (df['probabilities'] <= ranges_01[-1][1])].shape[0]]

                prob_range_90_91_lst.append(counts_01[0]); prob_range_91_92_lst.append(counts_01[1]); prob_range_92_93_lst.append(counts_01[2]); prob_range_93_94_lst.append(counts_01[3])
                prob_range_94_95_lst.append(counts_01[4])
                prob_range_95_96_lst.append(counts_01[5]); prob_range_96_97_lst.append(counts_01[6]); prob_range_97_98_lst.append(counts_01[7]); prob_range_98_99_lst.append(counts_01[8])
                prob_range_99_100_lst.append(counts_01[9])

            # end of if block: if(len(entire_ppi_score_lst) > 0):
            print(f'min_ppi: {min_ppi}  :: max_ppi: {max_ppi}')

            complx_chain_combo_sim_res_dict = {}
            complx_chain_combo_sim_res_dict['complx_nm'] = complx_nm
            complx_chain_combo_sim_res_dict['chain_combo_nm'] = chain_combo_name
            complx_chain_combo_sim_res_dict['tot_num_itr_executed'] = tot_num_itr_executed
            complx_chain_combo_sim_res_dict['tot_aaccpt_itr'] = tot_aaccpt_itr
            complx_chain_combo_sim_res_dict['accept_percent'] = accept_percent
            complx_chain_combo_sim_res_dict['min_ppi'] = min_ppi
            complx_chain_combo_sim_res_dict['max_ppi'] = max_ppi
            complx_chain_combo_sim_res_dict['prob_0.5_0.6'] = prob_range_05_06_lst[0]
            complx_chain_combo_sim_res_dict['prob_0.6_0.7'] = prob_range_06_07_lst[0]
            complx_chain_combo_sim_res_dict['prob_0.7_0.8'] = prob_range_07_08_lst[0]
            complx_chain_combo_sim_res_dict['prob_0.8_0.9'] = prob_range_08_09_lst[0]
            complx_chain_combo_sim_res_dict['prob_0.9_1.0'] = prob_range_09_1_lst[0]

            complx_chain_combo_sim_res_dict['prob_0.90_0.91'] = prob_range_90_91_lst[0]
            complx_chain_combo_sim_res_dict['prob_0.91_0.92'] = prob_range_91_92_lst[0]
            complx_chain_combo_sim_res_dict['prob_0.92_0.93'] = prob_range_92_93_lst[0]
            complx_chain_combo_sim_res_dict['prob_0.93_0.94'] = prob_range_93_94_lst[0]
            complx_chain_combo_sim_res_dict['prob_0.94_0.95'] = prob_range_94_95_lst[0]
            complx_chain_combo_sim_res_dict['prob_0.95_0.96'] = prob_range_95_96_lst[0]
            complx_chain_combo_sim_res_dict['prob_0.96_0.97'] = prob_range_96_97_lst[0]
            complx_chain_combo_sim_res_dict['prob_0.97_0.98'] = prob_range_97_98_lst[0]
            complx_chain_combo_sim_res_dict['prob_0.98_0.99'] = prob_range_98_99_lst[0]
            complx_chain_combo_sim_res_dict['prob_0.99_1.0'] = prob_range_99_100_lst[0]
            complx_chain_combo_sim_res_dict_lst.append(complx_chain_combo_sim_res_dict)
        # end of for loop: for indiv_chain_combo_itr_folder in chain_combo_itr_folder_lst:
    # end of for loop: for complx_sim_res in res_totItr_folders_lst:
    complx_chain_combo_sim_res_df = pd.DataFrame(complx_chain_combo_sim_res_dict_lst)
    complx_chain_combo_sim_res_csv_path = os.path.join(root_path, 'dataset/proc_data', 'result_dump_' + iteration_tag, 'percent_sim_itr_accept.csv')
    complx_chain_combo_sim_res_df.to_csv(complx_chain_combo_sim_res_csv_path, index=False)


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/matpip_pd_prj') 

    # iteration_tag_lst = ['fullLen_puTrue_thorough', 'fullLen_puTrue_fast', 'intrfc_puFalse_batch2_thorough', 'intrfc_puFalse_batch2_fast', 'fullLen_puFalse_batch5_thorough', 'fullLen_puFalse_batch5_fast']
    iteration_tag_lst = ['rl_fullLen_puFalse_batch5_mutPrcntLen10']
    for iteration_tag in iteration_tag_lst:
        calc_accept_percent(root_path, iteration_tag)

