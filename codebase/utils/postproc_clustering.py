import sys, os

from pathlib import Path
path_root = Path(__file__).parents[1]  
sys.path.insert(0, str(path_root))

from utils import dl_reproducible_result_util
from Bio.Align import substitution_matrices
from functools import partial
import hdbscan
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm


def perform_clustering(**kwargs):
    postproc_result_dir = kwargs.get('postproc_result_dir'); simulation_exec_mode = kwargs.get('simulation_exec_mode')
    dim_prot_complx_nm = kwargs.get('dim_prot_complx_nm'); curnt_chain_combo = kwargs.get('curnt_chain_combo')
    max_num_of_seq_for_af2_chain_struct_pred = kwargs.get('max_num_of_seq_for_af2_chain_struct_pred')
    psi_blast_thresholded_df = kwargs.get('psi_blast_thresholded_df')

    # At first, prepare the dataset based on the BLOSUM62 matrix.
    distances_df = prepare_dataset_for_clustering(**kwargs)
    distance_matrix = distances_df.drop('ref_seq_id', axis=1).values
    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=5)
    clusterer.fit(distance_matrix)
    labels = clusterer.labels_
    probabilities = clusterer.probabilities_
    distances_df['cluster'] = labels
    distances_df['cluster_probability'] = probabilities

    retrieved_seq_df = retrieve_seq(distances_df, max_num_of_seq_for_af2_chain_struct_pred)
    curnt_chain_combo_postproc_result_dir = os.path.join(postproc_result_dir, f'complex_{dim_prot_complx_nm}', f'{curnt_chain_combo}')
    retrieved_seq_df.to_csv(os.path.join(curnt_chain_combo_postproc_result_dir, 'retrieved_seq_a4_clustering.csv'), index=False)

    filtered_psi_blast_thresholded_df = None
    if(simulation_exec_mode == 'remc'):
        extracted_seq_id_dict_lst = []
        for _, row in retrieved_seq_df.iterrows():
            extracted_seq_id_dict = {}
            seq_id = row['Sequence_Id']  
            seq_id_splitted_lst = seq_id.split('_')
            extracted_seq_id_dict['cmplx'] = seq_id_splitted_lst[1]
            extracted_seq_id_dict['wrapper_step_idx'] = int(seq_id_splitted_lst[9])
            extracted_seq_id_dict['replica_index'] = int(seq_id_splitted_lst[12])
            extracted_seq_id_dict['batch_idx'] = int(seq_id_splitted_lst[15])
            extracted_seq_id_dict['simuln_idx'] = int(seq_id_splitted_lst[-1])
            extracted_seq_id_dict_lst.append(extracted_seq_id_dict)
        extracted_seq_id_df = pd.DataFrame(extracted_seq_id_dict_lst)
        extracted_seq_id_df = extracted_seq_id_df.drop('cmplx', axis=1)  
        filtered_psi_blast_thresholded_df = pd.merge(psi_blast_thresholded_df, extracted_seq_id_df, on=['wrapper_step_idx', 'replica_index', 'batch_idx', 'simuln_idx'])
    else:
        extracted_seq_id_dict_lst = []
        for _, row in retrieved_seq_df.iterrows():
            extracted_seq_id_dict = {}
            seq_id = row['Sequence_Id']  
            seq_id_splitted_lst = seq_id.split('_')
            extracted_seq_id_dict['cmplx'] = seq_id_splitted_lst[1]
            extracted_seq_id_dict['batch_idx'] = int(seq_id_splitted_lst[8])
            extracted_seq_id_dict['simuln_idx'] = int(seq_id_splitted_lst[-1])
            extracted_seq_id_dict_lst.append(extracted_seq_id_dict)
        extracted_seq_id_df = pd.DataFrame(extracted_seq_id_dict_lst)
        extracted_seq_id_df = extracted_seq_id_df.drop('cmplx', axis=1) 
        filtered_psi_blast_thresholded_df = pd.merge(psi_blast_thresholded_df, extracted_seq_id_df, on=['batch_idx', 'simuln_idx'])
    filtered_psi_blast_thresholded_df = filtered_psi_blast_thresholded_df.reset_index(drop=True)

    return filtered_psi_blast_thresholded_df


# Function to retrieve sequences based on the criteria
def retrieve_seq(distances_df, n_sequences):
    distances_df = distances_df[distances_df['cluster'] != -1]
    distances_df = distances_df.reset_index(drop=True)
    cluster_sizes_series = distances_df['cluster'].value_counts().sort_index()
    total_sequences = min(n_sequences, len(distances_df))
    seq_to_retrieve_ser = (cluster_sizes_series / cluster_sizes_series.sum() * total_sequences).round().astype(int)
    seq_to_retrieve_ser = seq_to_retrieve_ser.apply(lambda x: max(x, 1))
    seq_to_retrieve_dict = seq_to_retrieve_ser.to_dict()
    
    results = []
    for cluster_id, cluster_df in distances_df.groupby('cluster'):
        cluster_df = cluster_df.sort_values(by='cluster_probability', ascending=False)
        to_retrieve = seq_to_retrieve_dict.get(cluster_id, 0)
        retrieved = cluster_df.head(to_retrieve)
        for _, row in retrieved.iterrows():
            results.append((cluster_id, row['ref_seq_id'], row['cluster_probability']))

    retrieved_seq_df = pd.DataFrame(results, columns=['Cluster_Id', 'Sequence_Id', 'Clustering_Probability'])
    return retrieved_seq_df


def prepare_dataset_for_clustering(**kwargs):
    fix_mut_prot_id_tag = kwargs.get('fix_mut_prot_id_tag')
    psi_blast_thresholded_df = kwargs.get('psi_blast_thresholded_df')
    num_of_processes = mp.cpu_count() - 3
    
    with mp.Pool(processes=num_of_processes) as pool_outer:
        results_outer_loop = pool_outer.starmap(partial(prep_dist_data_for_indiv_member_v1_innerLoopItr, kwargs=kwargs)
                                                , tqdm([(ref_idx, ref_row,) for ref_idx, ref_row in psi_blast_thresholded_df.iterrows()])
                                                , chunksize=1)
    result_lst = []; error_lst = []
    for res, err in results_outer_loop:
        if err:
            error_lst.append(err)
        else:
            result_lst.append(res)
    
    if error_lst:
        raise Exception(f"{fix_mut_prot_id_tag} Errors occurred during processing: {error_lst}")
    distances_df = pd.DataFrame(result_lst)
    return distances_df


# Function to calculate distances for a given reference sequence
def prep_dist_data_for_indiv_member_v1_innerLoopItr(ref_idx, ref_row, kwargs=None, progress_counter=None):
    simulation_exec_mode = kwargs.get('simulation_exec_mode'); fix_mut_prot_id_tag = kwargs.get('fix_mut_prot_id_tag')
    dim_prot_complx_nm = kwargs.get('dim_prot_complx_nm'); curnt_chain_combo = kwargs.get('curnt_chain_combo')
    psi_blast_thresholded_df = kwargs.get('psi_blast_thresholded_df')
    try:
        blosum62 = substitution_matrices.load("BLOSUM62")
        ref_seq_id = None
        if(simulation_exec_mode == 'remc'):
            ref_seq_id = f'cmplx_{dim_prot_complx_nm}_{curnt_chain_combo}_wrapper_step_idx_{ref_row["wrapper_step_idx"]}_replica_index_{ref_row["replica_index"]}_batch_idx_{ref_row["batch_idx"]}_simuln_idx_{ref_row["simuln_idx"]}'
        else:
            ref_seq_id = f'cmplx_{dim_prot_complx_nm}_{curnt_chain_combo}_batch_idx_{ref_row["batch_idx"]}_simuln_idx_{ref_row["simuln_idx"]}'
        ref_seq = ref_row['prot_seq']
        ref_dict = {'ref_seq_id': ref_seq_id}
        for other_row_idx, other_row in psi_blast_thresholded_df.iterrows():
            other_seq_id = None
            if(simulation_exec_mode == 'remc'):
                other_seq_id = f'cmplx_{dim_prot_complx_nm}_{curnt_chain_combo}_wrapper_step_idx_{other_row["wrapper_step_idx"]}_replica_index_{other_row["replica_index"]}_batch_idx_{other_row["batch_idx"]}_simuln_idx_{other_row["simuln_idx"]}'
            else:
                other_seq_id = f'cmplx_{dim_prot_complx_nm}_{curnt_chain_combo}_batch_idx_{other_row["batch_idx"]}_simuln_idx_{other_row["simuln_idx"]}'
            other_seq = other_row['prot_seq']
            edit_dist = calc_edit_dist_Blosum62(ref_seq, other_seq, blosum62=blosum62, ref_seq_id=ref_seq_id, other_seq_id=other_seq_id)
            ref_dict[other_seq_id] = edit_dist
        return ref_dict, None
    except Exception as ex:
        return None, f"Error processing {ref_seq_id}: {ex}"


def calc_edit_dist_Blosum62(seq1, seq2, blosum62=None, ref_seq_id=None, other_seq_id=None):
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length")
    similarity_score = 0
    for a, b in zip(seq1, seq2):
        if (a, b) in blosum62:
            similarity_score += blosum62[(a, b)]
        elif (b, a) in blosum62:
            similarity_score += blosum62[(b, a)]
        else:
            raise ValueError(f"!! Error!! Invalid amino acid pair ({a}, {b}) found in the sequences where \n ref_seq_id: {ref_seq_id} \n and other_seq_id: {other_seq_id}.")
    edit_distance = -similarity_score  
    return edit_distance

