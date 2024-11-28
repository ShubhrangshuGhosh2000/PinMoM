import sys, os

from pathlib import Path
path_root = Path(__file__).parents[1]  # upto 'codebase' folder
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
    postproc_psiblast_exec_path = kwargs.get('postproc_psiblast_exec_path'); postproc_psiblast_uniprot_db_path = kwargs.get('postproc_psiblast_uniprot_db_path')
    postproc_psiblast_result_dir = kwargs.get('postproc_psiblast_result_dir')
    fix_mut_prot_id_tag = kwargs.get('fix_mut_prot_id_tag'); prob_thresholded_df = kwargs.get('prob_thresholded_df')
    dim_prot_complx_nm = kwargs.get('dim_prot_complx_nm'); curnt_chain_combo = kwargs.get('curnt_chain_combo')
    psiBlast_percent_similarity_score_threshold = kwargs.get('psiBlast_percent_similarity_score_threshold')
    max_num_of_seq_below_psiBlast_sim_score_threshold = kwargs.get('max_num_of_seq_below_psiBlast_sim_score_threshold')
    max_num_of_seq_for_af2_chain_struct_pred = kwargs.get('max_num_of_seq_for_af2_chain_struct_pred')
    psi_blast_thresholded_df = kwargs.get('psi_blast_thresholded_df')

    print(f"{fix_mut_prot_id_tag} Inside perform_clustering() method -Start")
    
    # At first, prepare the dataset based on the BLOSUM62 matrix.
    distances_df = prepare_dataset_for_clustering(**kwargs)
    distance_matrix = distances_df.drop('ref_seq_id', axis=1).values

    # Create the HDBSCAN clusterer
    clusterer = hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=5)

    # Fit the clusterer to the distance matrix
    clusterer.fit(distance_matrix)

    # Extract cluster labels and probabilities
    labels = clusterer.labels_
    probabilities = clusterer.probabilities_

    # Add cluster labels and probabilities to the dataframe
    distances_df['cluster'] = labels
    distances_df['cluster_probability'] = probabilities

    # Retrieve sequences and save the result
    retrieved_seq_df = retrieve_seq(distances_df, max_num_of_seq_for_af2_chain_struct_pred)
    curnt_chain_combo_postproc_result_dir = os.path.join(postproc_result_dir, f'complex_{dim_prot_complx_nm}', f'{curnt_chain_combo}')
    retrieved_seq_df.to_csv(os.path.join(curnt_chain_combo_postproc_result_dir, 'retrieved_seq_a4_clustering.csv'), index=False)

    # Filter out psi_blast_thresholded_df entries based on retrieved_seq_df
    print(f"{fix_mut_prot_id_tag} Filtering out psi_blast_thresholded_df entries based on retrieved_seq_df")

    # Extract parts of 'Sequence_Id' column values of retrieved_seq_df into another DataFrame
    filtered_psi_blast_thresholded_df = None
    if(simulation_exec_mode == 'remc'):
        extracted_seq_id_dict_lst = []
        for _, row in retrieved_seq_df.iterrows():
            extracted_seq_id_dict = {}
            seq_id = row['Sequence_Id']  # 'cmplx_2I25_fixed_N_mut_L_wrapper_step_idx_5_replica_index_0_batch_idx_34_simuln_idx_173'
            seq_id_splitted_lst = seq_id.split('_')
            extracted_seq_id_dict['cmplx'] = seq_id_splitted_lst[1]
            extracted_seq_id_dict['wrapper_step_idx'] = int(seq_id_splitted_lst[9])
            extracted_seq_id_dict['replica_index'] = int(seq_id_splitted_lst[12])
            extracted_seq_id_dict['batch_idx'] = int(seq_id_splitted_lst[15])
            extracted_seq_id_dict['simuln_idx'] = int(seq_id_splitted_lst[-1])
            extracted_seq_id_dict_lst.append(extracted_seq_id_dict)
        # End of for loop: for _, row in retrieved_seq_df.iterrows():
        extracted_seq_id_df = pd.DataFrame(extracted_seq_id_dict_lst)
        # extracted_seq_id_df = retrieved_seq_df['Sequence_Id'].str.extract(r'cmplx_([a-zA-Z0-9]+)_wrapper_step_idx_(\d+)_replica_index_(\d+)_batch_idx_(\d+)_simuln_idx_(\d+)')
        # extracted_seq_id_df.columns = ['cmplx', 'wrapper_step_idx', 'replica_index', 'batch_idx', 'simuln_idx']
        extracted_seq_id_df = extracted_seq_id_df.drop('cmplx', axis=1)  # drop 'cmplx' column
        # Merge psi_blast_thresholded_df with the extracted_seq_id_df parts
        filtered_psi_blast_thresholded_df = pd.merge(psi_blast_thresholded_df, extracted_seq_id_df, on=['wrapper_step_idx', 'replica_index', 'batch_idx', 'simuln_idx'])
    else:
        # for mcmc
        extracted_seq_id_dict_lst = []
        for _, row in retrieved_seq_df.iterrows():
            extracted_seq_id_dict = {}
            seq_id = row['Sequence_Id']  # 'cmplx_2I25_fixed_N_mut_L_batch_idx_1137_simuln_idx_5687'
            seq_id_splitted_lst = seq_id.split('_')
            extracted_seq_id_dict['cmplx'] = seq_id_splitted_lst[1]
            extracted_seq_id_dict['batch_idx'] = int(seq_id_splitted_lst[8])
            extracted_seq_id_dict['simuln_idx'] = int(seq_id_splitted_lst[-1])
            extracted_seq_id_dict_lst.append(extracted_seq_id_dict)
        # End of for loop: for _, row in retrieved_seq_df.iterrows():
        extracted_seq_id_df = pd.DataFrame(extracted_seq_id_dict_lst)
        # extracted_seq_id_df = retrieved_seq_df['Sequence_Id'].str.extract(r'cmplx_([a-zA-Z0-9]+)_batch_idx_(\d+)_simuln_idx_(\d+)')
        # extracted_seq_id_df.columns = ['cmplx', 'batch_idx', 'simuln_idx']
        extracted_seq_id_df = extracted_seq_id_df.drop('cmplx', axis=1) # drop 'cmplx' column
        # Merge psi_blast_thresholded_df with the extracted_seq_id_df parts
        filtered_psi_blast_thresholded_df = pd.merge(psi_blast_thresholded_df, extracted_seq_id_df, on=['batch_idx', 'simuln_idx'])
    # End of if-else block: if(simulation_exec_mode == 'remc'):
    # Reset index of filtered_psi_blast_thresholded_df
    filtered_psi_blast_thresholded_df = filtered_psi_blast_thresholded_df.reset_index(drop=True)
    print(f"{fix_mut_prot_id_tag} Inside perform_clustering() method -End")

    # return filtered_psi_blast_thresholded_df
    return filtered_psi_blast_thresholded_df
# End of perform_clustering() method


# Function to retrieve sequences based on the criteria
def retrieve_seq(distances_df, n_sequences):
    # Filter out noise (label -1)
    distances_df = distances_df[distances_df['cluster'] != -1]
    distances_df = distances_df.reset_index(drop=True)
    # Calculate the number of sequences in each cluster
    cluster_sizes_series = distances_df['cluster'].value_counts().sort_index()
    # Calculate the total number of sequences to retrieve
    total_sequences = min(n_sequences, len(distances_df))
    # Calculate the proportion of sequences to retrieve from each cluster
    seq_to_retrieve_ser = (cluster_sizes_series / cluster_sizes_series.sum() * total_sequences).round().astype(int)
    # Ensure at least one sequence is retrieved from each cluster
    seq_to_retrieve_ser = seq_to_retrieve_ser.apply(lambda x: max(x, 1))
    # Get the number of sequences to retrieve per cluster
    seq_to_retrieve_dict = seq_to_retrieve_ser.to_dict()

    # Create a list to store the results
    results = []
    # Retrieve sequences based on the proportion and probability
    for cluster_id, cluster_df in distances_df.groupby('cluster'):
        cluster_df = cluster_df.sort_values(by='cluster_probability', ascending=False)
        to_retrieve = seq_to_retrieve_dict.get(cluster_id, 0)
        retrieved = cluster_df.head(to_retrieve)
        for _, row in retrieved.iterrows():
            results.append((cluster_id, row['ref_seq_id'], row['cluster_probability']))
        # End of inner for loop
    #End of outer for loop: for cluster_id, cluster_df in distances_df.groupby('cluster'):

    # Convert results to DataFrame
    retrieved_seq_df = pd.DataFrame(results, columns=['Cluster_Id', 'Sequence_Id', 'Clustering_Probability'])
    return retrieved_seq_df
# End of function: def retrieve_seq(distances_df, n_sequences):


def prepare_dataset_for_clustering(**kwargs):
    simulation_exec_mode = kwargs.get('simulation_exec_mode')
    postproc_psiblast_exec_path = kwargs.get('postproc_psiblast_exec_path'); postproc_psiblast_uniprot_db_path = kwargs.get('postproc_psiblast_uniprot_db_path')
    postproc_psiblast_result_dir = kwargs.get('postproc_psiblast_result_dir')
    fix_mut_prot_id_tag = kwargs.get('fix_mut_prot_id_tag'); prob_thresholded_df = kwargs.get('prob_thresholded_df')
    dim_prot_complx_nm = kwargs.get('dim_prot_complx_nm'); curnt_chain_combo = kwargs.get('curnt_chain_combo')
    psiBlast_percent_similarity_score_threshold = kwargs.get('psiBlast_percent_similarity_score_threshold')
    max_num_of_seq_below_psiBlast_sim_score_threshold = kwargs.get('max_num_of_seq_below_psiBlast_sim_score_threshold')
    psi_blast_thresholded_df = kwargs.get('psi_blast_thresholded_df')
    
    print(f"{fix_mut_prot_id_tag} Inside prepare_dataset_for_clustering() method -Start")
    num_of_processes = mp.cpu_count() - 3
    # Create a pool of worker processes
    with mp.Pool(processes=num_of_processes) as pool_outer:
        # Apply the prep_dist_data_for_indiv_member_v1_innerLoopItr() function to each row of the psi_blast_thresholded_df in parallel
        results_outer_loop = pool_outer.starmap(partial(prep_dist_data_for_indiv_member_v1_innerLoopItr, kwargs=kwargs)
                                                , tqdm([(ref_idx, ref_row,) for ref_idx, ref_row in psi_blast_thresholded_df.iterrows()])
                                                , chunksize=1)
    # End of with block
    
    # Collect results and errors
    result_lst = []; error_lst = []
    for res, err in results_outer_loop:
        if err:
            error_lst.append(err)
        else:
            result_lst.append(res)
    
    # Check for errors and raise an exception if any are found
    if error_lst:
        raise Exception(f"{fix_mut_prot_id_tag} Errors occurred during processing: {error_lst}")
    
    # Convert the list of dictionaries to a DataFrame
    distances_df = pd.DataFrame(result_lst)

    # Store the DataFrame as a CSV file
    # distances_df.to_csv('distances.csv', index=False)
    print(f"{fix_mut_prot_id_tag} Inside prepare_dataset_for_clustering() method -End")
    # Return the distances_df
    return distances_df
# End of prepare_dataset_for_clustering() method


# Function to calculate distances for a given reference sequence
def prep_dist_data_for_indiv_member_v1_innerLoopItr(ref_idx, ref_row, kwargs=None, progress_counter=None):
    simulation_exec_mode = kwargs.get('simulation_exec_mode'); fix_mut_prot_id_tag = kwargs.get('fix_mut_prot_id_tag')
    dim_prot_complx_nm = kwargs.get('dim_prot_complx_nm'); curnt_chain_combo = kwargs.get('curnt_chain_combo')
    psi_blast_thresholded_df = kwargs.get('psi_blast_thresholded_df')
    # print(f"{fix_mut_prot_id_tag} Clustering dataset preparation: Starting {ref_idx + 1}-th reference-member out of {len(psi_blast_thresholded_df)}")
    try:
        # Load the BLOSUM62 matrix
        blosum62 = substitution_matrices.load("BLOSUM62")
        ref_seq_id = None
        if(simulation_exec_mode == 'remc'):
            ref_seq_id = f'cmplx_{dim_prot_complx_nm}_{curnt_chain_combo}_wrapper_step_idx_{ref_row["wrapper_step_idx"]}_replica_index_{ref_row["replica_index"]}_batch_idx_{ref_row["batch_idx"]}_simuln_idx_{ref_row["simuln_idx"]}'
        else:
            # For mcmc
            ref_seq_id = f'cmplx_{dim_prot_complx_nm}_{curnt_chain_combo}_batch_idx_{ref_row["batch_idx"]}_simuln_idx_{ref_row["simuln_idx"]}'
        # end of if-else block
        # print(f"{fix_mut_prot_id_tag} ref_seq_id: {ref_seq_id}")
        ref_seq = ref_row['prot_seq']
        
        # Initialize a dictionary for the current reference sequence
        ref_dict = {'ref_seq_id': ref_seq_id}
        
        # Calculate distances from the current reference sequence to all other sequences
        for other_row_idx, other_row in psi_blast_thresholded_df.iterrows():
            # print(f"{fix_mut_prot_id_tag} Clustering dataset preparation: Inner loop: {other_row_idx + 1} out of {len(psi_blast_thresholded_df)}")
            other_seq_id = None
            if(simulation_exec_mode == 'remc'):
                other_seq_id = f'cmplx_{dim_prot_complx_nm}_{curnt_chain_combo}_wrapper_step_idx_{other_row["wrapper_step_idx"]}_replica_index_{other_row["replica_index"]}_batch_idx_{other_row["batch_idx"]}_simuln_idx_{other_row["simuln_idx"]}'
            else:
                # For mcmc
                other_seq_id = f'cmplx_{dim_prot_complx_nm}_{curnt_chain_combo}_batch_idx_{other_row["batch_idx"]}_simuln_idx_{other_row["simuln_idx"]}'
            # end of if-else block
            other_seq = other_row['prot_seq']
            # End of if-else block: if(simulation_exec_mode == 'remc'):

            # Calculate Edit-distance between ref_seq and other_seq based on Blosum62 matrix
            edit_dist = calc_edit_dist_Blosum62(ref_seq, other_seq, blosum62=blosum62, ref_seq_id=ref_seq_id, other_seq_id=other_seq_id)
            ref_dict[other_seq_id] = edit_dist
        # End of for loop: for _, other_row in psi_blast_thresholded_df.iterrows():
        return ref_dict, None
    except Exception as ex:
        print(f"*** {fix_mut_prot_id_tag} Error processing {ref_seq_id}: {ex}")
        return None, f"Error processing {ref_seq_id}: {ex}"


def calc_edit_dist_Blosum62(seq1, seq2, blosum62=None, ref_seq_id=None, other_seq_id=None):
    """
    Calculate the edit distance between two protein sequences based on the BLOSUM62 matrix.
    The distance is defined as the negative similarity score.

    Args:
        seq1 (str): First protein sequence.
        seq2 (str): Second protein sequence.
        blosum62 (dict): BLOSUM62 substitution matrix.
        
    Returns:
        int: The calculated edit distance.
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length")
    
    # First calculate the similarity score based on BLOSUM62 matrix.
    # The BLOSUM62 matrix contains log-odds scores, which are positive for more likely substitutions (indicating higher similarity)
    # and negative for less likely substitutions (indicating lower similarity).
    # By summing these scores for corresponding amino acid pairs in two sequences, 
    # we get a value that represents the overall similarity of the sequences.
    similarity_score = 0
    
    for a, b in zip(seq1, seq2):
        if (a, b) in blosum62:
            similarity_score += blosum62[(a, b)]
        elif (b, a) in blosum62:
            similarity_score += blosum62[(b, a)]
        else:
            raise ValueError(f"!! Error!! Invalid amino acid pair ({a}, {b}) found in the sequences where \n ref_seq_id: {ref_seq_id} \n and other_seq_id: {other_seq_id}.")
    # End of for loop

    # Convert similarity to distance
    # One way to convert the BLOSUM62 similarity score into a distance score is by taking the negative of the similarity score.
    edit_distance = -similarity_score  
    return edit_distance
# End of calc_edit_dist_Blosum62() method
