import os, sys

from pathlib import Path
path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio import PDB
from Bio.PDB import PDBParser
from utils import PPIPUtils


def run_postproc_5_part2_mut_complx_struct_comparison(**kwargs):
    """Postprocessing stage where mutated complex structure (as predicted by AlphaFold2) is compared with that of the original dimeric complex.

    This method is called from a triggering method like mat_p2ip_pd_postproc_trigger.trigger_pd_postproc().
    """
    print('inside run_postproc_5_part2_mut_complx_struct_comparison() method - Start')
    print('####################################')
    # Iterate over kwargs and raise ValueError if any of the input arguments (except a few) is None. Also print each keyword argument name and respective value.
    for arg_name, arg_value in kwargs.items():
        if(arg_value is None):
            raise ValueError(f"Argument '{arg_name}' must be provided with a value.")
        print(f"'{arg_name}': {arg_value}")
    # end of for loop: for arg_name, arg_value in kwargs.items():

    # retrieve all the keyword arguments
    root_path = kwargs.get('root_path'); dim_prot_complx_nm = kwargs.get('dim_prot_complx_nm')
    postproc_result_dir = kwargs.get('postproc_result_dir'); pdb_file_location = kwargs.get('pdb_file_location')
    af2_use_amber = kwargs.get('af2_use_amber'); bb_rmsd_threshold_complx_struct_comp = kwargs.get('bb_rmsd_threshold_complx_struct_comp')
    inp_dir_mut_complx_struct_pred_af2 = kwargs.get('inp_dir_mut_complx_struct_pred_af2')
    out_dir_mut_complx_struct_pred_af2 = kwargs.get('out_dir_mut_complx_struct_pred_af2')
    print('####################################')

    # create the postprocess result directory for the given dimeric protein complex
    print(f'\nPostproc_5_part2: Creating the postprocess result directory for the given dimeric protein complex: {dim_prot_complx_nm}')
    curnt_dim_complx_postproc_result_dir = os.path.join(postproc_result_dir, f'complex_{dim_prot_complx_nm}')
    PPIPUtils.createFolder(curnt_dim_complx_postproc_result_dir, recreate_if_exists=False)
    
    dim_prot_complx_tag = f' Postproc_5_part2: complex_{dim_prot_complx_nm}:: '
    print(f'\ndim_prot_complx_tag: {dim_prot_complx_tag}')
    # creating few lists to track the mutated sequence structure comparison
    mut_complx_struct_comp_postproc_5_part2_info_lst, mut_complx_struct_comp_postproc_5_part2_val_lst = [], []

    # Search in all the CSV files with the specific name pattern 'af2inp_mut_cmpl_{dim_prot_complx_nm}_*.csv' in the inp_dir_mut_complx_struct_pred_af2
    af2_specific_name_pattern = os.path.join(inp_dir_mut_complx_struct_pred_af2, f'af2inp_mut_cmpl_{dim_prot_complx_nm}_*.csv')
    mut_complx_inp_csv_files_lst_for_chain_combo = glob.glob(af2_specific_name_pattern, recursive=False)
    # Find the number of resulting csv files
    num_mut_complx_inp_csv_files = len(mut_complx_inp_csv_files_lst_for_chain_combo)
    mut_complx_struct_comp_postproc_5_part2_info_lst.append('num_mut_complx_inp_csv_files'); mut_complx_struct_comp_postproc_5_part2_val_lst.append(f'{num_mut_complx_inp_csv_files}')
    # If the number of resulting csv files is zero, return
    if(num_mut_complx_inp_csv_files == 0):
        print(f"\n!!! {dim_prot_complx_tag} Found no CSV files with the specific name pattern 'af2inp_mut_cmpl_{dim_prot_complx_nm}_*.csv' in the {inp_dir_mut_complx_struct_pred_af2}\
              \nSo, NOT proceeding further for the current dimeric complex: {dim_prot_complx_nm}\n")
        mut_complx_struct_comp_postproc_5_part2_info_lst.append('status'); mut_complx_struct_comp_postproc_5_part2_val_lst.append('ERROR')
        # save tracking info records and continue with the next dimeric complex
        mut_complx_struct_comp_postproc_5_part2_info_df = pd.DataFrame({'misc_info': mut_complx_struct_comp_postproc_5_part2_info_lst, 'misc_info_val': mut_complx_struct_comp_postproc_5_part2_val_lst})
        mut_complx_struct_comp_postproc_5_part2_info_df.to_csv(os.path.join(curnt_dim_complx_postproc_result_dir, 'mut_complx_struct_comp_postproc_5_part2.csv'), index=False)
        return  # return back
    # end of if block: if(num_mut_complx_inp_csv_files == 0):

    # Load the original dimeric protein structure from the dimeric protein complex PDB file
    print(f'{dim_prot_complx_tag} Loading original dimeric protein ({dim_prot_complx_nm}) structure from the dimeric protein complex PDB file ({dim_prot_complx_nm}.pdb)')
    orig_prot_complx_pdb_file_path = os.path.join(pdb_file_location, f"{dim_prot_complx_nm}.pdb")
    orig_prot_complx_parser = PDBParser(QUIET=True)
    orig_prot_complx_struct = orig_prot_complx_parser.get_structure("orig_prot_complx", orig_prot_complx_pdb_file_path)
    # find the chain-name pair for the dimer
    orig_chain_combo_nm = mut_complx_inp_csv_files_lst_for_chain_combo[0].split('/')[-1].replace(f'af2inp_mut_cmpl_{dim_prot_complx_nm}_', '').replace('.csv', '')
    print(f'\n {dim_prot_complx_tag} orig_chain_combo_nm: {orig_chain_combo_nm}')  # Sample chain_combo_nm: "fpi_L_mpi_N"
    orig_chain_1_nm = orig_chain_combo_nm.split('_')[1]
    orig_chain_2_nm = orig_chain_combo_nm.split('_')[-1]
    print(f'{dim_prot_complx_tag} orig_chain_1_nm: {orig_chain_1_nm} :: orig_chain_2_nm: {orig_chain_2_nm}')

    orig_chain_1_seq = orig_prot_complx_struct[0][orig_chain_1_nm]
    orig_chain_2_seq = orig_prot_complx_struct[0][orig_chain_2_nm]

    bb_atoms_coord_orig_chain_1, bb_trace_atoms_coord_orig_chain_1 = [], []
    for residue in orig_chain_1_seq:
        # filter out non-standard amino acids
        if PDB.is_aa(residue, standard=True):
            # Iterate over each atom of the standard amino-acid
            for atom in residue.get_atoms():
                # Select only the backbone atoms ("N", "CA", "C") from the orig_chain_1_seq
                if(atom.name in ["N", "CA", "C"]):
                    bb_atoms_coord_orig_chain_1.append(atom.get_coord())
                # Again, select only the backbone-trace atoms ("CA") from the orig_chain_1_seq
                if(atom.name in ["CA"]):
                    bb_trace_atoms_coord_orig_chain_1.append(atom.get_coord())
            # end of for loop: for atom in residue.get_atoms():
        # End of if block: if PDB.is_aa(residue, standard=True):'
    # End of for loop: for residue in orig_chain_1_seq:

    bb_atoms_coord_orig_chain_2, bb_trace_atoms_coord_orig_chain_2 = [], []
    for residue in orig_chain_2_seq:
        # filter out non-standard amino acids
        if PDB.is_aa(residue, standard=True):
            # Iterate over each atom of the standard amino-acid
            for atom in residue.get_atoms():
                # Select only the backbone atoms ("N", "CA", "C") from the orig_chain_2_seq
                if(atom.name in ["N", "CA", "C"]):
                    bb_atoms_coord_orig_chain_2.append(atom.get_coord())
                # Again, select only the backbone-trace atoms ("CA") from the orig_chain_2_seq
                if(atom.name in ["CA"]):
                    bb_trace_atoms_coord_orig_chain_2.append(atom.get_coord())
            # end of for loop: for atom in residue.get_atoms():
        # End of if block: if PDB.is_aa(residue, standard=True):'
    # End of for loop: for residue in orig_chain_2_seq:

    bb_atoms_coord_orig_chain_1_2 = np.vstack(bb_atoms_coord_orig_chain_1 + bb_atoms_coord_orig_chain_2)  # converting to 2d array from the list of 1d arrays
    bb_trace_atoms_coord_orig_chain_1_2 = np.vstack(bb_trace_atoms_coord_orig_chain_1 + bb_trace_atoms_coord_orig_chain_2)  # converting to 2d array from the list of 1d arrays

    print(f"\n{dim_prot_complx_tag} There are {num_mut_complx_inp_csv_files} CSV file(s) with the specific name pattern 'af2inp_mut_cmpl_{dim_prot_complx_nm}_*.csv' in the {inp_dir_mut_complx_struct_pred_af2} \
           \n Parsing every CSV file with the specific name pattern 'af2inp_mut_cmpl_{dim_prot_complx_nm}_*.csv' in a loop ...")
    for indiv_mut_complx_inp_csv_file_name in mut_complx_inp_csv_files_lst_for_chain_combo:
        print(f'\n ###### {dim_prot_complx_tag} indiv_mut_complx_inp_csv_file_name: {indiv_mut_complx_inp_csv_file_name}')
        mut_chain_combo_nm = indiv_mut_complx_inp_csv_file_name.split('/')[-1].replace(f'af2inp_mut_cmpl_{dim_prot_complx_nm}_', '').replace('.csv', '')
        print(f'\n {dim_prot_complx_tag} mut_chain_combo_nm: {mut_chain_combo_nm}')  # Sample mut_chain_combo_nm: "fpi_L_mpi_N"
        # read the csv file (indiv_mut_complx_inp_csv_file_name) into a df
        indiv_mut_complx_inp_df = pd.read_csv(indiv_mut_complx_inp_csv_file_name)
        # Declare a list of  dictionaries which will be used to store the structure comparison result and later be used to 
        # create a CSV file
        con_complx_struct_comp_res_lst = []
        print(f'\n {dim_prot_complx_tag} Iterating {indiv_mut_complx_inp_csv_file_name} row-wise...')
        # iterate the df row-wise in an inner loop
        for index, row in indiv_mut_complx_inp_df.iterrows():
            id = row['id']; seq = row['sequence']
            print(f'\n {dim_prot_complx_tag} iteration index: {index} :: id: {id} :: seq: {seq}')
            id_parts_lst = id.split('_')  # sample id: cmplx_2I25_fpi_L_mpi_N_batch_idx_2507_simuln_idx_7523
            # dictioanry to store the structure comparison result for this iteration
            indiv_complx_struct_comp_res_dict = {'id': id, 'cmplx': id_parts_lst[1], 'fpi': id_parts_lst[3], 'mpi': id_parts_lst[5], 'batch_idx': int(id_parts_lst[8])
                                          , 'simuln_idx': int(id_parts_lst[11]), 'sequence': seq}
            # Extract the mutating part of the sequence
            mpi_seq = indiv_complx_struct_comp_res_dict['sequence'].split(':')[1]
            # retrieve and add ppi_score in indiv_complx_struct_comp_res_dict from specific prob_thresholded_df.csv
            prob_thresholded_df_loc = os.path.join(postproc_result_dir, f'complex_{indiv_complx_struct_comp_res_dict["cmplx"]}'
                                                   , f'fixed_{indiv_complx_struct_comp_res_dict["fpi"]}_mut_{indiv_complx_struct_comp_res_dict["mpi"]}'
                                                   , 'prob_thresholded_df.csv')
            prob_thresholded_df = pd.read_csv(prob_thresholded_df_loc)
            spec_row = prob_thresholded_df.loc[(prob_thresholded_df['batch_idx'] == indiv_complx_struct_comp_res_dict['batch_idx'])
                                                & (prob_thresholded_df['simuln_idx'] == indiv_complx_struct_comp_res_dict['simuln_idx'])
                                                & (prob_thresholded_df['prot_seq'] == mpi_seq)]
            ppi_score = float(spec_row['ppi_score'])
            indiv_complx_struct_comp_res_dict['ppi_score'] = ppi_score

            # Create the pdb file location of the best ranked mutated complex structure (as predicted by AlphaFold2)
            # sample pdb file name: cmplx_2I25_fpi_L_mpi_N_batch_idx_999_simuln_idx_2997_relaxed_rank_001_alphafold2_multimer_v3_model_4_seed_000.pdb
            relaxed_unrelaxed = 'unrelaxed'
            if(af2_use_amber):
                relaxed_unrelaxed = 'relaxed'
            best_rank_mut_complx_struct_pdb_fl_nm_pattern = os.path.join(out_dir_mut_complx_struct_pred_af2, dim_prot_complx_nm
                                                              , mut_chain_combo_nm, f'{id}_{relaxed_unrelaxed}_rank_001_alphafold2_multimer_v3_model_*.pdb')
            
            best_rank_pdb_fl_nm_lst = glob.glob(best_rank_mut_complx_struct_pdb_fl_nm_pattern, recursive=False)
            best_rank_mut_complx_struct_pdb_fl_nm = best_rank_pdb_fl_nm_lst[0]
            print(f'\n### {dim_prot_complx_tag} best_rank_mut_complx_struct_pdb_fl_nm: {best_rank_mut_complx_struct_pdb_fl_nm}')
            # Check whether the generated pdb file is empty. If yes, then skip this iteration
            if(os.path.getsize(best_rank_mut_complx_struct_pdb_fl_nm) == 0):
                print(f'\n***{dim_prot_complx_tag} best_rank_mut_complx_struct_pdb_fl_nm : ({best_rank_mut_complx_struct_pdb_fl_nm}) is empty. Hence, skipping this iteration.')
                continue
            # parse the pdb file
            mut_complx_parser = PDBParser(QUIET=True)
            mut_complx_struct = mut_complx_parser.get_structure("mut_complx", best_rank_mut_complx_struct_pdb_fl_nm)
            mut_complx_chain_1_seq = mut_complx_struct[0].get_list()[0]
            mut_complx_chain_2_seq = mut_complx_struct[0].get_list()[1]

            bb_atoms_coord_mut_complx_chain_1, bb_trace_atoms_coord_mut_complx_chain_1 = [], []
            for residue in mut_complx_chain_1_seq:
                # filter out non-standard amino acids
                if PDB.is_aa(residue, standard=True):
                    # Iterate over each atom of the standard amino-acid
                    for atom in residue.get_atoms():
                        # Select only the backbone atoms ("N", "CA", "C") from the mut_complx_chain_1_seq
                        if(atom.name in ["N", "CA", "C"]):
                            bb_atoms_coord_mut_complx_chain_1.append(atom.get_coord())
                        # Again, select only the backbone-trace atoms ("CA") from the mut_complx_chain_1_seq
                        if(atom.name in ["CA"]):
                            bb_trace_atoms_coord_mut_complx_chain_1.append(atom.get_coord())
                    # end of for loop: for atom in residue.get_atoms():
                # End of if block: if PDB.is_aa(residue, standard=True):'
            # End of for loop: for residue in mut_complx_chain_1_seq:

            bb_atoms_coord_mut_complx_chain_2, bb_trace_atoms_coord_mut_complx_chain_2 = [], []
            for residue in mut_complx_chain_2_seq:
                # filter out non-standard amino acids
                if PDB.is_aa(residue, standard=True):
                    # Iterate over each atom of the standard amino-acid
                    for atom in residue.get_atoms():
                        # Select only the backbone atoms ("N", "CA", "C") from the mut_complx_chain_2_seq
                        if(atom.name in ["N", "CA", "C"]):
                            bb_atoms_coord_mut_complx_chain_2.append(atom.get_coord())
                        # Again, select only the backbone-trace atoms ("CA") from the mut_complx_chain_2_seq
                        if(atom.name in ["CA"]):
                            bb_trace_atoms_coord_mut_complx_chain_2.append(atom.get_coord())
                    # end of for loop: for atom in residue.get_atoms():
                # End of if block: if PDB.is_aa(residue, standard=True):'
            # End of for loop: for residue in mut_complx_chain_2_seq:

            bb_atoms_coord_mut_complx_chain_1_2 = np.vstack(bb_atoms_coord_mut_complx_chain_1 + bb_atoms_coord_mut_complx_chain_2)  # converting to 2d array from the list of 1d arrays
            bb_trace_atoms_coord_mut_complx_chain_1_2 = np.vstack(bb_trace_atoms_coord_mut_complx_chain_1 + bb_trace_atoms_coord_mut_complx_chain_2)  # converting to 2d array from the list of 1d arrays

            # Calculate Backbone RMSD and Backbone-trace RMSD. Also the repective initial RMSD values before superimposition.
            print(f'\n{dim_prot_complx_tag} Calculating Backbone RMSD and Backbone-trace RMSD. Also the repective initial RMSD values before superimposition.')
            print(f'{dim_prot_complx_tag} bb_atoms_coord_orig_chain_1_2.shape = {bb_atoms_coord_orig_chain_1_2.shape} :: bb_atoms_coord_mut_complx_chain_1_2.shape = {bb_atoms_coord_mut_complx_chain_1_2.shape}')
            print(f'{dim_prot_complx_tag} bb_trace_atoms_coord_orig_chain_1_2.shape = {bb_trace_atoms_coord_orig_chain_1_2.shape} :: bb_trace_atoms_coord_mut_complx_chain_1_2.shape = {bb_trace_atoms_coord_mut_complx_chain_1_2.shape}')
            bb_rmsd_superimp, bb_rmsd_b4_superimp = calculate_rmsd_by_SVDSuperimposer(bb_atoms_coord_orig_chain_1_2, bb_atoms_coord_mut_complx_chain_1_2)
            # bb_trace_rmsd_superimp, bb_trace_rmsd_b4_superimp = calculate_rmsd_by_SVDSuperimposer(bb_trace_atoms_coord_orig_chain_1_2, bb_trace_atoms_coord_mut_complx_chain_1_2)
            # store the RMSD values in indiv_complx_struct_comp_res_dict
            indiv_complx_struct_comp_res_dict['bb_rmsd_superimp'] = bb_rmsd_superimp
            # indiv_complx_struct_comp_res_dict['bb_rmsd_b4_superimp'] = bb_rmsd_b4_superimp 
            # indiv_complx_struct_comp_res_dict['bb_trace_rmsd_superimp'] = bb_trace_rmsd_superimp
            # indiv_complx_struct_comp_res_dict['bb_trace_rmsd_b4_superimp'] = bb_trace_rmsd_b4_superimp
            # Append the indiv_complx_struct_comp_res_dict in the con_complx_struct_comp_res_lst
            con_complx_struct_comp_res_lst.append(indiv_complx_struct_comp_res_dict)
        # end of inner for loop: for index, row in indiv_mut_complx_inp_df.iterrows():
        
        # create a dataframe and sort it in the ascending order of the Backbone-RMSD values
        con_complx_struct_comp_res_df = pd.DataFrame(con_complx_struct_comp_res_lst)
        con_complx_struct_comp_res_df = con_complx_struct_comp_res_df.sort_values(by=['bb_rmsd_superimp'], ascending=True)
        con_complx_struct_comp_res_df = con_complx_struct_comp_res_df.reset_index(drop=True)
        # save the dataframe as a CSV file
        # mod_chain_combo_nm = mut_chain_combo_nm.replace('fpi', 'fixed').replace('mpi', 'mut')
        # con_complx_struct_comp_res_csv_fl_nm = os.path.join(curnt_dim_complx_postproc_result_dir, mod_chain_combo_nm, f'complx_struct_comp_res_{mut_chain_combo_nm}_postproc_5_part2.csv')
        # con_complx_struct_comp_res_df.to_csv(con_complx_struct_comp_res_csv_fl_nm, index=False)
        # check and update 'overall_complx_struct_comp_res_postproc_5_part2.csv'
        print(f"\n{dim_prot_complx_tag} Check and update 'overall_complx_struct_comp_res_postproc_5_part2.csv'")
        overall_complx_struct_comp_res_df = update_overall_complx_struct_comp_res(postproc_result_dir, con_complx_struct_comp_res_df)
    # end of for loop: for indiv_mut_complx_inp_csv_file_name in mut_complx_inp_csv_files_lst_for_chain_combo:
    
    # Filter overall_complx_struct_comp_res_df for bb_rmsd_superimp <= bb_rmsd_threshold_complx_struct_comp (e.g. 10 Angstrom) and save it 
    overall_accept_complx_struct_comp_res_df = overall_complx_struct_comp_res_df[overall_complx_struct_comp_res_df['bb_rmsd_superimp'] <= bb_rmsd_threshold_complx_struct_comp]
    overall_accept_complx_struct_comp_res_df = overall_accept_complx_struct_comp_res_df.reset_index(drop=True)
    overall_accept_complx_struct_comp_res_csv = os.path.join(postproc_result_dir, 'overall_accept_complx_struct_comp_res_postproc_5_part2.csv')
    overall_accept_complx_struct_comp_res_df.to_csv(overall_accept_complx_struct_comp_res_csv)

    # save tracking info records
    mut_complx_struct_comp_postproc_5_part2_info_df = pd.DataFrame({'misc_info': mut_complx_struct_comp_postproc_5_part2_info_lst, 'misc_info_val': mut_complx_struct_comp_postproc_5_part2_val_lst})
    mut_complx_struct_comp_postproc_5_part2_info_df.to_csv(os.path.join(curnt_dim_complx_postproc_result_dir, 'mut_complx_struct_comp_postproc_5_part2.csv'), index=False)
    print('inside run_postproc_5_part2_mut_complx_struct_comparison() method - End')


def calculate_rmsd_by_SVDSuperimposer(atoms_orig, atoms_mut):
    super_imposer = SVDSuperimposer()
    super_imposer.set(atoms_orig, atoms_mut)
    super_imposer.run()
    rmsd_superimp = super_imposer.get_rms()
    rmsd_superimp = round(rmsd_superimp, 3)
    rmsd_b4_superimp = super_imposer.get_init_rms()
    rmsd_b4_superimp = round(rmsd_b4_superimp, 3)
    return (rmsd_superimp, rmsd_b4_superimp)


def update_overall_complx_struct_comp_res(postproc_result_dir, delta_change_df):
    print('Inside update_overall_complx_struct_comp_res() method - Start')
    # Initialize a dataframe which will contain the updated overall complex structure comparison result and will be returned from this method
    con_df = None
    # check whether 'overall_complx_struct_comp_res_postproc_5_part2.csv' file already exists at postproc_result_dir
    csv_file_path = os.path.join(postproc_result_dir, 'overall_complx_struct_comp_res_postproc_5_part2.csv')
    if not os.path.exists(csv_file_path):
        print(f'"overall_complx_struct_comp_res_postproc_5_part2.csv" file does NOT exist at {postproc_result_dir}. Hence creating it...')
        con_df = delta_change_df
        con_df.to_csv(csv_file_path, index=False)
    else:
        print(f'"overall_complx_struct_comp_res_postproc_5_part2.csv" file already exists at {postproc_result_dir}. Hence updating or inserting rows...')
        con_df = pd.read_csv(csv_file_path)
        identifier_col_nm_lst = ['id', 'cmplx', 'fpi', 'mpi', 'batch_idx', 'simuln_idx', 'sequence']
        # modifiable_col_nm_lst = ['bb_rmsd_superimp', 'bb_rmsd_b4_superimp', 'bb_trace_rmsd_superimp', 'bb_trace_rmsd_b4_superimp']
        modifiable_col_nm_lst = ['bb_rmsd_superimp']
        
        # Case 1: Update existing rows
        con_df = con_df.set_index(identifier_col_nm_lst)
        delta_change_df = delta_change_df.set_index(identifier_col_nm_lst)
        update_mask_1 = con_df.index.isin(delta_change_df.index)
        update_mask_2 = delta_change_df.index.isin(con_df.index)
        con_df.loc[update_mask_1, modifiable_col_nm_lst] = delta_change_df.loc[update_mask_2, modifiable_col_nm_lst]

        # Case 2: Insert new rows
        insert_mask = ~delta_change_df.index.isin(con_df.index)
        con_df = pd.concat([con_df, delta_change_df.loc[insert_mask]])
        print("Rows updated or inserted successfully.")

        # reset index of con_df after update or insert
        con_df = con_df.reset_index()
        
        # overwrite the existing csv file with new content
        con_df.to_csv(csv_file_path, index=False)
    # end of if-else block: if not os.path.exists(csv_file_path):
    print('Inside update_overall_complx_struct_comp_res() method - End')
    return con_df



def generate_hist_rmsd_complex_struct_comp(root_path, itr_tag):
    postproc_result_dump_path = os.path.join(root_path, 'dataset/postproc_data/result_dump', itr_tag)
    overall_accept_complx_struct_comp_res_postproc_5_part2_df = pd.read_csv(os.path.join(postproc_result_dump_path, 'overall_accept_complx_struct_comp_res_postproc_5_part2.csv'))

    # Set the bin intervals (0.0 to 10.0 in steps of 1.0)
    bins = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

    # Create the histogram for the 'bb_rmsd_superimp' column with black bins
    plt.hist(overall_accept_complx_struct_comp_res_postproc_5_part2_df['bb_rmsd_superimp']
             , bins=bins, color='#6A6A6A', edgecolor='black', width=0.9)

    # Label the X-axis and Y-axis
    plt.xlabel('RMSD (Å) between native and designed complex structure', fontsize=12)
    plt.ylabel('Number of instances', fontsize=12)

    # Set the X and Y axis limits to ensure they start at the origin
    plt.xlim(0.0, 10.0)  # X-axis starts at 0.0
    plt.ylim(0, 1200)    # Y-axis starts at 0

    # Adjust the X-axis ticks to show decimal points
    plt.xticks(ticks=bins, labels=[f"{x:.1f}" for x in bins])

    # Save the histogram as a black-and-white JPG image
    hist_path = os.path.join(postproc_result_dump_path, 'hist_rmsd_complex_struct_comp.jpg')
    plt.savefig(hist_path, format='jpg', dpi=300, bbox_inches='tight')

    # Close the plot to avoid display in some environments
    plt.close()



if __name__ == '__main__':
    root_path = '/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/matpip_pd_prj'
    itr_tag = 'mcmc_fullLen_puFalse_batch5_mutPrcntLen10'
    generate_hist_rmsd_complex_struct_comp(root_path, itr_tag)
