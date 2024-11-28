import os, sys

from pathlib import Path
path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
import glob
import numpy as np
import pandas as pd
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB import PDBParser, PDBIO
from utils import PPIPUtils


def run_postproc_4_mut_seq_struct_overlay_for_complx_formn(**kwargs):
    """Postprocessing stage where mutated sequence structure (as predicted by AlphaFold2) is overlayed on the fixed chain structure (of the original dimeric complex) to form the mutated dimeric complex.

    This method is called from a triggering method like mat_p2ip_pd_postproc_trigger.trigger_pd_postproc().
    """
    print('inside run_postproc_4_mut_seq_struct_overlay_for_complx_formn() method - Start')
    print('####################################')
    # Iterate over kwargs and raise ValueError if any of the input arguments (except a few) is None. Also print each keyword argument name and respective value.
    for arg_name, arg_value in kwargs.items():
        if(arg_value is None):
            raise ValueError(f"Argument '{arg_name}' must be provided with a value.")
        print(f"'{arg_name}': {arg_value}")
    # end of for loop: for arg_name, arg_value in kwargs.items():

    # retrieve all the keyword arguments
    root_path = kwargs.get('root_path'); dim_prot_complx_nm = kwargs.get('dim_prot_complx_nm')
    postproc_result_dir = kwargs.get('postproc_result_dir'); pdb_file_location = kwargs.get('pdb_file_location'); af2_use_amber = kwargs.get('af2_use_amber')
    inp_dir_mut_seq_struct_pred_af2 = kwargs.get('inp_dir_mut_seq_struct_pred_af2')
    out_dir_mut_seq_struct_pred_af2 = kwargs.get('out_dir_mut_seq_struct_pred_af2')
    print('####################################')

    # Create the postprocess result directory for the given dimeric protein complex
    print(f'\nPostproc_4: Creating the postprocess result directory for the given dimeric protein complex: {dim_prot_complx_nm}')
    curnt_dim_complx_postproc_result_dir = os.path.join(postproc_result_dir, f'complex_{dim_prot_complx_nm}')
    PPIPUtils.createFolder(curnt_dim_complx_postproc_result_dir, recreate_if_exists=False)
    
    dim_prot_complx_tag = f' Postproc_4: complex_{dim_prot_complx_nm}:: '
    print(f'\ndim_prot_complx_tag: {dim_prot_complx_tag}')
    # creating few lists to track the mutated sequence structure overlaying on the fixed chain structure
    mut_seq_struct_overlay_postproc_4_info_lst, mut_seq_struct_overlay_postproc_4_val_lst = [], []

    # Search in all the CSV files with the specific name pattern 'af2inp_cmpl_{dim_prot_complx_nm}_*.csv' in the inp_dir_mut_seq_struct_pred_af2
    af2_specific_name_pattern = os.path.join(inp_dir_mut_seq_struct_pred_af2, f'af2inp_cmpl_{dim_prot_complx_nm}_*.csv')
    mut_seq_inp_csv_files_lst_for_chain_combo = glob.glob(af2_specific_name_pattern, recursive=False)
    # Find the number of resulting csv files
    num_mut_seq_inp_csv_files = len(mut_seq_inp_csv_files_lst_for_chain_combo)
    mut_seq_struct_overlay_postproc_4_info_lst.append('num_mut_seq_inp_csv_files'); mut_seq_struct_overlay_postproc_4_val_lst.append(f'{num_mut_seq_inp_csv_files}')
    # If the number of resulting csv files is zero, return
    if(num_mut_seq_inp_csv_files == 0):
        print(f"\n!!! {dim_prot_complx_tag} Found no CSV files with the specific name pattern 'af2inp_cmpl_{dim_prot_complx_nm}_*.csv' in the {inp_dir_mut_seq_struct_pred_af2}\
              \nSo, NOT proceeding further for the current dimeric complex: {dim_prot_complx_nm}\n")
        mut_seq_struct_overlay_postproc_4_info_lst.append('status'); mut_seq_struct_overlay_postproc_4_val_lst.append('ERROR')
        # save tracking info records and continue with the next dimeric complex
        mut_seq_struct_overlay_postproc_4_info_df = pd.DataFrame({'misc_info': mut_seq_struct_overlay_postproc_4_info_lst, 'misc_info_val': mut_seq_struct_overlay_postproc_4_val_lst})
        mut_seq_struct_overlay_postproc_4_info_df.to_csv(os.path.join(curnt_dim_complx_postproc_result_dir, 'mut_seq_struct_overlay_postproc_4.csv'), index=False)
        return  # return back
    # end of if block: if(num_mut_seq_inp_csv_files == 0):

    print(f"\n{dim_prot_complx_tag} There are {num_mut_seq_inp_csv_files} CSV file(s) with the specific name pattern 'af2inp_cmpl_{dim_prot_complx_nm}_*.csv' in the {inp_dir_mut_seq_struct_pred_af2} \
           \n Parsing every CSV file with the specific name pattern 'af2inp_cmpl_{dim_prot_complx_nm}_*.csv' in a loop ...")
    for indiv_mut_seq_inp_csv_file_name in mut_seq_inp_csv_files_lst_for_chain_combo:
        print(f'\n ###### {dim_prot_complx_tag} indiv_mut_seq_inp_csv_file_name: {indiv_mut_seq_inp_csv_file_name}')
        chain_combo_nm = indiv_mut_seq_inp_csv_file_name.split('/')[-1].replace(f'af2inp_cmpl_{dim_prot_complx_nm}_', '').replace('.csv', '')
        print(f'\n {dim_prot_complx_tag} chain_combo_nm: {chain_combo_nm}')  # Sample chain_combo_nm: "fpi_L_mpi_N"
        # Create the result directory to contain the overlaid mutated protein complex structires for the given chain combo
        print(f'\n{dim_prot_complx_tag} Creating the result directory to contain the overlaid mutated protein complex structires for the given chain combo ({chain_combo_nm}).')
        mod_chain_combo_nm = chain_combo_nm.replace('fpi', 'fixed').replace('mpi', 'mut')
        overlaid_mut_complx_struct_dir = os.path.join(curnt_dim_complx_postproc_result_dir, mod_chain_combo_nm, 'overlaid_mut_complx_struct_postproc_4')
        PPIPUtils.createFolder(overlaid_mut_complx_struct_dir, recreate_if_exists=False)
        # Next, extract fixed chain name from the chain_combo_nm
        fixed_chain_nm = chain_combo_nm.split('_')[1]
        # Load the fixed chain structure from the dimeric protein complex PDB file
        print(f'{dim_prot_complx_tag} Loading fixed chain ({fixed_chain_nm}) structure from the dimeric protein complex PDB file ({dim_prot_complx_nm}.pdb)')
        prot_complx_pdb_file_path = os.path.join(pdb_file_location, f"{dim_prot_complx_nm}.pdb")
        prot_complx_parser = PDBParser(QUIET=True)
        prot_complx_struct = prot_complx_parser.get_structure("prot_complx", prot_complx_pdb_file_path)
        fixed_chain = prot_complx_struct[0][fixed_chain_nm]

        # Select only the backbone atoms ("N", "CA", "C") from the fixed_chain
        bb_atoms_coord_fixed = [atom.get_coord() for atom in fixed_chain.get_atoms() if atom.name in ["N", "CA", "C"]]
        bb_atoms_coord_fixed = np.vstack(bb_atoms_coord_fixed)  # converting to 2d array from the list of 1d arrays
        
        # read the csv file (indiv_mut_seq_inp_csv_file_name) into a df
        indiv_mut_seq_inp_df = pd.read_csv(indiv_mut_seq_inp_csv_file_name)
        print(f'\n {dim_prot_complx_tag} Iterating {indiv_mut_seq_inp_csv_file_name} row-wise...')
        # iterate the df row-wise in an inner loop
        for index, row in indiv_mut_seq_inp_df.iterrows():
            id = row['id']  # Sample id: cmplx_2I25_fpi_L_mpi_N_batch_idx_2507_simuln_idx_7523 
            seq = row['sequence']
            print(f'\n {dim_prot_complx_tag} iteration index: {index} :: id: {id} :: seq: {seq}')
            # Create the pdb file location of the best ranked mutated sequence structure (as predicted by AlphaFold2)
            # sample pdb file name: cmplx_2I25_fpi_L_mpi_N_batch_idx_2507_simuln_idx_7523_unrelaxed_rank_001_alphafold2_ptm_model_2_seed_000.pdb
            relaxed_unrelaxed = 'unrelaxed'
            if(af2_use_amber):
                relaxed_unrelaxed = 'relaxed'
            best_rank_mut_seq_struct_pdb_fl_nm_pattern = os.path.join(out_dir_mut_seq_struct_pred_af2, dim_prot_complx_nm
                                                              , chain_combo_nm, f'{id}_{relaxed_unrelaxed}_rank_001_alphafold2_ptm_model_*.pdb')
            best_rank_pdb_fl_nm_lst = glob.glob(best_rank_mut_seq_struct_pdb_fl_nm_pattern, recursive=False)
            best_rank_mut_seq_struct_pdb_fl_nm = best_rank_pdb_fl_nm_lst[0]
            print(f'\n### {dim_prot_complx_tag} best_rank_mut_seq_struct_pdb_fl_nm: {best_rank_mut_seq_struct_pdb_fl_nm}')
            # parse the pdb file
            mutated_chain_parser = PDBParser(QUIET=True)
            mutated_chain_struct = mutated_chain_parser.get_structure("mutated_chain", best_rank_mut_seq_struct_pdb_fl_nm)
            mutated_chain = mutated_chain_struct[0].get_list()[0]
            # Select only the backbone atoms ("N", "CA", "C") from the mutated_chain
            bb_atoms_coord_mut = [atom.get_coord() for atom in mutated_chain.get_atoms() if atom.name in ["N", "CA", "C"]]
            bb_atoms_coord_mut = np.vstack(bb_atoms_coord_mut)  # converting to 2d array from the list of 1d arrays
            print(f'bb_atoms_coord_fixed.shape = {bb_atoms_coord_fixed.shape} :: bb_atoms_coord_mut.shape = {bb_atoms_coord_mut.shape}')


            print(f'\n{dim_prot_complx_tag} Overlaying process for id: {id} -Start')
            # TODO
            # ??? use PyMol??? https://pymol.org/dokuwiki/doku.php?id=api:cmd:alpha
            print('\n**** ???? NEED TO IMPLEMENT METHOD FOR Superimposition of two chains of varied lengths??? ***\n')

            print(f'\n{dim_prot_complx_tag} Overlaying process for id: {id} -End')
        # end of inner for loop: for index, row in indiv_mut_seq_inp_df.iterrows():
    # end of for loop: for indiv_mut_seq_inp_csv_file_name in mut_seq_inp_csv_files_lst_for_chain_combo:
    # save tracking info records
    mut_seq_struct_overlay_postproc_4_info_df = pd.DataFrame({'misc_info': mut_seq_struct_overlay_postproc_4_info_lst, 'misc_info_val': mut_seq_struct_overlay_postproc_4_val_lst})
    mut_seq_struct_overlay_postproc_4_info_df.to_csv(os.path.join(curnt_dim_complx_postproc_result_dir, 'mut_seq_struct_overlay_postproc_4.csv'), index=False)
    print('inside run_postproc_4_mut_seq_struct_overlay_for_complx_formn() method - End')


