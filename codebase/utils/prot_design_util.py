import sys, os
from pathlib import Path
path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
from Bio import PDB, SeqIO
import urllib.request
from itertools import permutations
import pandas as pd
import shutil


def get_protein_sequence(prot_id, pdb_file_location="."):
    """
    Retrieve the protein sequence for the given protein ID using structure information from a PDB file.
    If the PDB file is not found in the specified location, download and save it.

    Args:
    - prot_id (str): Protein ID as a string.
    - pdb_file_location (str): Directory location to search for or save PDB files. Default is the current directory.

    Returns:
    - str: Protein sequence corresponding to the given protein ID.

    Example:
    >>> get_protein_sequence("1abc")
    'MKTFI...'
    """
    # Define the PDB file path in the specified location
    pdb_file_path = os.path.join(pdb_file_location, f"{prot_id}.pdb")

    # Check if the PDB file exists in the specified location
    if not os.path.isfile(pdb_file_path):
        # If not found, download the PDB file and save it
        pdb_url = f"https://files.rcsb.org/download/{prot_id}.pdb"
        try:
            urllib.request.urlretrieve(pdb_url, pdb_file_path)
        except Exception as e:
            return f"Error downloading PDB file: {e}"

    # Create a PDB parser
    parser = PDB.PDBParser(QUIET=True)
    # parser = PDB.PDBParser(QUIET=False)

    try:
        # Parse the PDB file
        structure = parser.get_structure(prot_id, pdb_file_path)

        # Initialize an empty string to store the protein sequence
        protein_sequence = ""

        # Iterate through the structure and extract the amino acid sequence
        for model in structure:
            for chain in model:
                for residue in chain:
                    if PDB.is_aa(residue, standard=True):
                        # protein_sequence += PDB.Polypeptide.three_to_one(residue.get_resname())
                        protein_sequence += PDB.Polypeptide.index_to_one(PDB.Polypeptide.three_to_index(residue.get_resname()))

        return protein_sequence

    except Exception as e:
        # Handle exceptions such as parsing errors
        return f"Error: {e}"


def extract_chain_sequences(pdb_id, pdb_directory):
    """
    Extracts chain names and sequences from a protein complex PDB file.

    Args:
        pdb_id (str): The protein complex PDB ID.
        pdb_directory (str): The directory containing the downloaded PDB file.

    Returns:
        dict: A dictionary containing unique sequences as keys and their corresponding first encountered chain names as values.

    Raises:
        FileNotFoundError: If the PDB file corresponding to the given ID is not found in the specified directory.
    """
    # Construct the path to the PDB file
    pdb_file_path = os.path.join(pdb_directory, f"{pdb_id}.pdb")

    # Initialize an empty dictionary to store sequences and corresponding first encountered chain names
    sequences_first_chain_mapping = {}

    try:
        # Parse the PDB file and extract chain names and sequences
        with open(pdb_file_path, "r") as pdb_file:
            for record in SeqIO.parse(pdb_file, "pdb-seqres"):
                chain_id = record.annotations["chain"]
                sequence = str(record.seq)

                # If the sequence is encountered for the first time, store its first encountered chain name
                if sequence not in sequences_first_chain_mapping:
                    sequences_first_chain_mapping[sequence] = chain_id

        # Print the extracted sequences and corresponding first encountered chain names
        # for sequence, first_chain_name in sequences_first_chain_mapping.items():
        #     print(f"Sequence: {sequence}, First encountered chain name: {first_chain_name}")
        
        # Reverse the seq_first_chain_mapping (swap keys and values)
        # chain_sequences_dict contains chain names as keys and their corresponding sequences as values
        chain_sequences_dict = {chain_name: sequence for sequence, chain_name in sequences_first_chain_mapping.items()}

        # ################### Refine the sequences to filter out non-standard amino acids -Start
        # Create a PDB parser
        parser = PDB.PDBParser(QUIET=True)
        # parser = PDB.PDBParser(QUIET=False)
        try:
            # Parse the PDB file
            structure = parser.get_structure(pdb_id, pdb_file_path)
            # Iterate through the structure and extract the amino acid sequence
            for model in structure:
                for chain in model:
                    chain_name = chain.get_id()
                    # Initialize an empty string to store the chain sequence
                    chain_sequence = ""
                    for residue in chain:
                        # filter out non-standard amino acids
                        if PDB.is_aa(residue, standard=True):
                            # chain_sequence += PDB.Polypeptide.three_to_one(residue.get_resname())
                            chain_sequence += PDB.Polypeptide.index_to_one(PDB.Polypeptide.three_to_index(residue.get_resname()))
                        # else:
                        #     print(f'####### Warning!! Warning !! non-standard amino acid: {residue}')
                    # end of for loop: for residue in chain:
                    # check if the current chain name is a member of chain_sequences_dict
                    if(chain_name in chain_sequences_dict.keys()):
                        # if yes, then replace the existing chain sequence with the refined sequence
                        chain_sequences_dict[chain_name] = chain_sequence
                # end of for loop: for chain in model:
            # end of outermost for loop: for model in structure:
        except Exception as e:
            # Handle exceptions such as parsing errors
            return f"Error: {e}"
        # end of try-catch block
        # ################### Refine the sequences to filter out non-standard amino acids -End
                        
        # Return chain_sequences_dict
        return chain_sequences_dict

    except FileNotFoundError:
        raise FileNotFoundError(f"PDB file {pdb_id}.pdb not found in directory: {pdb_directory}")


def gen_unique_tuples_from_keys(dictionary):
    """
    Generate all possible tuples of keys from the given dictionary where the tuple containing the same keys is not allowed.
    
    Args:
        dictionary (dict): The input dictionary.

    Returns:
        list: A list of tuples containing all possible combinations of keys from the dictionary where the tuple containing the same keys is not allowed.

    Example:
        >>> my_dict = {'a': 1, 'b': 2}
        >>> generate_unique_tuples(my_dict)
        [('a', 'b'), ('b', 'a')]
    """
    keys = list(dictionary.keys())
    unique_tuples_lst = []

    # Generate permutations of keys
    for key1, key2 in permutations(keys, 2):
        unique_tuples_lst.append((key1, key2))

    return unique_tuples_lst


def create_mdSim_res_dir_and_return_status(dir_path, recreate_if_exists=False):
    mdSim_status = 'exec_mdSim'
    try:
        # check if the dir_path already exists and if not, then create it
        if not os.path.exists(dir_path):
            print(f"The directory: '{dir_path}' does not exist. Creating it...")
            os.makedirs(dir_path)
        else:
            print(f"The directory '{dir_path}' already exists.")
            if(recreate_if_exists):
                print(f"As the input argument 'recreate_if_exists' is True, deleting the existing directory and recreating the same...")
                shutil.rmtree(dir_path, ignore_errors=False, onerror=None)
                os.makedirs(dir_path)
            else:
                print(f"As the input argument 'recreate_if_exists' is False, keeping the existing directory as it is ...")
                mdSim_status = 'mdSim_res_already_exists'
    except OSError as ex:
        errorMessage = "Creation of the directory " + dir_path + " failed. Exception is: " + str(ex)
        raise Exception(errorMessage)
    else:
        print("Returning back from the createFolder() method.")
        return mdSim_status


def create_pdb_for_chain(dimer_id='2I25', pdb_file_path='./', chain_name='L', chain_pdb_file_path='./'):
    """
    Extracts a specified chain from a PDB file, filters out non-standard amino acids,
    and saves the filtered chain to a new PDB file.
    
    Args:
        dimer_id (str): PDB id of the dimeric protein complex (e.g., '2I25').
        pdb_file_path (str): File path to the dimer PDB file (e.g., 'path/to/2I25.pdb').
        chain_name (str): Chain identifier to be extracted (e.g., 'L').
        chain_pdb_file_path (str): File path where the filtered chain PDB will be saved (e.g., 'path/to/2I25_L.pdb').
    
    Returns:
        None
    """
    # print('Inside create_pdb_for_chain() method - Start')

    # Construct the path to the PDB file
    pdb_file_path = os.path.join(pdb_file_path, f"{dimer_id}.pdb")

    # Parse the PDB file
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(dimer_id, pdb_file_path)
    
     # Initialize a list to hold residues of the desired chain
    selected_residues = []

    # Extract the specified chain
    for model in structure:
        for chain in model:
            if chain.id == chain_name:
                for residue in chain:
                    # Check for non-standard amino acids (usually residue names longer than 3 characters)
                    if PDB.is_aa(residue, standard=True):
                        selected_residues.append(residue)
    # end of for loop: for model in structure:
    
    # Create a new structure with only the filtered residues
    new_structure = PDB.Structure.Structure(dimer_id)
    model = PDB.Model.Model(0)
    new_chain = PDB.Chain.Chain(chain_name)
    
    for residue in selected_residues:
        new_residue = PDB.Residue.Residue(residue.id, residue.resname, residue.segid)
        for atom in residue:
            new_residue.add(atom)
        new_chain.add(new_residue)
    # end of for loop: for residue in selected_residues:
    
    model.add(new_chain)
    new_structure.add(model)

    # Output file path
    output_file_path = os.path.join(chain_pdb_file_path, f"{dimer_id}_{chain_name}.pdb")

    # Save the filtered chain to a new PDB file
    io = PDB.PDBIO()
    io.set_structure(new_structure)
    io.save(output_file_path)

    # print(f"Filtered chain saved to {output_file_path}")
    # print('Inside create_pdb_for_chain() method - End')


def pdb_chain_splitter(root_path='./', pdb_file_location='./'):
    print('Inside pdb_chain_splitter() method - Start')

    # dim_prot_complx_nm_lst = ['1GLA']
    dim_prot_complx_nm_lst = ['2I25', '4Y7M', '5JMO', '5SV3', '6CWG', '6DBG', '1AVX', '1AY7', '1BUH', '1BVN'] \
    + ['1CLV', '1D6R', '1DFJ', '1E6E', '1EAW', '1EWY', '1F34', '1FLE', '1GL1', '1GLA'] \
    + ['1GXD', '1JTG', '1MAH', '1OC0', '1OPH', '1OYV', '1PPE', '1R0R', '1TMQ'] \
    + ['1UDI', '1US7', '1YVB', '1Z5Y', '2A9K', '2ABZ', '2AYO', '2B42', '2J0T', '2O8V'] \
    + ['2OOB', '2OUL', '2PCC', '2SIC', '2SNI', '2UUY', '3SGQ', '4CPA', '7CEI', '1AK4'] \
    + ['1E96', '1EFN', '1FFW', '1FQJ', '1GCQ', '1GHQ', '1GPW', '1H9D', '1HE1', '1J2J'] \
    + ['1KAC', '1KTZ', '1KXP', '1PVH', '1QA9', '1S1Q', '1SBB', '1T6B', '1XD3', '1Z0K'] \
    + ['1ZHH', '1ZHI', '2A5T', '2AJF', '2BTF', '2FJU', '2G77', '2HLE', '2HQS', '2VDB', '3D5S']

    chain_pdb_file_path = os.path.join(root_path, 'dataset/preproc_data/pdb_chain_files')

    for idx, dim_prot_complx_nm in enumerate(dim_prot_complx_nm_lst):
        print(f'dim_prot_complx_nm: {dim_prot_complx_nm}')
        chain_sequences_dict = extract_chain_sequences(dim_prot_complx_nm, pdb_file_location)
        pdb_chain_nm_lst = list(chain_sequences_dict.keys())
        for chain_nm in pdb_chain_nm_lst:
            print(f'idx: {idx} :: dim_prot_complx_nm: {dim_prot_complx_nm} :: chain_nm: {chain_nm}')
            create_pdb_for_chain(dimer_id=dim_prot_complx_nm, pdb_file_path=pdb_file_location
                                 , chain_name=chain_nm, chain_pdb_file_path=chain_pdb_file_path)
        # end of for loop: for idx, chain_nm in enumerate(pdb_chain_nm_lst):
    # end of for loop: for dim_prot_complx_nm in dim_prot_complx_nm_lst:
    print('Inside pdb_chain_splitter() method - End')


def list_missing_cmplex_nm_a4_chainStructComparison_pp3(root_path, iteration_tag):
    # ####################################### missing_complx_nm_lst after chain structure comparison (postproc_3) #########################################
    dim_prot_complx_nm_lst_10_orig = ['2I25', '4Y7M', '5JMO', '5SV3', '6CWG', '6DBG', '1AVX', '1AY7', '1BUH', '1BVN'] \
    + ['1CLV', '1D6R', '1DFJ', '1E6E', '1EAW', '1EWY', '1F34', '1FLE', '1GL1', '1GLA'] \
    + ['1GXD', '1JTG', '1MAH', '1OC0', '1OPH', '1OYV', '1PPE', '1R0R', '1TMQ'] \
    + ['1UDI', '1US7', '1YVB', '1Z5Y', '2A9K', '2ABZ', '2AYO', '2B42', '2J0T', '2O8V'] \
    + ['2OOB', '2OUL', '2PCC', '2SIC', '2SNI', '2UUY', '3SGQ', '4CPA', '7CEI', '1AK4'] \
    + ['1E96', '1EFN', '1FFW', '1FQJ', '1GCQ', '1GHQ', '1GPW', '1H9D', '1HE1', '1J2J'] \
    + ['1KAC', '1KTZ', '1KXP', '1PVH', '1QA9', '1S1Q', '1SBB', '1T6B', '1XD3', '1Z0K'] \
    + ['1ZHH', '1ZHI', '2A5T', '2AJF', '2BTF', '2FJU', '2G77', '2HLE', '2HQS', '2VDB', '3D5S']
    dim_prot_complx_nm_lst_done =   []
    dim_prot_complx_nm_lst_notConsidered = ['5JMO', '1PPE', '1FQJ', '1OYV', '2BTF', '1H9D', '1BUH', '1QA9', '1S1Q', '1XD3', '1PVH']
    dim_prot_complx_nm_lst_chainBreak = ['1AVX', '1AY7', '1BUH', '1D6R', '1EAW', '1EFN', '1F34', '1FLE', '1GL1', '1GLA', '1GPW', '1GXD', '1H9D', '1JTG', '1KTZ', '1KXP', '1MAH', '1OC0', '1OPH', '1OYV' \
                                        ,'1PPE', '1R0R', '1S1Q', '1SBB', '1T6B', '1US7', '1XD3', '1YVB', '1Z5Y', '1ZHH', '1ZHI', '2AST', '2ABZ', '2AJF', '2AYO', '2B42', '2BTF', '2FJU', '2G77', '2HLE' \
                                        ,'2HQS', '2O8V', '2PCC', '2UUY', '2VDB', '3SGQ', '4Y7M', '5JMO', '5SV3', '6CWG', '6DBG']
    dim_prot_complx_nm_lst_excluded = dim_prot_complx_nm_lst_done + dim_prot_complx_nm_lst_notConsidered
    dim_prot_complx_nm_lst_effective = [prot_id for prot_id in dim_prot_complx_nm_lst_10_orig if prot_id not in dim_prot_complx_nm_lst_excluded]

    postproc_dump_res_dir = os.path.join(root_path, 'dataset/postproc_data/result_dump', iteration_tag)
    overall_accept_chain_struct_comp_res_df = pd.read_csv(os.path.join(postproc_dump_res_dir, 'overall_accept_chain_struct_comp_res_postproc_3.csv'))
    overall_accept_complx_lst = overall_accept_chain_struct_comp_res_df['cmplx'].drop_duplicates().tolist()
    missing_complx_nm_lst = [complx_nm for complx_nm in dim_prot_complx_nm_lst_effective if complx_nm not in overall_accept_complx_lst]
    print(f'missing_complx_nm_lst after chain structure comparison (postproc_3):\n {missing_complx_nm_lst}')
    # ### missing_complx_nm_lst: ['1US7', '1GPW']


def list_missing_cmplex_nm_a4_complxStructComparison_pp5_2(root_path, iteration_tag):
    # ############################################# considered complex list for MD simulation in the sorted order of length #######################
    dim_prot_complx_nm_lst_10_orig = ['2I25', '4Y7M', '5JMO', '5SV3', '6CWG', '6DBG', '1AVX', '1AY7', '1BUH', '1BVN'] \
    + ['1CLV', '1D6R', '1DFJ', '1E6E', '1EAW', '1EWY', '1F34', '1FLE', '1GL1', '1GLA'] \
    + ['1GXD', '1JTG', '1MAH', '1OC0', '1OPH', '1OYV', '1PPE', '1R0R', '1TMQ'] \
    + ['1UDI', '1US7', '1YVB', '1Z5Y', '2A9K', '2ABZ', '2AYO', '2B42', '2J0T', '2O8V'] \
    + ['2OOB', '2OUL', '2PCC', '2SIC', '2SNI', '2UUY', '3SGQ', '4CPA', '7CEI', '1AK4'] \
    + ['1E96', '1EFN', '1FFW', '1FQJ', '1GCQ', '1GHQ', '1GPW', '1H9D', '1HE1', '1J2J'] \
    + ['1KAC', '1KTZ', '1KXP', '1PVH', '1QA9', '1S1Q', '1SBB', '1T6B', '1XD3', '1Z0K'] \
    + ['1ZHH', '1ZHI', '2A5T', '2AJF', '2BTF', '2FJU', '2G77', '2HLE', '2HQS', '2VDB', '3D5S']
    dim_prot_complx_nm_lst_done =   []
    dim_prot_complx_nm_lst_notConsidered = ['5JMO', '1PPE', '1FQJ', '1OYV', '2BTF', '1H9D', '1BUH', '1QA9', '1S1Q', '1XD3', '1PVH']
    dim_prot_complx_nm_lst_chainBreak = ['1AVX', '1AY7', '1BUH', '1D6R', '1EAW', '1EFN', '1F34', '1FLE', '1GL1', '1GLA', '1GPW', '1GXD', '1H9D', '1JTG', '1KTZ', '1KXP', '1MAH', '1OC0', '1OPH', '1OYV' \
                                        ,'1PPE', '1R0R', '1S1Q', '1SBB', '1T6B', '1US7', '1XD3', '1YVB', '1Z5Y', '1ZHH', '1ZHI', '2AST', '2ABZ', '2AJF', '2AYO', '2B42', '2BTF', '2FJU', '2G77', '2HLE' \
                                        ,'2HQS', '2O8V', '2PCC', '2UUY', '2VDB', '3SGQ', '4Y7M', '5JMO', '5SV3', '6CWG', '6DBG']
    dim_prot_complx_nm_lst_excluded = dim_prot_complx_nm_lst_done + dim_prot_complx_nm_lst_notConsidered
    dim_prot_complx_nm_lst_effective = [prot_id for prot_id in dim_prot_complx_nm_lst_10_orig if prot_id not in dim_prot_complx_nm_lst_excluded]

    postproc_dump_res_dir = os.path.join(root_path, 'dataset/postproc_data/result_dump', iteration_tag)
    overall_accept_complx_struct_comp_res_df = pd.read_csv(os.path.join(postproc_dump_res_dir, 'overall_accept_complx_struct_comp_res_postproc_5_part2.csv'))
    overall_accept_complx_lst = overall_accept_complx_struct_comp_res_df['cmplx'].drop_duplicates().tolist()
    missing_complx_nm_lst = [complx_nm for complx_nm in dim_prot_complx_nm_lst_effective if complx_nm not in overall_accept_complx_lst]
    print(f'\n missing_complx_nm_lst after complex structure comparison (postproc_5_2):\n {missing_complx_nm_lst}')
    # ### already missing_complx_nm_lst from chain-structure comparison: ['1US7', '1GPW']

    preproc_protein_complex_details_df = pd.read_csv(os.path.join(root_path, 'dataset/preproc_data/protein_complex_details.csv'))
    considered_complx_for_mds_df = preproc_protein_complex_details_df[preproc_protein_complex_details_df['PDB_ID'].isin(dim_prot_complx_nm_lst_effective)]

    considered_complx_for_mds_csv_loc = os.path.join(postproc_dump_res_dir, 'considered_complx_for_mds_postproc_5_part2.csv')
    considered_complx_for_mds_df.to_csv(considered_complx_for_mds_csv_loc, index=False)
    print(f'\n considered complex names for MD simulation in ascending order of length (postproc_5_2):\n {considered_complx_for_mds_df["PDB_ID"].tolist()}')




if __name__ == '__main__':
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/matpip_pd_prj')
    iteration_tag = 'mcmc_intrfc_puFalse_batch5_mutPrcntLen10'
    
    # chain_sequences_dict = extract_chain_sequences('2I25', os.path.join(root_path, 'dataset/preproc_data/pdb_files'))
    # print(f'len(seq): {len(chain_sequences_dict["N"])}')
    # print(f'seq: {chain_sequences_dict["N"]}')

    # create_pdb_for_chain(dimer_id='2I25'
    #                      , pdb_file_path=os.path.join(root_path, 'dataset/preproc_data/pdb_files')
    #                      , chain_name='L'
    #                      , chain_pdb_file_path=os.path.join(root_path, 'dataset/preproc_data/pdb_chain_files'))

    # pdb_chain_splitter(root_path=root_path
    #                    , pdb_file_location=os.path.join(root_path, 'dataset/preproc_data/pdb_files'))
    
    list_missing_cmplex_nm_a4_chainStructComparison_pp3(root_path, iteration_tag)