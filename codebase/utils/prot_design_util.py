import sys, os
from pathlib import Path
path_root = Path(__file__).parents[1]  
sys.path.insert(0, str(path_root))

from Bio import PDB, SeqIO
from itertools import permutations


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
    pdb_file_path = os.path.join(pdb_directory, f"{pdb_id}.pdb")
    sequences_first_chain_mapping = {}
    try:
        with open(pdb_file_path, "r") as pdb_file:
            for record in SeqIO.parse(pdb_file, "pdb-seqres"):
                chain_id = record.annotations["chain"]
                sequence = str(record.seq)
                if sequence not in sequences_first_chain_mapping:
                    sequences_first_chain_mapping[sequence] = chain_id
        chain_sequences_dict = {chain_name: sequence for sequence, chain_name in sequences_first_chain_mapping.items()}
        parser = PDB.PDBParser(QUIET=True)
        try:
            structure = parser.get_structure(pdb_id, pdb_file_path)
            for model in structure:
                for chain in model:
                    chain_name = chain.get_id()
                    chain_sequence = ""
                    for residue in chain:
                        if PDB.is_aa(residue, standard=True):
                            chain_sequence += PDB.Polypeptide.index_to_one(PDB.Polypeptide.three_to_index(residue.get_resname()))
                    if(chain_name in chain_sequences_dict.keys()):
                        chain_sequences_dict[chain_name] = chain_sequence
        except Exception as e:
            return f"Error: {e}"
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
    for key1, key2 in permutations(keys, 2):
        unique_tuples_lst.append((key1, key2))
    return unique_tuples_lst
