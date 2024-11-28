# %%
import sys, os
from pathlib import Path
path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from Bio.PDB import PDBParser, PDBIO
from utils import naccess_util, PPIPUtils, prot_design_util
import traceback


def extract_chain(pdb_file, chain_id):
    """
    Extract coordinates and other information for a specific chain from a PDB file.

    Parameters:
    - pdb_file (str): Path to the input PDB file containing the dimeric protein complex.
    - chain_id (str): Chain identifier for the chain to be extracted.

    Returns:
    - chain_structure (Bio.PDB.Chain.Chain): Chain structure object containing the atoms of the specified chain.
    """
    # Create a PDB parser object
    parser = PDBParser()

    # Parse the PDB file
    structure = parser.get_structure("complex", pdb_file)

    # Extract the specified chain
    model = structure[0]
    for chain in model:
        if chain.id == chain_id:
            return chain

    return None


def save_chain_pdb(chain_structure, output_file):
    """
    Save the coordinates and other information for a chain into a PDB file.

    Parameters:
    - chain_structure (Bio.PDB.Chain.Chain): Chain structure object containing the atoms of the chain.
    - output_file (str): Path to the output PDB file to save the chain.
    """
    # Create a PDBIO object
    io = PDBIO()

    # Set the structure to save
    io.set_structure(chain_structure)

    # Save the chain to a PDB file
    io.save(output_file)


def find_interface_residue_indices_for_dimer(root_path='./', pdb_file=None, chain_nm_lst=None, naccess_path='.'):
    print('inside find_interface_residue_indices_for_dimer() method - Start')
    pdb_file_nm = pdb_file.split('/')[-1]
    print(f'pdb_file_nm: {pdb_file_nm}')
    prot_nm = pdb_file_nm.replace('.pdb', '')

    # Create naccess specific temp directory
    naccess_temp_dir_path = os.path.join(root_path, 'temp_naccess', prot_nm)
    PPIPUtils.createFolder(naccess_temp_dir_path)

    # Chaeck "chain_nm_lst" argument contains exactly 2 values. Otherwise raise exception.
    print(f'chain_nm_lst: {chain_nm_lst}')
    if(len(chain_nm_lst) != 2):
        raise Exception(f'Eoor!! Error!! For dimeric protein complex: {prot_nm} there must be 2 chains but input argument "chain_nm_lst" has {len(chain_nm_lst)} values: {chain_nm_lst}.')
    
    # Segregate and save chain-specific pdb files in the temp directory
    print('Segregating and saving chain-specific pdb files in the temp directory')
    output_chain_0_pdb_file = os.path.join(naccess_temp_dir_path, f"{prot_nm}_chain_{chain_nm_lst[0]}.pdb")
    output_chain_1_pdb_file = os.path.join(naccess_temp_dir_path, f"{prot_nm}_chain_{chain_nm_lst[1]}.pdb")
    # Extract chain_0
    chain_0_structure = extract_chain(pdb_file, chain_nm_lst[0])
    if chain_0_structure:
        save_chain_pdb(chain_0_structure, output_chain_0_pdb_file)
    # Extract chain_1
    chain_1_structure = extract_chain(pdb_file, chain_nm_lst[1])
    if chain_1_structure:
        save_chain_pdb(chain_1_structure, output_chain_1_pdb_file)

    # Extract chain sequences from the dimer to be used for the validation later
    chain_sequences_dict = prot_design_util.extract_chain_sequences(prot_nm, pdb_file.replace(f'{prot_nm}.pdb', ''))

    if(prot_nm == '2A5T'):  # special handling for '2A5T'
        chain_sequences_dict['A'] = chain_sequences_dict['A'][:-1]
        chain_sequences_dict['B'] = chain_sequences_dict['B'][:-1]
    if(prot_nm == '4CPA'):  # special handling for '4CPA'
        chain_sequences_dict['A'] = chain_sequences_dict['A'][:-1]

    # Run naccess for the dimeric protein complex
    print(f'#### Running naccess for the dimeric protein complex: {prot_nm}')
    dimer_rsa_data, dimer_asa_data = naccess_util.run_naccess(None, pdb_file, naccess=naccess_path, temp_dir=naccess_temp_dir_path)
    # Process the .rsa output file: residue level SASA data
    dimer_naccess_rsa_dict = naccess_util.process_rsa_data(dimer_rsa_data)

    # Run naccess for the chain_0
    print(f'#### Running naccess for the chain_0: {chain_nm_lst[0]}')
    chain_0_rsa_data, chain_0_asa_data = naccess_util.run_naccess(None, output_chain_0_pdb_file, naccess=naccess_path, temp_dir=naccess_temp_dir_path)
    # Process the .rsa output file: residue level SASA data
    chain_0_naccess_rsa_dict = naccess_util.process_rsa_data(chain_0_rsa_data)
    # validate rsa_dict
    print(f'Validating rsa_dict for the chain: {chain_nm_lst[0]}')
    naccess_util.validate_rsa_dict(chain_sequences_dict, chain_nm_lst[0], chain_0_naccess_rsa_dict)

    # Run naccess for the chain_1
    print(f'#### Running naccess for the chain_1: {chain_nm_lst[1]}')
    chain_1_rsa_data, chain_1_asa_data = naccess_util.run_naccess(None, output_chain_1_pdb_file, naccess=naccess_path, temp_dir=naccess_temp_dir_path)
    # Process the .rsa output file: residue level SASA data
    chain_1_naccess_rsa_dict = naccess_util.process_rsa_data(chain_1_rsa_data)
    # validate rsa_dict
    print(f'Validating rsa_dict for the chain: {chain_nm_lst[1]}')
    naccess_util.validate_rsa_dict(chain_sequences_dict, chain_nm_lst[1], chain_1_naccess_rsa_dict)
    # Find the interfacing residues for the given dimer
    print(f'\n Find the interfacing residues for the dimeric protein complex: {prot_nm}')
    intrfc_residues_dict = estimate_interface_residues(chain_nm_lst, chain_sequences_dict, dimer_naccess_rsa_dict
                                                             , chain_0_naccess_rsa_dict, chain_1_naccess_rsa_dict)
    print('inside find_interface_residue_indices_for_dimer() method - End')
    return intrfc_residues_dict


def estimate_interface_residues(chain_nm_lst, chain_sequences_dict, dimer_naccess_rsa_dict, chain_0_naccess_rsa_dict, chain_1_naccess_rsa_dict):
    print('inside estimate_interface_residues() method - Start')
    # Find the interfacing residues for the given dimer following the 
    # paper "Interfacial residues in protein–protein complexes are in the eyes of the beholder" (https://doi.org/10.1101/2023.04.24.538134).
    
    # ####################### Identification of interfacing residues (core and rim) by ASA method
    # In ASA-based methods, those residues in a protein–protein complex that are solvent-exposed in the unbound form of the protein to which
    # it belongs and become buried upon binding with the partner protein are considered to be at the geometrical interface. This study identifies
    # two categories of interface residues using the ASA-based method. A residue that is well solvent-exposed in the isolated
    # form of the protein, as reflected by the Relative Accessibility Surface Area (RASA) value of greater than 10% and becomes buried in the
    # complex form with a RASA value less than 7%, is located at the center  of the interface and categorized as “core” residue. The residues par-
    # tially exposed in the unbound form of the protein (RASA > 7%) and lose the ASA by more than 1 Å2 include residues at the center as well
    # as the periphery of the interface. Those residues that are placed at the periphery of the interface are named “rim” residues. This method
    # identified a more extensive set of residues that are part of the geometrical interface and lie at the center as well as the periphery of the
    # interface (Figure 1B). Naccess has been used to calculate residue-wise ASA and RASA values in the complex and isolated forms of the protein.
    print('####################### Identification of interfacing residues (core and rim) by ASA method -Start')
    intrfc_residues_asa_dict = {}
    # Find core and rim residues for each chain of the dimer
    for chain_idx, chain_name in enumerate(chain_nm_lst):
        print(f'chain_idx: {chain_idx} :: chain_name: {chain_name}')
        chain_sequence = chain_sequences_dict[chain_name]
        chain_sequence_as_lst = list(chain_sequence)
        chain_naccess_rsa_dict = chain_0_naccess_rsa_dict if(chain_idx == 0) else chain_1_naccess_rsa_dict
        core_residue_idx_lst, rim_residue_idx_lst = [], []
        # Iterate over each amino-acid (residue) of the current chain and check for core and rim residue criteria
        for seq_idx, aa in enumerate(chain_sequence_as_lst):
            aa_rasa_isolated = chain_naccess_rsa_dict[(chain_name, seq_idx)]["all_atoms_rel"]
            aa_rasa_complx = dimer_naccess_rsa_dict[(chain_name, seq_idx)]["all_atoms_rel"]
            # check for the "core" residue criteria
            if((aa_rasa_isolated > 10.0) and (aa_rasa_complx < 7.0)):
                # satisfies "core" residue criteria
                core_residue_idx_lst.append(seq_idx)
            
            # check for the "rim" residue criteria
            aa_asa_isolated = chain_naccess_rsa_dict[(chain_name, seq_idx)]["all_atoms_abs"]
            aa_asa_complx = dimer_naccess_rsa_dict[(chain_name, seq_idx)]["all_atoms_abs"]
            aa_loss_in_asa = aa_asa_isolated - aa_asa_complx
            if((aa_rasa_isolated > 7.0) and (aa_loss_in_asa > 1.0)):
                # satisfies "rim" residue criteria
                rim_residue_idx_lst.append(seq_idx)
        # end of for loop: for seq_idx, aa in enumerate(chain_sequence_as_lst):
        # Take the union of core_residue_idx_lst and rim_residue_idx_lst
        intrfc_residue_lst = list(set(core_residue_idx_lst + rim_residue_idx_lst))
        # Add the intrfc_residue_lst corresponding to the current chain in intrfc_residues_asa_dict
        intrfc_residues_asa_dict[chain_name] = intrfc_residue_lst
    # end of for loop: for chain_idx, chain_name in enumerate(chain_nm_lst):
    print('####################### Identification of interfacing residues (core and rim) by ASA method -End')
    print('inside estimate_interface_residues() method - End')
    return intrfc_residues_asa_dict


def find_interface_residue_for_all_dimers(root_path='./', naccess_path='./') :
    # dim_prot_complx_nm_lst_10_orig = ['2I25']
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

    pdb_directory = os.path.join(root_path, 'dataset/preproc_data/pdb_files')
    for itr, dim_prot_complx_nm in enumerate(dim_prot_complx_nm_lst_effective):
        print(f'\n\n #### dim_prot_complx_nm: {dim_prot_complx_nm}: Iteration {itr+1} out of {len(dim_prot_complx_nm_lst_effective)}\n\n')
        try:
            pdb_file = os.path.join(pdb_directory, f"{dim_prot_complx_nm}.pdb")
            chain_sequences_dict = prot_design_util.extract_chain_sequences(dim_prot_complx_nm, pdb_directory)
            chain_nm_lst = list(chain_sequences_dict.keys())
            intrfc_resid_idx_dict = find_interface_residue_indices_for_dimer(root_path=root_path, pdb_file=pdb_file, chain_nm_lst=chain_nm_lst, naccess_path=naccess_path)
        except Exception as ex:
            # printing stack trace 
            traceback.print_exc(file=sys.stdout)
            print(f'************ ############## Error in processing :: {dim_prot_complx_nm} :: ex: {ex}')
        # End of try-except block
    # end of for loop: for itr, dim_prot_complx_nm in enumerate(dim_prot_complx_nm_lst_effective):


if __name__ == '__main__':
    root_path = '/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/matpip_pd_prj'
    naccess_path = "/scratch/pralaycs/Shubh_Working_Remote/naccess/naccess"
    # pdb_file = os.path.join(root_path, 'dataset/preproc_data/pdb_files/2I25.pdb')
    # chain_nm_lst = ['L', 'N']
    # intrfc_resid_idx_dict = find_interface_residue_indices_for_dimer(root_path=root_path, pdb_file=pdb_file, chain_nm_lst=chain_nm_lst, naccess_path=naccess_path)
    # print('done')

    find_interface_residue_for_all_dimers(root_path=root_path, naccess_path=naccess_path) 
# %%
