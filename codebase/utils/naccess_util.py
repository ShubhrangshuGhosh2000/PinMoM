# NACCESS interface adapted from Bio/PDB/DSSP.py

"""Interface for the program NACCESS.

See: http://www.bioinf.manchester.ac.uk/naccess/
Atomic Solvent Accessible Area Calculations

errors likely to occur with the binary:
default values are often due to low default settings in accall.pars
- e.g. max cubes error: change in accall.pars and recompile binary

use naccess -y, naccess -h or naccess -w to include HETATM records
"""

import os
import shutil
import subprocess
import warnings
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.AbstractPropertyMap import (
    AbstractResiduePropertyMap,
    AbstractAtomPropertyMap,
)
from Bio import PDB


def run_naccess(
    model, pdb_file, probe_size=None, z_slice=None, naccess="naccess", temp_dir="/tmp/"
):
    """Run naccess for a pdb file."""
    pdb_file_nm = pdb_file.split('/')[-1]
    prot_nm = pdb_file_nm.replace('.pdb', '')

    if pdb_file:
        pdb_file = os.path.abspath(pdb_file)
        # Copy only the PDB file for the dimeric protein complex in the temp_dir.
        # PDB files for the isolated chains are already in the temp_dir
        if(("_chain_") not in pdb_file_nm):
            shutil.copy(pdb_file, os.path.join(temp_dir, pdb_file_nm))
    else:
        writer = PDBIO()
        writer.set_structure(model.get_parent())
        writer.save(temp_dir)

    # chdir to temp directory, as NACCESS writes to current working directory
    old_dir = os.getcwd()
    os.chdir(temp_dir)

    # create the command line and run
    # catch standard out & err
    command = [naccess, os.path.join(temp_dir, pdb_file_nm)]
    if probe_size:
        command.extend(["-p", probe_size])
    if z_slice:
        command.extend(["-z", z_slice])

    p = subprocess.Popen(
        command, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out, err = p.communicate()
    os.chdir(old_dir)

    rsa_file = os.path.join(temp_dir, f'{prot_nm}.rsa')
    asa_file = os.path.join(temp_dir, f'{prot_nm}.asa')
    # Alert user for errors
    if err.strip():
        warnings.warn(err)

    if (not os.path.exists(rsa_file)) or (not os.path.exists(asa_file)):
        print(f'{prot_nm}:: !!! Error !!! NACCESS did not execute or finish properly: "{out}"')
        raise Exception("NACCESS did not execute or finish properly.")

    # get the output, then delete the temp directory
    with open(rsa_file) as rf:
        rsa_data = rf.readlines()
    with open(asa_file) as af:
        asa_data = af.readlines()

    # shutil.rmtree(tmp_path, ignore_errors=True)
    return rsa_data, asa_data


def process_rsa_data(rsa_data):
    """Process the .rsa output file: residue level SASA data."""
    naccess_rsa_dict = {}
    seq_idx = 0
    prv_chain_id = None
    for line in rsa_data:
        if line.startswith("RES"):
            res_name = line[4:7]
            if(not PDB.is_aa(res_name, standard=True)):
                continue  # For non-standard aa, skip the line
            chain_id = line[8]
            if(prv_chain_id is None):
                prv_chain_id = chain_id
            elif(prv_chain_id != chain_id):
                # There is a change in chain id. So, reset seq_idx
                seq_idx = 0
                prv_chain_id = chain_id
            # end of if-else block
            resseq = int(line[9:13])
            icode = line[13]
            res_id = (" ", resseq, icode)
            naccess_rsa_dict[(chain_id, seq_idx)] = {
                "resseq": resseq,
                "res_name": res_name,
                "res_short_name": PDB.Polypeptide.index_to_one(PDB.Polypeptide.three_to_index(res_name)),
                "all_atoms_abs": float(line[16:22]),
                "all_atoms_rel": float(line[23:28]),
                "side_chain_abs": float(line[29:35]),
                "side_chain_rel": float(line[36:41]),
                "main_chain_abs": float(line[42:48]),
                "main_chain_rel": float(line[49:54]),
                "non_polar_abs": float(line[55:61]),
                "non_polar_rel": float(line[62:67]),
                "all_polar_abs": float(line[68:74]),
                "all_polar_rel": float(line[75:80]),
            }
            seq_idx += 1
        # end of if block: if line.startswith("RES"):
    # end of for loop: for line in rsa_data:
    return naccess_rsa_dict


def process_asa_data(rsa_data):
    """Process the .asa output file: atomic level SASA data."""
    naccess_atom_dict = {}
    for line in rsa_data:
        full_atom_id = line[12:16]
        atom_id = full_atom_id.strip()
        chainid = line[21]
        resseq = int(line[22:26])
        icode = line[26]
        res_id = (" ", resseq, icode)
        id = (chainid, res_id, atom_id)
        asa = line[54:62]  # solvent accessibility in Angstrom^2
        naccess_atom_dict[id] = asa
    return naccess_atom_dict


def validate_rsa_dict(chain_sequences_dict, chain_nm, rsa_dict):
    chain_seq_frm_pdb = chain_sequences_dict[chain_nm]
    chain_seq_formation_lst = []
    for key, value in rsa_dict.items():
        chain_seq_formation_lst.append(value['res_short_name'])
    chain_seq_frm_rsa = ''.join(chain_seq_formation_lst)
    print(f'\nchain_seq_frm_pdb: {chain_seq_frm_pdb}')
    print(f'chain_seq_frm_rsa: {chain_seq_frm_rsa}\n')
    if(chain_seq_frm_pdb != chain_seq_frm_rsa):
        err_msg = f"Error!! Error!! chain_seq_frm_pdb amd chain_seq_frm_rsa are not same.\
              \n chain_seq_frm_pdb: {chain_seq_frm_pdb} \n chain_seq_frm_rsa: {chain_seq_frm_rsa}"
        raise Exception(err_msg)


class NACCESS(AbstractResiduePropertyMap):
    """Define NACCESS class for residue properties map."""

    def __init__(
        self, model, pdb_file=None, naccess_binary="naccess", tmp_directory="/tmp"
    ):
        """Initialize the class."""
        res_data, atm_data = run_naccess(
            model, pdb_file, naccess=naccess_binary, temp_dir=tmp_directory
        )
        naccess_dict = process_rsa_data(res_data)
        property_dict = {}
        property_keys = []
        property_list = []
        for chain in model:
            chain_id = chain.get_id()
            for res in chain:
                res_id = res.get_id()
                if (chain_id, res_id) in naccess_dict:
                    item = naccess_dict[(chain_id, res_id)]
                    res_name = item["res_name"]
                    assert res_name == res.get_resname()
                    property_dict[(chain_id, res_id)] = item
                    property_keys.append((chain_id, res_id))
                    property_list.append((res, item))
                    res.xtra["EXP_NACCESS"] = item
                else:
                    pass
        AbstractResiduePropertyMap.__init__(
            self, property_dict, property_keys, property_list
        )


class NACCESS_atomic(AbstractAtomPropertyMap):
    """Define NACCESS atomic class for atom properties map."""

    def __init__(
        self, model, pdb_file=None, naccess_binary="naccess", tmp_directory="/tmp"
    ):
        """Initialize the class."""
        res_data, atm_data = run_naccess(
            model, pdb_file, naccess=naccess_binary, temp_dir=tmp_directory
        )
        self.naccess_atom_dict = process_asa_data(atm_data)
        property_dict = {}
        property_keys = []
        property_list = []
        for chain in model:
            chain_id = chain.get_id()
            for residue in chain:
                res_id = residue.get_id()
                for atom in residue:
                    atom_id = atom.get_id()
                    full_id = (chain_id, res_id, atom_id)
                    if full_id in self.naccess_atom_dict:
                        asa = self.naccess_atom_dict[full_id]
                        property_dict[full_id] = asa
                        property_keys.append(full_id)
                        property_list.append((atom, asa))
                        atom.xtra["EXP_NACCESS"] = asa
        AbstractAtomPropertyMap.__init__(
            self, property_dict, property_keys, property_list
        )


if __name__ == "__main__":
    import sys
    from Bio.PDB.PDBParser import PDBParser

    p = PDBParser()
    s = p.get_structure("X", sys.argv[1])
    model = s[0]

    n = NACCESS(model, sys.argv[1])
    for e in n:
        """Initialize the class."""
        print(e)