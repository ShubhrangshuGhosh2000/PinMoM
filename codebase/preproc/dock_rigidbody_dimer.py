import os
import sys
from pathlib import Path

path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

import pandas as pd
import urllib.request
from utils import PPIPUtils


def prepare_rigidbody_dimer(root_path='./'):
    print('inside prepare_rigidbody_dimer() method - Start')
    # read Hwang's Docking Benchmark 5.5 specific to Rigid-Body
    dock55_rigidbody_csv_fl_nm_with_path = os.path.join(root_path, 'dataset/orig_data', 'Table_BM5.5_rigidbody.csv')
    rigidbody_df = pd.read_csv(dock55_rigidbody_csv_fl_nm_with_path)
    # Filter the DataFrame based on the specified pattern
    # the regular expression r'^\w+_[A-Z]:[A-Z]$' is used to match the pattern where 'Complex' values consist
    # of one or more alphanumeric characters for the name, followed by an underscore, then exactly one
    # letter, followed by a colon, and finally, exactly one more letter
    dimeric_prot_complex_df = rigidbody_df[rigidbody_df['Complex'].str.match(r'^\w+_[A-Z]:[A-Z]$')]
    # save dimeric_prot_complex_df as csv
    dimeric_prot_complex_csv_name = os.path.join(root_path, 'dataset/preproc_data', 'dimeric_prot_complex.csv')
    dimeric_prot_complex_df.to_csv(dimeric_prot_complex_csv_name, index=False)
    print('inside prepare_rigidbody_dimer() method - End')


def download_rigidbody_dimer_pdb(root_path='./'):
    print('inside download_rigidbody_dimer_pdb() method - Start')
    # The directory to save the downloaded PDB file.
    pdb_file_location = os.path.join(root_path, "dataset/preproc_data/pdb_files")
    PPIPUtils.createFolder(pdb_file_location)

    # read dimeric_prot_complex.csv file 
    dimeric_prot_complex_csv_name = os.path.join(root_path, 'dataset/preproc_data', 'dimeric_prot_complex.csv')
    dimeric_prot_complex_df = pd.read_csv(dimeric_prot_complex_csv_name)
    # iterate over the dimeric_prot_complex_df and download the respecive pdb files 
    for index, row in dimeric_prot_complex_df.iterrows():
        print(f'Downloading {index +1}-th PDB file out of {dimeric_prot_complex_df.shape[0]}')
        complex_nm = row['Complex']
        complx_prot_id = complex_nm.split('_')[0]
        # Define the PDB file path in the specified location
        pdb_file_path = os.path.join(pdb_file_location, f"{complx_prot_id}.pdb")
        # Check if the PDB file exists in the specified location
        if not os.path.isfile(pdb_file_path):
            # If not found, download the PDB file and save it
            pdb_url = f"https://files.rcsb.org/download/{complx_prot_id}.pdb"
            try:
                urllib.request.urlretrieve(pdb_url, pdb_file_path)
            except Exception as e:
                return f"Error downloading PDB file: {e}"
        # end of if block: if not os.path.isfile(pdb_file_path):
    # end of for loop: for index, row in dimeric_prot_complex_df.iterrows():
    print('inside download_rigidbody_dimer_pdb() method - End')


if __name__ == '__main__':
    root_path = os.path.join('/project/root/directory/path/here')

    # prepare_rigidbody_dimer(root_path)
    download_rigidbody_dimer_pdb(root_path)