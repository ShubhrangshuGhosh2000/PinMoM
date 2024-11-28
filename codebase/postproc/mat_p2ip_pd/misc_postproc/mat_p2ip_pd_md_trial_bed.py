import os, sys

from pathlib import Path

import gromacs.core
path_root = Path(__file__).parents[2]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)


# ## module load apps/gromacs/2022/gpu
# ## export PATH='/home/pralaycs/miniconda3/envs/py3114_torch_gpu_param/bin':$PATH
# ## source /home/apps/gromacs/gromacs-2022.2/installGPUIMPI/bin/GMXRC

import gromacs
# gromacs.config.setup()
# print(gromacs.tools.registry.keys())
print(gromacs.release)


# ## you may have to run NPT equilibration slightly longer than is specified here.
# # ## For the 'Production Run', The recommendation is to use tau_t = 1 ps for V-rescale and tau_p = 5 ps for C-rescale.
# ; Pressure coupling is on
# pcoupl                  = C-rescale               ; Pressure coupling on in NPT
# pcoupltype              = isotropic             ; uniform scaling of box vectors
# tau_p                   = 5.0                   ; time constant, in ps
# ref_p                   = 1.0                   ; reference pressure, in bar
# compressibility         = 4.5e-5                ; isothermal compressibility of water, bar^-1 

def sample_md_run():
    root_path = os.path.join('/scratch/pralaycs/Shubh_Working_Remote/PPI_Wkspc/PPI_Code/matpip_pd_prj')
    pdb_file_location = os.path.join(root_path, "dataset/preproc_data/pdb_files") 
    prot_id = '1fjs_protein' # '2jof', '1fjs'

    # generating gmx input files
    inp_pdb = os.path.join(pdb_file_location, f'{prot_id}.pdb')
    output_topol = os.path.join(root_path, 'dataset/postproc_data/dm_result/1fjs/gmx_inp', 'topol.top')
    output_itp = os.path.join(root_path, 'dataset/postproc_data/dm_result/1fjs/gmx_inp', 'posre.itp')
    output_gro = os.path.join(root_path, 'dataset/postproc_data/dm_result/1fjs/gmx_inp', 'protein.gro')
    gromacs.pdb2gmx_GPUIMPI(f=inp_pdb, o=output_gro, p=output_topol, i=output_itp, ff="amber99sb-ildn", water="tip3p", ignh=True)
    print('hello')


if __name__ == '__main__':
    sample_md_run()
