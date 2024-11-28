import sys, os
from pathlib import Path
path_root = Path(__file__).parents[1]  # upto 'codebase' folder
sys.path.insert(0, str(path_root))
# print(sys.path)

from utils import dl_reproducible_result_util
import multiprocessing
import pandas as pd
from utils import PPIPUtils

import gromacs
from matplotlib.ticker import FuncFormatter

# The following line needs to be executed just once after the GromacsWrapper package installation. 
# As we are loading our Gromacs environment by sourcing the GMXRC file ourselves or via module, we do not configure 
# anything and let GromacsWrapper find our Gromacs installation. 
# ## gromacs.config.setup()

# All currently available Gromacs commands are listed in the dictionary gromacs.tools.registry, in particular, gromacs.tools.registry.keys() lists the names.
# ## print(gromacs.tools.registry.keys())


def execute_md_simulation(**kwargs):
    """
    Execute Molecular Dynamics (MD) Simulation on the protein complex.

    Please refer to https://tutorials.gromacs.org/docs/md-intro-tutorial.html for a brief introduction.
    """
    print('\nInside execute_md_simulation() method - Start')
    out = PPIPUtils.execute_unix_command('echo $PATH')
    print(f'echo $PATH:  {out}')

    # Determine the number of available CPU cores
    num_cpu_cores = multiprocessing.cpu_count()

    print('####################################')
    # Iterate over kwargs and raise ValueError if any of the input arguments (except a few) is None. Also print each keyword argument name and respective value.
    for arg_name, arg_value in kwargs.items():
        if(arg_value is None):
            raise ValueError(f"Argument '{arg_name}' must be provided with a value.")
        print(f"'{arg_name}': {arg_value}")
    # end of for loop: for arg_name, arg_value in kwargs.items():

    # retrieve all the keyword arguments
    root_path = kwargs.get('root_path'); dim_prot_complx_nm = kwargs.get('dim_prot_complx_nm')
    cuda_index_mdSim = kwargs.get('cuda_index_mdSim')
    postproc_result_dir = kwargs.get('postproc_result_dir'); pdb_file_location = kwargs.get('pdb_file_location')
    af2_use_amber = kwargs.get('af2_use_amber')
    out_dir_mut_complx_struct_pred_af2 = kwargs.get('out_dir_mut_complx_struct_pred_af2')
    mdSim_overwrite_existing_results = kwargs.get('mdSim_overwrite_existing_results'); forcefield_mdSim = kwargs.get('forcefield_mdSim') 
    max_cadidate_count_mdSim = kwargs.get('max_cadidate_count_mdSim')
    mdSim_result_dir = kwargs.get('mdSim_result_dir'); prot_complx_tag = kwargs.get('prot_complx_tag') 
    specific_inp_pdb_file = kwargs.get('specific_inp_pdb_file') 
    specific_mdSim_result_dir = kwargs.get('specific_mdSim_result_dir') 
    print('####################################\n')

    prot_id = specific_inp_pdb_file.split('/')[-1].replace('.pdb', '')
    gromacs.start_logging(logfile=os.path.join(specific_mdSim_result_dir, f'gromacs_{prot_id}.log'))

    print(f'\n##################{prot_complx_tag} MdSim_Step-1: Cleaning the input structure -Start ##################\n')
    # Strip out all the atoms that do not belong to the protein (i.e. crystal waters, ligands, etc).
    command = f"grep -v HETATM {specific_inp_pdb_file} > {os.path.join(specific_mdSim_result_dir, prot_id + '_temp.pdb')}"
    print(f'executing "{command}"')
    PPIPUtils.execute_unix_command(command)
    
    command = f"grep -v CONECT {os.path.join(specific_mdSim_result_dir, prot_id + '_temp.pdb')} > {os.path.join(specific_mdSim_result_dir, prot_id + '_cleaned.pdb')}"
    print(f'\nexecuting "{command}"')
    PPIPUtils.execute_unix_command(command)
    
    command = f"rm {os.path.join(specific_mdSim_result_dir, prot_id + '_temp.pdb')}"
    print(f'\nexecuting "{command}"')
    PPIPUtils.execute_unix_command(command)
    print(f'\n##################{prot_complx_tag} MdSim_Step-1: Cleaning the input structure -End ##################\n')

    print(f'\n##################{prot_complx_tag} MdSim_Step-2: pdb2gmx (Generating a topology) -Start ##################\n')
    # The purpose of 'gmx pdb2gmx' is to generate three files:
    # 1. The topology for the molecule (topol.top).
    # 2. A position restraint file (posre.itp).
    # 3. A post-processed structure file (conf.gro).

    # Create the output folder for this step
    pdb2gmx_out_dir = os.path.join(specific_mdSim_result_dir, '2_pdb2gmx_out')
    PPIPUtils.createFolder(pdb2gmx_out_dir, recreate_if_exists=mdSim_overwrite_existing_results)

    # Prepare the arguments for the pdb2gmx command execution (https://manual.gromacs.org/current/onlinehelp/gmx-pdb2gmx.html)
    inp_pdb_pdb2gmx = os.path.join(specific_mdSim_result_dir, prot_id + '_cleaned.pdb')
    out_topol_pdb2gmx = os.path.join(pdb2gmx_out_dir, 'topol.top')
    out_itp_pdb2gmx = os.path.join(pdb2gmx_out_dir, 'posre.itp')
    out_gro_pdb2gmx = os.path.join(pdb2gmx_out_dir, 'processed.gro')
    # Execute pdb2gmx command (ForceField(ff) could be "amber99sb-ildn" or "charmm27")
    gromacs.pdb2gmx(f=inp_pdb_pdb2gmx, o=out_gro_pdb2gmx, p=out_topol_pdb2gmx, i=out_itp_pdb2gmx, ff=forcefield_mdSim, water="tip3p", ignh=True)
    print(f'\n##################{prot_complx_tag} MdSim_Step-2: pdb2gmx (Generating a topology) -End ##################\n')
    
    print(f'\n##################{prot_complx_tag} MdSim_Step-3: Solvating the simulation system -Start ##################\n')
    # We are going to be simulating a simple aqueous system. 
    # There are two steps to define the box and fill it with solvent (water):
    # a) Define the box dimensions using the `gmx editconf <https://manual.gromacs.org/current/onlinehelp/gmx-editconf.html>`.
    # b) Fill the box with water using the `gmx solvate <https://manual.gromacs.org/current/onlinehelp/gmx-solvate.html>`.

    print('######################### Configure the simulation box #########################')
    # Create the output folder for this step
    editconf_out_dir = os.path.join(specific_mdSim_result_dir, '3a_editconf_out')
    PPIPUtils.createFolder(editconf_out_dir, recreate_if_exists=mdSim_overwrite_existing_results)

    # Prepare the arguments for the editconf command execution
    inp_gro_editconf = out_gro_pdb2gmx
    out_gro_editconf = os.path.join(editconf_out_dir, 'boxed.gro')
    # Execute editconf command.
    # The below command centers the protein in the box (-c), and places it at least 1.0 nm from the box edge (-d 1.0). 
    # The box type is defined as a rhombic dodecahedron (-bt dodecahedron). 
    gromacs.editconf(f=inp_gro_editconf, o=out_gro_editconf, bt="dodecahedron", d=1.0, c=True, input="Protein")

    print('######################### Fill the box with water #########################')
    # Create the output folder for this step
    solvate_out_dir = os.path.join(specific_mdSim_result_dir, '3b_solvate_out')
    PPIPUtils.createFolder(solvate_out_dir, recreate_if_exists=mdSim_overwrite_existing_results)

    # Prepare the arguments for the solvate command execution
    inp_gro_solvate = out_gro_editconf
    out_gro_solvate = os.path.join(solvate_out_dir, 'solv.gro')
    # Execute solvate command
    # The configuration of the protein (-cp) is contained in the output of the previous gmx editconf step, and 
    # the configuration of the solvent (-cs) is part of the standard GROMACS installation. We are using spc216.gro, which is 
    # a generic equilibrated 3-point solvent model box. The output (-o) is called solvated.gro, and we tell the solvate the name 
    # of the topology file (-p) so it can be modified.
    gromacs.solvate(cp=inp_gro_solvate, cs="spc216.gro", p=out_topol_pdb2gmx, o=out_gro_solvate)
    print(f'\n##################{prot_complx_tag} MdSim_Step-3: Solvating the simulation system -End ##################\n')
    
    print(f'\n##################{prot_complx_tag} MdSim_Step-4: Adding Ions -Start ##################\n')
    # We now have a solvated system that contains a charged protein. The output of pdb2gmx told us that the protein has a net charge.
    # Since life does not exist at a net charge, we must add ions to our system. Further, we aim to approximate physiological conditions and 
    # use therefore a NaCl concentration of 0.15 M.

    # The tool for adding ions within GROMACS is called `gmx genion <https://manual.gromacs.org/current/onlinehelp/gmx-genion.html>`
    # What gmx genion does is read through the topology and replace water molecules with the ions that the user specifies. 
    # The input is a .tpr file created using `gmx grompp <https://manual.gromacs.org/current/onlinehelp/gmx-grompp.html>`

    # To produce a .tpr file with `gmx grompp`, we will need an additional input file, with the extension .mdp (molecular dynamics parameter file); 
    # gmx grompp will assemble the parameters specified in the .mdp file with the coordinates and topology information to generate a .tpr file.
    
    # Create the output folder for this step
    ions_out_dir = os.path.join(specific_mdSim_result_dir, '4_ions_out')
    PPIPUtils.createFolder(ions_out_dir, recreate_if_exists=mdSim_overwrite_existing_results)

    # Prepare the arguments for the grompp command execution
    inp_mdp_grompp_ion = os.path.join(root_path, 'codebase/utils/md_utils/mdp_files', 'ions.mdp')
    out_tpr_grompp_ion = os.path.join(ions_out_dir, 'ions.tpr')
    # Execute grompp command
    gromacs.grompp(f=inp_mdp_grompp_ion, o=out_tpr_grompp_ion, c=out_gro_solvate, p=out_topol_pdb2gmx, maxwarn=5)

    # Prepare the arguments for the genion command execution
    out_gro_genion = os.path.join(ions_out_dir, 'solv_ions.gro')
    # Execute genion command (We chose group “SOL” for embedding ions).
    gromacs.genion(s=out_tpr_grompp_ion, o=out_gro_genion, conc=0.15, p=out_topol_pdb2gmx, pname="NA", nname="CL", neutral=True, input="SOL")  # 
    print(f'\n##################{prot_complx_tag} MdSim_Step-4: Adding Ions -End ##################\n')

    print(f'\n##################{prot_complx_tag} MdSim_Step-5: Energy minimisation (EM) -Start ##################\n')
    # To perform energy minimization, we are once again going to use gmx grompp to assemble the structure (.gro)
    # , topology (.top), and simulation parameters (.mdp) into a binary input file (.tpr), then 
    # we will use GROMACS MD engine, mdrun, to run the energy minimization.

    # Create the output folder for this step
    em_out_dir = os.path.join(specific_mdSim_result_dir, '5_em_out')
    PPIPUtils.createFolder(em_out_dir, recreate_if_exists=mdSim_overwrite_existing_results)

    # Prepare the arguments for the grompp command execution
    inp_mdp_grompp_em = None
    if(forcefield_mdSim.startswith('amber')):
        inp_mdp_grompp_em = os.path.join(root_path, 'codebase/utils/md_utils/mdp_files', 'emin-amber.mdp')
    else:
        inp_mdp_grompp_em = os.path.join(root_path, 'codebase/utils/md_utils/mdp_files', 'emin-charmm.mdp')

    out_tpr_grompp_em = os.path.join(em_out_dir, 'em.tpr')
    # Execute grompp command
    gromacs.grompp(f=inp_mdp_grompp_em, o=out_tpr_grompp_em, c=out_gro_genion, p=out_topol_pdb2gmx, maxwarn=5)

    # Prepare the arguments for the mdrun command execution
    # Once run, we will find the energy-minimized structure in a file called em.gro. Additionally to this we will find more information 
    # on the run in an ASCII-text log file of the EM process, em.log, a file for storage of energy, em.edr and a binary full-precision trajectory em.trr.
    inp_tpr_mdrun_em = out_tpr_grompp_em
    out_gro_mdrun_em = os.path.join(em_out_dir, 'em.gro')
    out_log_mdrun_em = os.path.join(em_out_dir, 'em.log')
    out_edr_mdrun_em = os.path.join(em_out_dir, 'em.edr')
    out_trr_mdrun_em = os.path.join(em_out_dir, 'em.trr')

    # Execute mdrun command (https://manual.gromacs.org/current/onlinehelp/gmx-mdrun.html)
    # 'v' for verbose.
    gromacs.mdrun(v=True, pin='on', gpu_id=cuda_index_mdSim, s=inp_tpr_mdrun_em, o=out_trr_mdrun_em, c=out_gro_mdrun_em, e=out_edr_mdrun_em, g=out_log_mdrun_em)
    print(f'\n##################{prot_complx_tag} MdSim_Step-5: Energy minimisation (EM) -End ##################\n')

    print(f'\n##################{prot_complx_tag} MdSim_Step-6: Equilibration run - temperature (NVT) -Start ##################\n')
    # EM ensured that we have a reasonable starting structure, in terms of geometry and solvent orientation. Now the system needs to be 
    # brought to the temperature we wish to simulate and establish the proper orientation about the solute (the protein). 
    # After we arrive at the correct temperature (based on kinetic energies), we will apply pressure to the system until it reaches the proper density.
    # Equilibration is often conducted in two phases. The first phase is conducted under an NVT ensemble (constant Number of particles, Volume, and Temperature). 

    # Create the output folder for this step
    nvt_out_dir = os.path.join(specific_mdSim_result_dir, '6_nvt_out')
    PPIPUtils.createFolder(nvt_out_dir, recreate_if_exists=mdSim_overwrite_existing_results)

    # We will call grompp and mdrun just as we did at the EM step, but this time with the energy minimised structure as input and 
    # a different .mdp file for the run.
    # Prepare the arguments for the grompp command execution
    inp_mdp_grompp_nvt = None
    if(forcefield_mdSim.startswith('amber')):
        inp_mdp_grompp_nvt = os.path.join(root_path, 'codebase/utils/md_utils/mdp_files', 'nvt-amber.mdp')
    else:
        inp_mdp_grompp_nvt = os.path.join(root_path, 'codebase/utils/md_utils/mdp_files', 'nvt-charmm.mdp')

    out_tpr_grompp_nvt = os.path.join(nvt_out_dir, 'nvt.tpr')
    # Execute grompp command
    gromacs.grompp(f=inp_mdp_grompp_nvt, o=out_tpr_grompp_nvt, c=out_gro_mdrun_em, r=out_gro_mdrun_em
                           , p=out_topol_pdb2gmx, maxwarn=5)

    # Prepare the arguments for the mdrun command execution
    inp_tpr_mdrun_nvt = out_tpr_grompp_nvt
    out_gro_mdrun_nvt = os.path.join(nvt_out_dir, 'nvt.gro')
    out_log_mdrun_nvt = os.path.join(nvt_out_dir, 'nvt.log')
    out_edr_mdrun_nvt = os.path.join(nvt_out_dir, 'nvt.edr')
    out_cpt_mdrun_nvt = os.path.join(nvt_out_dir, 'nvt.cpt')
    # Execute mdrun command (https://manual.gromacs.org/current/onlinehelp/gmx-mdrun.html)
    # 'v' for verbose.
    gromacs.mdrun(v=True, pin='on', gpu_id=cuda_index_mdSim, s=inp_tpr_mdrun_nvt, cpo=out_cpt_mdrun_nvt, c=out_gro_mdrun_nvt
                          , e=out_edr_mdrun_nvt, g=out_log_mdrun_nvt)
    print(f'\n##################{prot_complx_tag} MdSim_Step-6: Equilibration run - temperature (NVT) -End ##################\n')

    print(f'\n##################{prot_complx_tag} MdSim_Step-7: Equilibration run - pressure (NPT) -Start ##################\n')
    # The previous step, NVT equilibration, stabilized the temperature of the system. Prior 
    # to data collection, we must also stabilize the pressure (and thus also the density) of the system. Equilibration 
    # of pressure is conducted under an NPT ensemble, wherein the Number of particles, Pressure, and Temperature are 
    # all constant. 

    # Create the output folder for this step
    npt_out_dir = os.path.join(specific_mdSim_result_dir, '7_npt_out')
    PPIPUtils.createFolder(npt_out_dir, recreate_if_exists=mdSim_overwrite_existing_results)

    # We will call grompp and mdrun just as we did for NVT equilibration. Note that we are 
    # now including the -t flag to include the checkpoint file (.cpt) from the NVT equilibration; 
    # this file contains all the necessary state variables to continue our simulation. To conserve 
    # the velocities produced during NVT, we must include the final coordinate file (.gro) of the NVT simulation using the option (-c).
    
    # Prepare the arguments for the grompp command execution
    inp_mdp_grompp_npt = None
    if(forcefield_mdSim.startswith('amber')):
        inp_mdp_grompp_npt = os.path.join(root_path, 'codebase/utils/md_utils/mdp_files', 'npt-amber.mdp')
    else:
        inp_mdp_grompp_npt = os.path.join(root_path, 'codebase/utils/md_utils/mdp_files', 'npt-charmm.mdp')

    out_tpr_grompp_npt = os.path.join(npt_out_dir, 'npt.tpr')
    # Execute grompp command
    gromacs.grompp(f=inp_mdp_grompp_npt, o=out_tpr_grompp_npt, c=out_gro_mdrun_nvt
                           , r=out_gro_mdrun_nvt, t=out_cpt_mdrun_nvt, p=out_topol_pdb2gmx, maxwarn=5)

    # Prepare the arguments for the mdrun command execution
    inp_tpr_mdrun_npt = out_tpr_grompp_npt
    out_gro_mdrun_npt = os.path.join(npt_out_dir, 'npt.gro')
    out_log_mdrun_npt = os.path.join(npt_out_dir, 'npt.log')
    out_edr_mdrun_npt = os.path.join(npt_out_dir, 'npt.edr')
    out_cpt_mdrun_npt = os.path.join(npt_out_dir, 'npt.cpt')
    # Execute mdrun command (https://manual.gromacs.org/current/onlinehelp/gmx-mdrun.html)
    # 'v' for verbose.
    gromacs.mdrun(v=True, pin='on', gpu_id=cuda_index_mdSim, s=inp_tpr_mdrun_npt, cpo=out_cpt_mdrun_npt, c=out_gro_mdrun_npt
                          , e=out_edr_mdrun_npt, g=out_log_mdrun_npt)
    print(f'\n##################{prot_complx_tag} MdSim_Step-7: Equilibration run - pressure (NPT) -End ##################\n')

    print(f'\n\n##################{prot_complx_tag} MdSim_Step-8: ***The “production” run*** -Start ##################\n')
    # Upon completion of the two equilibration phases, the system is now well-equilibrated at the desired temperature and pressure. 
    # We are now ready to release the position restraints and run production MD for data collection. 

    # Create the output folder for this step
    prod_out_dir = os.path.join(specific_mdSim_result_dir, '8_prod_out')
    PPIPUtils.createFolder(prod_out_dir, recreate_if_exists=mdSim_overwrite_existing_results)

    # We will call grompp and mdrun just as we did before. Note that we are 
    # now including the -t flag to include the checkpoint file (.cpt) from the NPT equilibration; 
    # Note we have explictly add a section in .mdp file that controls the output frequency in log file (.log), 
    # energy file (.edr), in the trajcotry file (.trr) and the compress trajectory file (.xtc).
    
    # Prepare the arguments for the grompp command execution
    inp_mdp_grompp_prod = None
    if(forcefield_mdSim.startswith('amber')):
        inp_mdp_grompp_prod = os.path.join(root_path, 'codebase/utils/md_utils/mdp_files', 'prod-amber.mdp')
    else:
        inp_mdp_grompp_prod = os.path.join(root_path, 'codebase/utils/md_utils/mdp_files', 'prod-charmm.mdp')

    out_tpr_grompp_prod = os.path.join(prod_out_dir, 'prod.tpr')
    # Execute grompp command
    gromacs.grompp(f=inp_mdp_grompp_prod, o=out_tpr_grompp_prod, c=out_gro_mdrun_npt
                           , t=out_cpt_mdrun_npt, p=out_topol_pdb2gmx, maxwarn=5)

    # Prepare the arguments for the mdrun command execution
    inp_tpr_mdrun_prod = out_tpr_grompp_prod
    out_gro_mdrun_prod = os.path.join(prod_out_dir, 'prod.gro')
    out_log_mdrun_prod = os.path.join(prod_out_dir, 'prod.log')
    out_edr_mdrun_prod = os.path.join(prod_out_dir, 'prod.edr')
    out_cpt_mdrun_prod = os.path.join(prod_out_dir, 'prod.cpt')
    out_xtc_mdrun_prod = os.path.join(prod_out_dir, 'prod.xtc')
    # Execute mdrun command (https://manual.gromacs.org/current/onlinehelp/gmx-mdrun.html)
    # 'v' for verbose.
    gromacs.mdrun(v=True, pin='on', gpu_id=cuda_index_mdSim, s=inp_tpr_mdrun_prod
                          , cpo=out_cpt_mdrun_prod, cpi=out_cpt_mdrun_prod, c=out_gro_mdrun_prod
                          , e=out_edr_mdrun_prod, g=out_log_mdrun_prod, x=out_xtc_mdrun_prod)
    print(f'\n##################{prot_complx_tag} MdSim_Step-8: ***The “production” run*** -End ##################\n\n')

    print(f'\n##################{prot_complx_tag} MdSim_Step-9: Result analysis -Start ##################\n\n')
    # Create the output folder for this step
    analysis_out_dir = os.path.join(specific_mdSim_result_dir, '9_analysis_out')
    PPIPUtils.createFolder(analysis_out_dir, recreate_if_exists=mdSim_overwrite_existing_results)
    
    print(f'\n##################{prot_complx_tag} MdSim_Step-9A: Peforming "gmx trjconv"  -Start ##################\n\n')
    # The first tool for analysis is `gmx trjconv <https://manual.gromacs.org/current/onlinehelp/gmx-trjconv.html>, which is used 
    # as a post-processing tool to strip out coordinates, correct for periodicity, or manually alter the trajectory (time units, frame frequency, etc).

    # Prepare the arguments for the trjconv command execution
    inp_tpr_trjconv_analysis = out_tpr_grompp_prod
    inp_xtc_trjconv_analysis = out_xtc_mdrun_prod
    out_xtc_trjconv_analysis = os.path.join(analysis_out_dir, 'prod_center.xtc')
    # Execute trjconv command
    gromacs.trjconv(input=('Protein', 'Protein'), s=inp_tpr_trjconv_analysis, f=inp_xtc_trjconv_analysis, o=out_xtc_trjconv_analysis, center=True, pbc="mol")
    print(f'\n##################{prot_complx_tag} MdSim_Step-9A: Peforming "gmx trjconv"  -End ##################\n\n')

    print(f'\n##################{prot_complx_tag} MdSim_Step-9B: Peforming "gmx rms" (RMSD calculation) -Start ##################\n\n')
    # Now let’s look at structural stability. GROMACS has a built-in utility for RMSD calculations called `gmx rms <https://manual.gromacs.org/current/onlinehelp/gmx-rms.html>`
    # Choose 4 (“Backbone”) for both the least-squares fit and the group for RMSD calculation. 
    # The -tu flag will output the results in terms of ns, even though the trajectory was written in ps. This is done for 
    # clarity of the output (especially if you have a long simulation). 
    # The output plot will show the RMSD relative to the structure present in the original pdb file
    
    # Create the output folder for this step
    rmsd_analysis_out_dir = os.path.join(analysis_out_dir, 'rmsd')
    PPIPUtils.createFolder(rmsd_analysis_out_dir, recreate_if_exists=mdSim_overwrite_existing_results)

    # Prepare the arguments for the rms command execution
    inp_tpr_rms_analysis = out_tpr_grompp_prod
    inp_xtc_rms_analysis = out_xtc_trjconv_analysis
    out_xvg_rms_analysis = os.path.join(rmsd_analysis_out_dir, 'rmsd.xvg')
    # Execute rms command
    gromacs.rms(input=("Backbone", "Backbone"), s=inp_tpr_rms_analysis, f=inp_xtc_rms_analysis, o=out_xvg_rms_analysis, tu='ns', what='rmsd', xvg='none')
    # generate and save dataframe
    rms_df = pd.read_csv(out_xvg_rms_analysis, sep='\s+', header=None, names=['time(ns)','RMSD(nm)'])
    # calculate average rmsd (in nm) 
    avg_rmsd_nm = round(rms_df['RMSD(nm)'].mean(), ndigits=3)
    rms_df.to_csv(os.path.join(rmsd_analysis_out_dir, 'rmsd.csv'), index=False)
    # generate and save plot
    rms_plt = rms_df.plot('time(ns)')  # rms_plt is a matplotlib.axes.AxesSubplot object.
    rms_plt.set(xlabel="time(ns)", ylabel="RMSD(nm)")
    rms_plt.figure.savefig(os.path.join(rmsd_analysis_out_dir, 'rmsd.png'))
    print(f'\n##################{prot_complx_tag} MdSim_Step-9B: Peforming "gmx rms" (RMSD calculation) -End ##################\n\n')

    print(f'\n##################{prot_complx_tag} MdSim_Step-9C: Peforming "gmx gyrate" (Radius of gyration) -Start ##################\n\n')
    # The radius of gyration of a protein is a measure of its compactness. If a protein is stably folded, it will likely maintain 
    # a relatively steady value of Rg. If a protein unfolds, its Rg will change over time. Let’s analyze the radius of gyration for 
    # the protein in our simulation using GROMACS `gmx gyrate <https://manual.gromacs.org/current/onlinehelp/gmx-gyrate.html>`
    
    # Create the output folder for this step
    rg_analysis_out_dir = os.path.join(analysis_out_dir, 'rg')
    PPIPUtils.createFolder(rg_analysis_out_dir, recreate_if_exists=mdSim_overwrite_existing_results)

    # Prepare the arguments for the rg command execution
    inp_tpr_rg_analysis = out_tpr_grompp_prod
    inp_xtc_rg_analysis = out_xtc_trjconv_analysis
    out_xvg_rg_analysis = os.path.join(rg_analysis_out_dir, 'rg.xvg')
    # Execute gyrate command
    gromacs.gyrate(input='Protein', s=inp_tpr_rg_analysis, f=inp_xtc_rg_analysis, o=out_xvg_rg_analysis, xvg='none')
    # generate and save dataframe
    rg_df = pd.read_csv(out_xvg_rg_analysis, sep='\s+', header=None, names=['time','rg(nm)', 'rg_x', 'rg_y', 'rg_z'], usecols=[0, 1, 2, 3, 4])
    # calculate average rg (in nm) 
    avg_rg_nm = round(rg_df['rg(nm)'].mean(), ndigits=3)
    rg_df.to_csv(os.path.join(rg_analysis_out_dir, 'rg.csv'), index=False)
    # generate and save plot
    rg_plt = rg_df.plot(x='time', y='rg(nm)')  # rg_plt is a matplotlib.axes.AxesSubplot object.
    rg_plt.set(xlabel="time", ylabel="Rg(nm)")
    rg_plt.figure.savefig(os.path.join(rg_analysis_out_dir, 'rg.png'))
    print(f'\n##################{prot_complx_tag} MdSim_Step-9C: Peforming "gmx gyrate" (Radius of gyration) -End ##################\n\n')

    print(f'\n##################{prot_complx_tag} MdSim_Step-9D: Peforming "gmx rmsf" (Root mean square fluctuation) -Start ##################\n\n')
    # gmx rmsf computes the root mean square fluctuation (RMSF, i.e. standard deviation) of atomic positions in
    # the trajectory (supplied with -f) after (optionally) fitting to a reference frame (supplied with -s).
    # https://manual.gromacs.org/current/onlinehelp/gmx-rmsf.html
    # The difference between RMSD and RMSF is given in 
    # 1. https://www.researchgate.net/post/What_is_the_major_difference_between_RMSD_and_RMSF_analysis
    # 2. https://www.compchems.com/how-to-compute-the-rmsf-using-gromacs/#the-gmx-rmsf-command
    
    # Create the output folder for this step
    rmsf_analysis_out_dir = os.path.join(analysis_out_dir, 'rmsf')
    PPIPUtils.createFolder(rmsf_analysis_out_dir, recreate_if_exists=mdSim_overwrite_existing_results)

    # Prepare the arguments for the rmsf command execution
    inp_tpr_rmsf_analysis = out_tpr_grompp_prod
    inp_xtc_rmsf_analysis = out_xtc_trjconv_analysis
    out_xvg_rmsf_analysis = os.path.join(rmsf_analysis_out_dir, 'rmsf.xvg')
    # Execute rmsf command
    gromacs.rmsf(input=("Backbone", "Backbone"), s=inp_tpr_rmsf_analysis, f=inp_xtc_rmsf_analysis, o=out_xvg_rmsf_analysis, xvg='none', res='yes')
    # generate and save dataframe
    rmsf_df = pd.read_csv(out_xvg_rmsf_analysis, sep='\s+', header=None, names=['chain_res','rmsf(nm)'], usecols=[0, 1])
    rmsf_df = rmsf_df.reset_index(drop=False, names='residue')
    # calculate average rmsf (in nm) 
    avg_rmsf_nm = round(rmsf_df['rmsf(nm)'].mean(), ndigits=3)
    rmsf_df.to_csv(os.path.join(rmsf_analysis_out_dir, 'rmsf.csv'), index=False)
    # generate and save plot
    rmsf_plt = rmsf_df.plot(x='residue', y='rmsf(nm)')  # rmsf_plt is a matplotlib.axes.AxesSubplot object.
    rmsf_plt.set(xlabel="Residue", ylabel="RMSF(nm)")
    rmsf_plt.figure.savefig(os.path.join(rmsf_analysis_out_dir, 'rmsf.png'))
    print(f'\n##################{prot_complx_tag} MdSim_Step-9D: Peforming "gmx rmsf" (Root mean square fluctuation) -End ##################\n\n')

    print(f'\n##################{prot_complx_tag} MdSim_Step-9: Result analysis -End ##################\n\n')
    gromacs.stop_logging()
    print('inside execute_md_simulation() method - End')
    return (avg_rmsd_nm, avg_rg_nm, avg_rmsf_nm)


# Custom formatter function for X,Y-axis
def custom_format(val, pos):
    if val == 0.0:
        return '0'  # Display 0.0 as 0
    else:
        return '{:.1f}'.format(val)  # Keep other ticks in float format
