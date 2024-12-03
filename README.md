# PinMoM
Official repository of the paper "_Enhancing protein interaction design through Monte Carlo Simulation with learning-based interaction score, AlphaFold, and KAN-based Positive-Unlabeled learning_". We are in the process of checking in the relevant files.

### TODO
- Make it like MaTPIP.

### TO BE PUT IN THE FINAL VERSION OF THE REPO
- All experiments were conducted using a runtime infrastructure that utilizes a single machine equipped with 187 GB of RAM, a 16 GB GPU (Nvidia Tesla V100), and an Intel Xeon Gold 6148 CPU @ 2.40 GHz. The selection of machines for each experiment is based on the availability of a cluster with similar machine specifications.

- We utilized a conda environment with Python 3.11.4 for code execution. The environment was created using the [py3114_torch_gpu_param.yml](https://github.com/ShubhrangshuGhosh2000/PinMoM/blob/main/py3114_torch_gpu_param.yml) file.

- To generate designed sequences, please refer to the [mat_p2ip_pd_trigger_mcmc.py](https://github.com/ShubhrangshuGhosh2000/PinMoM/tree/main/codebase/proc/mat_p2ip_pd/mcmc/mat_p2ip_pd_trigger_mcmc.py) file. For selecting designed sequences, refer to the [mat_p2ip_pd_postproc_trigger_mcmc.py](https://github.com/ShubhrangshuGhosh2000/PinMoM/blob/main/codebase/postproc/mat_p2ip_pd/mcmc/mat_p2ip_pd_postproc_trigger_mcmc.py) file.

- To verify the selected designed sequences, we utilized AlphaFold2 and Molecular Dynamics simulations. For running AlphaFold2, we used [LocalColabFold](https://github.com/YoshitakaMo/localcolabfold?tab=readme-ov-file); please refer to their website for guidance on its execution. For Molecular Dynamics simulations, we employed [GROMACS](https://www.gromacs.org/); detailed instructions can be found on their website.

- For KAN-based Positive Unlabeled learning, please refer [pul_train_evaluate.py](https://github.com/ShubhrangshuGhosh2000/PinMoM/blob/main/codebase/postproc/mat_p2ip_pd/pul/proc_pul/pul_train_evaluate.py).

- We explored multiple approaches while implementing our ideas, which resulted in a codebase that initially included numerous options, only a few of which were ultimately selected. We made an effort to clean up the codebase as thoroughly as possible before uploading. If you encounter any issues, please let us know.



