class MutationNumberAdjuster():
    """
    Class for dynamic adjustment of number of mutation points based on the probability of interaction and the stage of simulation. 
 
    Args:
    prob_threshold_for_mut_num_adj_mna (float): Threshold value for the PPI probability metric. If this threshold is reached then 'fixed mutation number' strategy will not be applied and Early Stopping Criterion check will be enabled.
    num_of_itr (int): Total number of iterations in the simulation.
    batch_size (int): Batch size for the current simulation.
    num_of_stages_mna (int): Number of stages in which the entire simulation process will be divided w.r.t. the total number of batches. For example, if total number of simulations=30k, batch_size=5 and num_of_stages_mna=6 implies the entire simulation will be broken into 6 different ranges w.r.t. the total number of batches and they are [0-1k), [1k-2k), [2k -3k), [3k -4k), [4k - 5k] and [5k - 6k).
    fixed_mut_num_trigg_stage_mna (int): Stage number from which fixed number of mutations will take place. For example, if 'fixed_mut_num_trigg_stage_mna' = 4 in the above example, then from batch index 3k onwards fixed number of mutations will take place.
    fixed_mut_num_mna (int): The integer indicating the fixed number of mutations which will take place from 'fixed_mut_num_trigg_stage_mna' onwards.
    """

    def __init__(self, prob_threshold_for_mut_num_adj_mna=0.8, num_of_itr=30000, batch_size=5, num_of_stages_mna=6, fixed_mut_num_trigg_stage_mna=4, fixed_mut_num_mna=1):
        """
        Initialize MutationNumberAdjuster instance.

        Args:
        prob_threshold_for_mut_num_adj_mna (float): Threshold value for the PPI probability metric. If this threshold is reached then 'fixed mutation number' strategy will not be applied and Early Stopping Criterion check will be enabled.
        num_of_itr (int): Total number of iterations in the simulation.
        batch_size (int): Batch size for the current simulation.
        num_of_stages_mna (int): Number of stages in which the entire simulation process will be divided w.r.t. the total number of batches. For example, if total number of simulations=30k, batch_size=5 and num_of_stages_mna=6 implies the entire simulation will be broken into 6 different ranges w.r.t. the total number of batches and they are [0-1k), [1k-2k), [2k -3k), [3k -4k), [4k - 5k] and [5k - 6k).
        fixed_mut_num_trigg_stage_mna (int): Stage number from which fixed number of mutations will take place. For example, if 'fixed_mut_num_trigg_stage_mna' = 4 in the above example, then from batch index 3k onwards fixed number of mutations will take place.
        fixed_mut_num_mna (int): The integer indicating the fixed number of mutations which will take place from 'fixed_mut_num_trigg_stage_mna' onwards.
        """
        self.prob_threshold_for_mut_num_adj_mna = prob_threshold_for_mut_num_adj_mna
        self.num_of_itr = num_of_itr
        self.num_of_stages_mna = num_of_stages_mna
        self.batch_size = batch_size
        self.fixed_mut_num_mna = fixed_mut_num_mna
        self.fixed_mut_num_trigg_stage_mna = fixed_mut_num_trigg_stage_mna
        # initialize following variables
        self.best_ppi = 0
        self.prv_percent_len_for_calc_mut_pts = 10
        self.prv_running_stage = 1
    # end of __init__() method


    def adjust_mutation_number(self, crnt_prob_value=0.0, crnt_batch_indx=0):
        """
        Adjust the mutation number based on the probability of interaction and the stage of simulation. 

        Args:
        crnt_prob_value (float): Probability value for the current batch.

        Returns:
        A tuple consisting of an integer (updated value of percent_len_for_calc_mut_pts or fixed_mut_num_mna) and a boolean flag (whether the fixed mutation number will be triggered) and a boolean flag (indicating whether Early Stopping Criterion will be enabled or not)
        """
        enable_early_stopping_check = True
        trigger_fixed_mut_num_mna = False

        if(crnt_prob_value > self.best_ppi):
            self.best_ppi = crnt_prob_value

        total_number_of_batches = self.num_of_itr // self.batch_size
        num_of_batches_per_stage = total_number_of_batches // self.num_of_stages_mna
        print(f'\nadjust_mutation_number(): \n self.best_ppi: {self.best_ppi} :: crnt_prob_value: {crnt_prob_value}') 
        print(f'crnt_batch_indx: {crnt_batch_indx} :: total_number_of_batches: {total_number_of_batches} :: num_of_batches_per_stage: {num_of_batches_per_stage}')
        print(f'self.prv_percent_len_for_calc_mut_pts: {self.prv_percent_len_for_calc_mut_pts} :: fixed_mut_num_trigg_stage_mna: {self.fixed_mut_num_trigg_stage_mna} :: self.prv_running_stage: {self.prv_running_stage}')

        # First check whether 'fixed mutation number' strategy is already set. If yes, then just return fixed_mut_num_mna, trigger_fixed_mut_num_mna = False (as 'fixed mutation number' strategy is already triggered, so no need to re-trigger it) and enable_early_stopping_check = True
        if(self.prv_running_stage == self.fixed_mut_num_trigg_stage_mna):
            print(f'The "fixed mutation number" strategy is already set. So, just returning fixed_mut_num_mna, trigger_fixed_mut_num_mna = False (as "fixed mutation number" strategy is already triggered, so no need to re-trigger it) and enable_early_stopping_check = True')
            return (-self.fixed_mut_num_mna, trigger_fixed_mut_num_mna, enable_early_stopping_check)
        elif(self.best_ppi >= self.prob_threshold_for_mut_num_adj_mna):  # check whether best_ppi reaches the respective threshold
            # As ppi threshold is reached, so 'fixed mutation number' strategy will not be applied and Early Stopping Criterion check will be enabled.
            print('As ppi threshold is reached, so "fixed mutation number" strategy will not be applied, "percent_len_for_calc_mut_pts" will not be changed and Early Stopping Criterion check will be enabled...')
            return (self.prv_percent_len_for_calc_mut_pts, trigger_fixed_mut_num_mna, enable_early_stopping_check)
        else:
            # not in "fixed mutation number" stage and also ppi threshold is not reached
            # Find the current stage
            crnt_stage = (crnt_batch_indx // num_of_batches_per_stage ) + 1
            # check whether crnt_stage just reaches fixed_mut_num_trigg_stage_mna
            if(crnt_stage == self.fixed_mut_num_trigg_stage_mna):
                print(f'crnt_stage ({crnt_stage}) just reaches the fixed_mut_num_trigg_stage_mna ({self.fixed_mut_num_trigg_stage_mna}). So, returning fixed_mut_num_mna, trigger_fixed_mut_num_mna = True and enable_early_stopping_check = True')
                trigger_fixed_mut_num_mna = True
                # update prv_running_stage
                self.prv_running_stage = self.fixed_mut_num_trigg_stage_mna
                return (-self.fixed_mut_num_mna, trigger_fixed_mut_num_mna, enable_early_stopping_check)
            elif((crnt_stage < self.fixed_mut_num_trigg_stage_mna) and (crnt_stage == self.prv_running_stage + 1)):
                print(f'crnt_stage ({crnt_stage}) just incements from the previous stage ({self.prv_running_stage}) but it is not the "fixed mutation number" triggering stage. So, just reducing the current value of "percent_len_for_calc_mut_pts" ({self.prv_percent_len_for_calc_mut_pts}) to half, if it is not already 1.')
                # Check whether prv_percent_len_for_calc_mut_pts is already 1. If yes, just avoid reducing it further.
                if(self.prv_percent_len_for_calc_mut_pts != 1):
                    self.prv_percent_len_for_calc_mut_pts = self.prv_percent_len_for_calc_mut_pts // 2
                # Not enable_early_stopping_check
                enable_early_stopping_check = False
                # update prv_running_stage
                self.prv_running_stage = crnt_stage
                return (self.prv_percent_len_for_calc_mut_pts, trigger_fixed_mut_num_mna, enable_early_stopping_check)
            elif((crnt_stage < self.fixed_mut_num_trigg_stage_mna) and (crnt_stage == self.prv_running_stage)):
                print(f'crnt_stage ({crnt_stage}) is same as the previous stage ({self.prv_running_stage}). So, just maintain the status-quo.')
                enable_early_stopping_check = False
                return (self.prv_percent_len_for_calc_mut_pts, trigger_fixed_mut_num_mna, enable_early_stopping_check)
            # end of if-else block
        # end of if-else block: if(self.prv_running_stage == self.fixed_mut_num_trigg_stage_mna):
    # end of adjust_mutation_number() method

