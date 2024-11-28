# Define the TemperatureScheduler class
class TemperatureScheduler():
    """
    Class for scheduling the temperature parameter used as a part of Metropolis Criterion in Monte Carlo simulation during protein design through interaction procedure.

    It continuously monitors the PPI confidence score (which is a probability value between 0.0 and 1.0). If the probability value is not increasing for certain number of batches, it increases the temperature by a given amount to encourage exploration of the conformational space.
    """
    def __init__(self, initial_temp_sch=None, temp_inc_factor_sch=None, patience_for_inc_sch=None
                 , cooldown_sch=None, max_temp_sch=None, verbose_sch=False):
        """
        Initialize TemperatureScheduler.

        Args:
            initial_temp_sch (float): Initial temperature value for the scheduler.
            temp_inc_factor_sch (float): Factor by which the temperature is increased in the temperature scheduler. e.g. if temp_inc_factor_sch = 0.1, then new_temp = old_temp * 1.1 i.e. 10% increase in the current temperature.
            patience_for_inc_sch (int): Number of batches with no probability improvement before increasing the temperature by scheduler.
            cooldown_sch (int): Number of batches to wait before resuming normal scheduling operation after temperature change.
            max_temp_sch (float): Upper bound on temperature for the temperature scheduler.
            verbose_sch (bool): Whether to enable verbose output from temperature scheduler.
        """
        self.temp = initial_temp_sch
        if((temp_inc_factor_sch <= 0.0) or (temp_inc_factor_sch >= 1.0)):
            raise ValueError('temp_inc_factor_sch should be > 0.0 and < 1.0.')
        self.temp_inc_factor_sch = temp_inc_factor_sch
        self.patience_for_inc_sch = patience_for_inc_sch
        self.cooldown_sch = cooldown_sch
        self.max_temp_sch = max_temp_sch
        self.verbose_sch = verbose_sch

        # initialize a few counters to keep track of the prob values
        self.cooldown_wait_count = 0  # to keep track of the cooldown_sch
        self.prob_inc_wait_count = 0  # to keep track of the patience_for_inc_sch
        self.best_prob = 0.0  # to decide about the increase in the probabilities 
    # end of __init__() method


    def adjust_temperature(self, crnt_prob):
        """
        Adjust temperature based on current probability value.

        Args:
            crnt_prob (float): current probability value.

        Returns:
            float: Adjusted temperature.
        """
        # First checks for the cooldown
        if(self.cooldown_wait_count < self.cooldown_sch):
            # still in the cooldown phase, hence just return the current temeprature
            if self.verbose_sch:  
                print(f"\n ##### crnt_prob: {crnt_prob}; best_prob: {self.best_prob}")
                print(f"Still in the cooldown phase as cooldown_wait_count ({self.cooldown_wait_count + 1}) <= cooldown_sch ({self.cooldown_sch}), \
                      \n hence returning the current temperature")
            self.cooldown_wait_count += 1
            # Checks and adjusts the best_prob as it would be required later
            if(crnt_prob >= self.best_prob): 
                self.best_prob = crnt_prob
            # end of if block
            return self.temp
        # end of if block: if(self.cooldown_wait_count < self.cooldown_sch):

        # update the counters based on the crnt_prob
        if(crnt_prob >= self.best_prob): 
            self.best_prob = crnt_prob
            # reset the counter
            self.prob_inc_wait_count = 0
        else:
            # increase the counter
            self.prob_inc_wait_count += 1
        # end of if-else block
        if self.verbose_sch:  
            print(f"\n ##### crnt_prob: {crnt_prob}; best_prob: {self.best_prob}")
            print(f"##### self.prob_inc_wait_count: {self.prob_inc_wait_count}\n")

        if(self.prob_inc_wait_count > self.patience_for_inc_sch):
            # if the prob values does not increase for quite some time and prob_inc_wait_count exceeds the patience for increment (patience_for_inc_sch),
            # then increase the temperature to encourage exploration of the conformational space
            if self.verbose_sch:  
                print(f'\n prob_inc_wait_count ({self.prob_inc_wait_count}) exceeds the patience for increment ({self.patience_for_inc_sch})')
                print(f" ans so, attempting to increase the temperature...")

            new_temp = self.temp * (1.0 + self.temp_inc_factor_sch)
            new_temp = round(new_temp, 3)
            if(new_temp > self.max_temp_sch):
                if self.verbose_sch: print(f'Alas!! current temperature {self.temp} cannot be further increased to {new_temp} as it would be higher than the temperature upper bound ({self.max_temp_sch})')
            else:
                self.temp = new_temp
                if self.verbose_sch: print(f"temperature is increased to {self.temp}")
            # end of if-else block
            # reset all the counters
            self.cooldown_wait_count = self.prob_inc_wait_count = 0
        # end of if-else block

        return self.temp
    # end of adjust_temperature() method


# # Demo code to test TemperatureScheduler
# def main():
#     # Define parameters
#     initial_temp_sch = 0.01
#     temp_inc_factor_sch = 0.1
#     patience_for_inc_sch = 3  # 200
#     cooldown_sch = 2  # 50
#     max_temp_sch = 1.0
#     verbose_sch = True

#     # Initialize TemperatureScheduler
#     scheduler = TemperatureScheduler(initial_temp_sch, temp_inc_factor_sch, patience_for_inc_sch, cooldown_sch, max_temp_sch, verbose_sch)

#     # Simulated iterations
#     prob_values = [0.08, 0.07, 0.075, 0.076, 0.074, 0.073, 0.072, 0.071, 0.07, 0.069, 0.1, 0.15, 0.12, 0.14, 0.09, 0.13]

#     # Simulate temperature adjustment based on crnt_prob values
#     for crnt_prob in prob_values:
#         scheduler.adjust_temperature(crnt_prob)

# if __name__ == "__main__":
#     main()
