class EarlyStoppingCriteria():
    """
    Class for implementing early stopping criteria based on improvement checking and/or threshold crossing.
 
    Args:
    prob_threshold (float): Threshold value for the probability metric.
    stopping_patience (int): Number of batches to wait for improvement before stopping.
    delta_prob_improv (float): Minimum increase in probability to qualify as improvement.
    check_improvement (bool): Whether to check for improvement.
    check_threshold (bool): Whether to check for threshold crossing.
    """
    def __init__(self, prob_threshold=None, stopping_patience=None, delta_prob_improv=None, check_improvement=None, check_threshold=None):
        """
        Initialize EarlyStoppingCriteria instance.

        Args:
        prob_threshold (float): Threshold value for the probability metric.
        stopping_patience (int): Number of batches to wait for improvement before stopping.
        delta_prob_improv (float): Minimum increase in probability to qualify as improvement.
        check_improvement (bool): Whether to check for improvement.
        check_threshold (bool): Whether to check for threshold crossing.
        """
        self.prob_threshold = prob_threshold
        self.stopping_patience = stopping_patience
        self.delta_prob_improv = delta_prob_improv
        self.check_improvement = check_improvement
        self.check_threshold = check_threshold
        # initialize following variables
        self.best_prob = -1
        self.no_improvement_count = 0
    # end of __init__() method


    def check_early_stopping(self, crnt_prob_value):
        """
        Check for early stopping based on improvement checking and/or prob_threshold crossing.

        Args:
        crnt_prob_value (float): Probability value for the current batch.

        Returns:
        A tuple consisting of a boolean flag (set to True if early stopping criteria met, False otherwise) and a string (containing early stopping criterion if early stopping criteria met, None otherwise). Abbreviated early stopping criteria are: 'NIM' (Not IMproved), 'THC' (THreshold Crossed).
        """
        if self.check_improvement:
            if crnt_prob_value > self.best_prob + self.delta_prob_improv:
                self.best_prob = crnt_prob_value
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1

            if self.no_improvement_count >= self.stopping_patience:
                print(f"No improvement for {self.stopping_patience} batches. Stopping simulation.")
                return (True, 'NIM')

        if self.check_threshold and crnt_prob_value <= self.prob_threshold:
            print(f"Current probability ({crnt_prob_value}) crossed threshold ({self.prob_threshold}). Stopping simulation.")
            return (True, 'THC')

        return (False, None)
    # end of check_early_stopping() method
    
