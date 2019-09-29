from vimms.PlotsForPaper import compute_performance_scenario_2


class Evaluation(object):
    """
    A class to compute performance evaluation
    """

    def __init__(self, controller, evaluation_strategy):
        """
        Initialises an evaluation object
        :param controller: a controller
        :param evaluation_strategy: a strategy to evaluate performance
        """
        self.controller = controller
        self.strategy = evaluation_strategy

    def compute_performance(self):
        """
        Evaluates a controller performance based on the provided evaluation strategy
        :return: performance values
        """
        return self.strategy.evaluate(self.controller)


class TopNEvaluationStrategy(object):
    """
    A class to compute Top-N performance acccording to Section 3.3 of the paper
    """

    def __init__(self, **params):
        """
        Evaluation parameters
        :param params: multiple keyword arguments of parameters that we need

        For Top-N these parameters are:
        - dataset = the list of chemicals to evaluate performance on
        - min_ms1_intensity = minimum MS1 intensity to fragment
        - fullscan_peaks_df = a dataframe of XCMS peak-picking result on the full-scan file
        - fragmentation_peaks_df = a dataframe of XCMS peak-picking result on the fragmentation file
        - fullscan_filename = an optional filename to filter fullscan_peaks_df, otherwise None
        - fragfile_filename = an optional filename to filter fragmentation_peaks_df, otherwise None
        - matching_mz_tol = matching tolerance (in ppm) to match peaks
        - matching_rt_tol = matching tolerance (in seconds) to match peaks
        """
        self.params = params

    def evaluate(self, controller):
        return compute_performance_scenario_2(controller,
                                       self.params['dataset'],
                                       self.params['min_ms1_intensity'],
                                       self.params['fullscan_filename'],
                                       self.params['fragfile_filename'],
                                       self.params['fullscan_peaks_df'],
                                       self.params['fragmentation_peaks_df'],
                                       self.params['matching_mz_tol'],
                                       self.params['matching_rt_tol'],
                                       chem_to_frag_events=None)
