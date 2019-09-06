import numpy as np
from tqdm import tqdm
import math
import sys
import copy

from vimms.Controller import *
from vimms.MassSpec import *
from vimms.Common import  POSITIVE, DEFAULT_MS1_SCAN_WINDOW, LoggerMixin


def DiaRestrictedScanner(dataset, ps, dia_design, window_type, kaufmann_design, extra_bins, num_windows=None, pbar=False):
    mass_spec = IndependentMassSpectrometer(POSITIVE, dataset, ps)
    controller = TreeController(mass_spec, dia_design, window_type, kaufmann_design, extra_bins, num_windows)
    controller.run(10, 20, pbar)
    controller.scans[2] = controller.scans[2][0:(controller.scans[1][1].scan_id-1)]
    controller.scans[1] = controller.scans[1][0:2]
    return controller


class DiaAnalyser(object):
    def __init__(self, controller, min_intensity=0):
        self.controller = controller
        self.scans = controller.scans
        self.dataset = controller.mass_spec.chemicals
        self.chemicals_identified = 0
        self.ms2_matched = 0
        self.entropy = 0
        self.ms1_range = np.array([0,1000]) #TODO: fix this (ie make it so it can be controller in controller and then here
        self.min_intensity = min_intensity

        self.ms1_scan_times = np.array([scan.rt for scan in self.scans[1]])
        self.ms2_scan_times = np.array([scan.rt for scan in self.scans[2]])
        self.ms1_mzs = [self.controller.mass_spec._get_all_mz_peaks(self.dataset[i], self.dataset[i].rt + 0.01, 1, [[(0, 1000)]])[0][0] for i in range(len(self.dataset))]
        self.ms1_start_rt = np.array([data.rt for data in self.dataset])
        self.ms1_end_rt = np.array([data.rt + data.chromatogram.max_rt for data in self.dataset])
        self.first_scans, self.last_scans = self._get_scan_times()

        self.chemical_locations = []

        with tqdm(total=len(self.dataset)) as pbar:
            for chem_num in range(len(self.dataset)):
                chemical_location = self._get_chemical_location(chem_num)
                chemical_time = [(self.first_scans[chem_num], self.last_scans[chem_num])]
                self.chemical_locations.append(chemical_location)
                num_ms1_options = 0
                for i in range(len(chemical_location)):
                    mz_location = np.logical_and(np.array(self.ms1_mzs) > chemical_location[i][0],
                                                          np.array(self.ms1_mzs) <= chemical_location[i][1])
                    time_location = np.logical_and(self.ms1_start_rt <= chemical_time[0][1],
                                                                 self.ms1_end_rt >= chemical_time[0][0])
                    num_ms1_options += sum(mz_location * time_location)
                if num_ms1_options == 0:
                    self.entropy += -len(self.dataset[chem_num].children) * len(self.dataset) * math.log(1 / len(self.dataset))
                else:
                    self.entropy += -len(self.dataset[chem_num].children) * num_ms1_options * math.log(1 / num_ms1_options)
                    if num_ms1_options == 1:
                        self.chemicals_identified += 1
                        self.ms2_matched += len(self.dataset[chem_num].children)
                pbar.update(1)
            pbar.close()

    def _get_scan_times(self):
        max_time = self.scans[1][-1].rt
        if self.scans[2] != []:
            max_time = max(self.scans[1][-1].rt, self.scans[2][-1].rt) + 1
        first_scans = [max_time for i in self.dataset]
        last_scans = [max_time for i in self.dataset]
        for chem_num in range(len(self.dataset)):
            relevant_times = self.ms1_scan_times[(self.ms1_start_rt[chem_num] < self.ms1_scan_times) & (self.ms1_scan_times < self.ms1_end_rt[chem_num])]
            for time in relevant_times:
                intensity = self.controller.mass_spec._get_all_mz_peaks(self.dataset[chem_num], time, 1, [[(0,1000)]])[0][1] #TODO: Make MS1 range more general
                if intensity > self.min_intensity:
                    first_scans[chem_num] = min(first_scans[chem_num], time)
                    last_scans[chem_num] = time
        return first_scans, last_scans
    #
    # def _get_ms1_mzs(self):
    #     # get list of ms1s
    #     ms1_mzs = []
    #     if isinstance(self.scans[1], list):
    #         for j in range(len(self.scans[1])):
    #             ms1_mzs.extend(self.scans[1][j].mzs)
    #         ms1_mzs = np.unique(np.array(ms1_mzs))
    #     else:
    #         ms1_mzs = self.scans[1].mzs
    #     return ms1_mzs

    def _get_chemical_location(self, chem_num):
        # find location where ms2s of chemical can be narrowed down to
        which_scans = np.where(np.logical_and(np.array(self.ms2_scan_times) > self.first_scans[chem_num],
                                              np.array(self.ms2_scan_times) < self.last_scans[chem_num]))
        chemical_scans = np.array(self.scans[2])[which_scans]
        if chemical_scans.size == 0:
            possible_locations = [(0,1000)]  # TODO: Make this more general
        else:
            locations = [scan.isolation_windows for scan in chemical_scans]
            scan_times = [scan.rt for scan in chemical_scans]
            split_points = np.unique(np.array(list(sum(sum(sum(locations, []), []), ()))))
            split_points = np.unique(np.concatenate((split_points, self.ms1_range)))
            mid_points = [(split_points[i] + split_points[i + 1]) / 2 for i in range(len(split_points) - 1)]
            possible_mid_points = self._get_mid_points_in_location(chem_num, mid_points, locations, scan_times)
            possible_locations = self._get_possible_locations(possible_mid_points, split_points)
        return possible_locations

    def _get_mid_points_in_location(self, chem_num, mid_points, locations, scan_times):
        # find mid points which satisfying scans locations
        current_mid_points = mid_points
        for i in range(len(locations)):
            chem_scanned = isinstance(
                self.controller.mass_spec._get_all_mz_peaks(self.dataset[chem_num], scan_times[i], 2, locations[i]),
                list)
            new_mid_points = []
            for j in range(len(current_mid_points)):
                if chem_scanned == self._in_window(current_mid_points[j], locations[i]):
                    new_mid_points.append(current_mid_points[j])
            current_mid_points = new_mid_points
        return current_mid_points

    def _get_possible_locations(self, possible_mid_points, split_points):
        # find locations where possible mid points can be in, then simplify locations
        possible_locations = []
        for i in range(len(possible_mid_points)):
            min_location = max(
                np.array(split_points)[np.where(np.array(split_points) < possible_mid_points[i])].tolist())
            max_location = min(
                np.array(split_points)[np.where(np.array(split_points) >= possible_mid_points[i])].tolist())
            possible_locations.extend([(min_location, max_location)])
            # TODO: need to simplify still
        return possible_locations

    def _in_window(self, mid_point, locations):
        for window in locations[0]:
            if (mid_point > window[0] and mid_point <= window[1]):
                return True
        return False


class RestrictedDiaAnalyser(object):
    def __init__(self, controller):
        self.entropy = []
        self.chemicals_identified = []
        self.ms2_matched = []
        self.scan_num = []
        temp_controller = copy.deepcopy(controller)
        start = len(temp_controller.scans[2])
        for num_ms2_scans in range(start, -1, -1):
            temp_controller.scans[2] = temp_controller.scans[2][0:num_ms2_scans]
            analyser = DiaAnalyser(temp_controller)
            self.entropy.append(analyser.entropy)
            self.chemicals_identified.append(analyser.chemicals_identified)
            self.ms2_matched.append(analyser.ms2_matched)
            self.scan_num.append(num_ms2_scans + 1)
        self.entropy.reverse()
        self.chemicals_identified.reverse()
        self.ms2_matched.reverse()
        self.scan_num.reverse()



############################# ok up to here #####################################


class Scan_Results_Calculator(object):
    """
    Method for taking raw results, grouping ms2 fragments and determining in which scans they were found
    """

    def __init__(self, dia_results, ms2_mz_slack=0.00001, ms2_intensity_slack=0.1):
        self.intensities_in_scans = dia_results.intensities_in_scans
        self.mz_in_scans = dia_results.mz_in_scans
        self.locations = dia_results.locations
        self.bin_walls = dia_results.bin_walls
        self.ms1_values = dia_results.ms1_values
        self.results = [[] for i in range(len(dia_results.locations))]
        unlisted_mz_in_scans = np.concatenate(dia_results.mz_in_scans)
        unlisted_intensities_in_scans = np.concatenate(dia_results.intensities_in_scans)
        # find unique mz
        unique_mz = [[unlisted_mz_in_scans[0]]]
        unique_intensities = [[unlisted_intensities_in_scans[0]]]
        for unlisted_mz_index in range(1, len(unlisted_mz_in_scans)):
            unique_mz_min = math.inf
            for unique_mz_index in range(len(unique_mz)):
                unique_dist = abs(
                    sum(unique_mz[unique_mz_index]) / len(unique_mz[unique_mz_index]) - unlisted_mz_in_scans[
                        unlisted_mz_index])
                if (unique_dist < unique_mz_min):
                    unique_mz_min = unique_dist
                    unique_mz_which = unique_mz_index
            if unique_mz_min < ms2_mz_slack:
                unique_mz[unique_mz_which].append(unlisted_mz_in_scans[unlisted_mz_index])
                unique_intensities[unique_mz_which].append(unlisted_intensities_in_scans[unlisted_mz_index])
            else:
                unique_mz.append([unlisted_mz_in_scans[unlisted_mz_index]])
                unique_intensities.append([unlisted_intensities_in_scans[unlisted_mz_index]])
        self.ms2_intensities = unique_intensities
        self.ms2_mz = unique_mz
        # find where intensities are unique and assign them a scan result
        for unique_mz_index in range(len(unique_mz)):
            if max(abs(unique_intensities[0] - sum(unique_intensities[0]) / len(
                    unique_intensities[0]))) > ms2_intensity_slack:
                print("not ready yet")
            else:
                for location_index in range(len(dia_results.locations)):
                    TF_in_location = []
                    for unique_index in range(len(unique_mz[unique_mz_index])):
                        TF_in_location.append(
                            unique_mz[unique_mz_index][unique_index] in dia_results.mz_in_scans[location_index] and
                            unique_intensities[unique_mz_index][unique_index] in dia_results.intensities_in_scans[
                                location_index])
                    if any(TF_in_location):
                        self.results[location_index].append(1)
                    else:
                        self.results[location_index].append(0)


class Dia_Location_Finder(object):
    """
    Method for finding location of ms2 fragments based on which DIA scans they are seen in
    """

    def __init__(self, scan_results):
        self.locations = scan_results.locations
        self.results = scan_results.results
        self.bin_walls = scan_results.bin_walls
        self.ms1_values = scan_results.ms1_values
        self.ms2_intensities = scan_results.ms2_intensities
        self.ms2_mz = scan_results.ms2_mz
        self.location_all = []
        bin_mid_points = list((np.array(self.bin_walls[1:]) + np.array(self.bin_walls[:(len(self.bin_walls) - 1)])) / 2)
        for sample_index in range(0, len(self.results[0])):
            mid_point_TF = []
            for mid_points_index in range(0, len(bin_mid_points)):
                mid_point_TF.append(self._mid_point_in_location(bin_mid_points[mid_points_index], sample_index))
            self.location_all.append([(list(np.array(self.bin_walls)[np.where(np.array(mid_point_TF) == True)])[0],
                                       list(np.array(self.bin_walls[1:])[np.where(np.array(mid_point_TF) == True)])[
                                           0])])

    def _mid_point_in_location(self, mid_point, sample_index):
        for locations_index in range(0, len(self.locations)):
            if self._in_window(mid_point, self.locations[locations_index]) == True and self.results[locations_index][
                sample_index] == 0:
                return False
            if self._in_window(mid_point, self.locations[locations_index]) == False and self.results[locations_index][
                sample_index] == 1:
                return False
        else:
            return True

    def _in_window(self, mid_point, locations):
        for window in locations:
            if (mid_point > window[0] and mid_point <= window[1]):
                return True
        return False


class Entropy(object):
    """
    Method for calculating entropy based on locations of ms2 components
    """

    def __init__(self, dia_location_finder):
        self.entropy = []
        self.components_determined = []
        ms1_vec = []
        ms2_vec = []
        for i in range(0, len(dia_location_finder.bin_walls) - 1):
            ms2_vec.extend([0])
            ms1_vec.extend([len(np.where(
                np.logical_and(np.array(dia_location_finder.ms1_values) > dia_location_finder.bin_walls[i],
                               np.array(dia_location_finder.ms1_values) <= dia_location_finder.bin_walls[i + 1]))[0])])
            # fix this
            for j in range(0, len(dia_location_finder.location_all)):
                if [(dia_location_finder.bin_walls[i], dia_location_finder.bin_walls[i + 1])] == \
                        dia_location_finder.location_all[j]:
                    ms2_vec[i] += 1
        ms1_vec_nozero = [value for value in ms1_vec if value != 0]
        ms2_vec_nozero = [value for value in ms1_vec if value != 0]
        entropy_vec = []
        for j in range(0, len(ms2_vec_nozero)):
            entropy_vec.append(-ms2_vec_nozero[j] * ms1_vec_nozero[j] * math.log(1 / ms1_vec_nozero[j]))
        self.entropy = sum(entropy_vec)
        self.components_determined = sum(np.extract(np.array(ms1_vec_nozero) == 1, ms2_vec_nozero))
        self.components = sum(ms2_vec_nozero)


class Entropy_List(object):
    """
    Method for calculating entropy on multiple subsets of the DIA results. Useful for creating plots and monitoring performance over multiple scans
    """

    def __init__(self, dataset, ms_level, rt, dia_design, window_type, kaufmann_design, extra_bins=0, range_slack=0.01,
                 ms1_range=[(None, None)], num_windows=None, ms2_mz_slack=0.00001, ms2_intensity_slack=0.1):
        self.entropy = []
        self.components_determined = []
        if (dia_design != "kaufmann"):
            sys.exit("Only the 'kaufmann' method can be used with Entropy_List")
        if (kaufmann_design == "tree"):
            self.start_subsample_scans = 2
            self.end_subsample_scans = 7 + extra_bins
        elif (kaufmann_design == "nested"):
            self.start_subsample_scans = 8
            self.end_subsample_scans = 12 + extra_bins
        else:
            sys.exit("Cannot use Entropy_List with this design. Kaufmann 'nested' or 'tree' only.")
        for i in range(self.start_subsample_scans, self.end_subsample_scans):
            dia = Dia_Methods_Subsample(
                Dia_Methods(dataset, ms_level, rt, dia_design, window_type, kaufmann_design, extra_bins, range_slack,
                            ms1_range, num_windows), i)
            results = Entropy(Dia_Location_Finder(Scan_Results_Calculator(dia, ms2_mz_slack, ms2_intensity_slack)))
            self.entropy.append(results.entropy)
            self.components_determined.append(results.components_determined)
            self.components = results.components




class Dia_Methods(object):
    """
    Method for doing DIA on a dataset of ms1 and ms2 peaks. Creates windows and then return attributes of scan results
    """

    def __init__(self, dataset, ms_level, rt, dia_design, window_type, kaufmann_design=None, extra_bins=0,
                 range_slack=0.01, ms1_range=[(None, None)], num_windows=None):
        dia_windows = Dia_Windows(dataset, dia_design, window_type, kaufmann_design, extra_bins, range_slack, ms1_range,
                                  num_windows)
        self.bin_walls = dia_windows.bin_walls
        self.locations = dia_windows.locations
        self.ms1_values = dia_windows.ms1_values
        self.mz_in_scans = []
        self.intensities_in_scans = []
        for window_index in range(0, len(self.locations)):
            data_scan = Dataset_Scan(dataset, ms_level, rt, self.locations[window_index])
            self.mz_in_scans.append(np.array(data_scan.mz_in_scan))
            self.intensities_in_scans.append(np.array(data_scan.scan_intensities))


class Dia_Methods_Subsample(object):
    """
    Method for taking a sumsample of DIA results. Helpful for visualising results as scans progress
    """

    def __init__(self, dia_methods, num_scans):
        self.bin_walls = list(set(np.array(sum(dia_methods.locations[0:num_scans], [])).flatten()))
        self.bin_walls.sort()
        self.locations = dia_methods.locations[0:num_scans]
        self.ms1_values = dia_methods.ms1_values
        self.mz_in_scans = dia_methods.mz_in_scans[0:num_scans]
        self.intensities_in_scans = dia_methods.intensities_in_scans[0:num_scans]