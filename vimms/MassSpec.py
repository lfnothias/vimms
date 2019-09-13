import math
from collections import defaultdict
from collections import namedtuple

import numpy as np
import pandas as pd
from events import Events

from vimms.Common import LoggerMixin, adduct_transformation


class Peak(object):
    """
    A simple class to represent an empirical or sampled scan-level peak object
    """

    def __init__(self, mz, rt, intensity, ms_level):
        self.mz = mz
        self.rt = rt
        self.intensity = intensity
        self.ms_level = ms_level

    def __repr__(self):
        return 'Peak mz=%.4f rt=%.2f intensity=%.2f ms_level=%d' % (self.mz, self.rt, self.intensity, self.ms_level)

    def __eq__(self, other):
        if not isinstance(other, Peak):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return math.isclose(self.mz, other.mz) and \
               math.isclose(self.rt, other.rt) and \
               math.isclose(self.intensity, other.intensity) and \
               self.ms_level == other.ms_level

class Scan(object):
    def __init__(self, scan_id, mzs, intensities, ms_level, rt, scan_duration=None, isolation_windows=None, parent=None):
        assert len(mzs) == len(intensities)
        self.scan_id = scan_id

        # ensure that mzs and intensites are sorted by their mz values
        p = mzs.argsort()
        self.mzs = mzs[p]
        self.intensities = intensities[p]

        self.ms_level = ms_level
        self.rt = rt
        self.num_peaks = len(mzs)

        self.scan_duration = scan_duration
        self.isolation_windows = isolation_windows
        self.parent = parent

    def __repr__(self):
        return 'Scan %d num_peaks=%d rt=%.2f ms_level=%d' % (self.scan_id, self.num_peaks, self.rt, self.ms_level)


class ScanParameters(object):
    MS_LEVEL = 'ms_level'
    ISOLATION_WINDOWS = 'isolation_windows'
    PRECURSOR = 'precursor'
    DYNAMIC_EXCLUSION_MZ_TOL = 'mz_tol'
    DYNAMIC_EXCLUSION_RT_TOL = 'rt_tol'
    TIME = 'time'
    N = 'N'

    def __init__(self):
        self.params = {}

    def set(self, key, value):
        self.params[key] = value

    def get(self, key):
        if key in self.params:
            return self.params[key]
        else:
            return None

    def __repr__(self):
        return 'ScanParameters %s' % (self.params)


class FragmentationEvent(object):  # for benchmarking purpose
    def __init__(self, chem, query_rt, ms_level, peaks, scan_id):
        self.chem = chem
        self.query_rt = query_rt
        self.ms_level = ms_level
        self.peaks = peaks
        self.scan_id = scan_id

    def __repr__(self):
        return 'MS%d FragmentationEvent for %s at %f' % (self.ms_level, self.chem, self.query_rt)


class MassSpectrometer(LoggerMixin):
    MS_SCAN_ARRIVED = 'MsScanArrived'
    ACQUISITION_STREAM_OPENING = 'AcquisitionStreamOpening'
    ACQUISITION_STREAM_CLOSING = 'AcquisitionStreamClosing'

    def __init__(self, ionisation_mode):
        self.ionisation_mode = ionisation_mode

        # following IAPI events
        self.events = Events((self.MS_SCAN_ARRIVED, self.ACQUISITION_STREAM_OPENING, self.ACQUISITION_STREAM_CLOSING,))
        self.event_dict = {
            self.MS_SCAN_ARRIVED: self.events.MsScanArrived,
            self.ACQUISITION_STREAM_OPENING: self.events.AcquisitionStreamOpening,
            self.ACQUISITION_STREAM_CLOSING: self.events.AcquisitionStreamClosing
        }

    def get_next_scan(self):
        raise NotImplementedError()

    def fire_event(self, event_name, arg=None):
        if event_name not in self.event_dict:
            raise ValueError('Unknown event name')

        # fire the event
        e = self.event_dict[event_name]
        if arg is not None:
            e(arg)
        else:
            e()

    def register(self, event_name, handler):
        if event_name not in self.event_dict:
            raise ValueError('Unknown event name')
        e = self.event_dict[event_name]
        e += handler  # register a new event handler for e

    def clear(self, event_name):
        if event_name not in self.event_dict:
            raise ValueError('Unknown event name')
        e = self.event_dict[event_name]
        e.targets = []


ExclusionItem = namedtuple('ExclusionItem', 'from_mz to_mz from_rt to_rt')


# Independent here refers to how the intensity of each peak in a scan is independent of each other
# i.e. there's no ion supression effect
class IndependentMassSpectrometer(MassSpectrometer):
    def __init__(self, ionisation_mode, chemicals, peak_sampler, schedule_file=None, add_noise=False):
        super().__init__(ionisation_mode)
        self.chemicals = chemicals
        self.peak_sampler = peak_sampler
        self.idx = 0
        self.time = 0
        self.queue = []
        self.repeating_scan_parameters = None
        self.precursor_information = defaultdict(list)  # key: Precursor object, value: ms2 scans
        self.schedule_file = schedule_file
        if self.schedule_file is not None:
            self.schedule = pd.read_csv(schedule_file)
        self.add_noise = add_noise # whether to add noise to the generated fragment peaks

        self.fragmentation_events = []  # which chemicals produce which peaks
        self.previous_level = None  # ms_level of the previous scan

        chem_rts = np.array([chem.rt for chem in self.chemicals])
        self.chrom_min_rts = np.array([chem.chromatogram.min_rt for chem in self.chemicals]) + chem_rts
        self.chrom_max_rts = np.array([chem.chromatogram.max_rt for chem in self.chemicals]) + chem_rts
        self.exclusion_list = []  # a list of ExclusionItem

        # required to sample for different scan durations based on (N, DEW) in the hybrid controller
        self.current_N = 0
        self.current_DEW = 0

    def run(self, min_time, max_time, pbar=None):
        if self.schedule_file is None:
            self.time = min_time
        else:
            self.time = self.schedule["targetTime"].values[0]
            max_time = self.schedule["targetTime"].values[-1]
        self.fire_event(MassSpectrometer.ACQUISITION_STREAM_OPENING)
        try:
            while self.time < max_time:
                initial_time = self.time

                # if the processing queue is empty, then just do the repeating scan
                if len(self.queue) == 0:
                    param = self.repeating_scan_parameters
                else:
                    # otherwise pop the parameter for the next scan from the queue
                    param = self.queue.pop(0)

                # do one scan and increase time
                scan = self.get_next_scan(param)
                self.logger.info('Time %f Len(queue)=%d' % (self.time, len(self.queue)))

                # if MS2 and above, and the controller tells us which precursor ion the scan is coming from, store it
                precursor = param.get(ScanParameters.PRECURSOR)
                if scan.ms_level >= 2 and precursor is not None:
                    # update precursor ion information
                    isolation_windows = param.get(ScanParameters.ISOLATION_WINDOWS)
                    iso_min = isolation_windows[0][0][0]
                    iso_max = isolation_windows[0][0][1]
                    self.logger.debug('Time {:.6f} Isolated precursor ion {:.4f} at ({:.4f}, {:.4f})'.format(self.time,
                                                                                                             precursor.precursor_mz,
                                                                                                             iso_min,
                                                                                                             iso_max))
                    self.precursor_information[precursor].append(scan)

                    # update dynamic exclusion list to prevent the same precursor ion being fragmented multiple times in
                    # the same mz and rt window
                    # Note: at this point, fragmentation has occurred and time has been incremented!
                    # TODO: we need to add a repeat count too, i.e. how many times we've seen a fragment peak before
                    #  it gets excluded (now it's basically 1)
                    mz = precursor.precursor_mz
                    mz_tol = param.get(ScanParameters.DYNAMIC_EXCLUSION_MZ_TOL)
                    rt_tol = param.get(ScanParameters.DYNAMIC_EXCLUSION_RT_TOL)
                    mz_lower = mz * (1 - mz_tol / 1e6)
                    mz_upper = mz * (1 + mz_tol / 1e6)
                    rt_lower = self.time
                    rt_upper = self.time + rt_tol
                    x = ExclusionItem(from_mz=mz_lower, to_mz=mz_upper, from_rt=rt_lower, to_rt=rt_upper)
                    self.logger.debug('Time {:.6f} Created dynamic exclusion window mz ({}-{}) rt ({}-{})'.format(
                        self.time,
                        x.from_mz, x.to_mz, x.from_rt, x.to_rt
                    ))
                    self.exclusion_list.append(x)

                # remove expired items from dynamic exclusion list
                self.exclusion_list = list(filter(lambda x: x.to_rt > self.time, self.exclusion_list))

                # store previous ms_level
                self.previous_level = scan.ms_level

                # print a progress bar if provided
                if pbar is not None:
                    elapsed = self.time - initial_time
                    if self.current_N > 0 and self.current_DEW > 0:
                        msg = '(%.3fs) ms_level=%d N=%d DEW=%d' % (self.time, scan.ms_level,
                                                                   self.current_N, self.current_DEW)
                    else:
                        msg = '(%.3fs) ms_level=%d' % (self.time, scan.ms_level)
                    pbar.update(elapsed)
                    pbar.set_description(msg)

        finally:
            self.fire_event(MassSpectrometer.ACQUISITION_STREAM_CLOSING)
            if pbar is not None:
                pbar.close()

    def get_next_scan(self, param):
        if param is not None:
            # generate a new scan at self.time
            scan = self._get_scan(self.time, param)

            # notify controller about this scan
            # the queue will be updated by the controller if necessary
            self.fire_event(self.MS_SCAN_ARRIVED, scan)

            # look into the queue, find out what the next scan ms_level is, and compute the scan duration
            try:
                next_scan_param = self.queue[0]
                next_level = next_scan_param.get(ScanParameters.MS_LEVEL)

                # Only the hybrid controller sends these N and DEW parameters
                # So for other controllers they will be None
                next_N = next_scan_param.get(ScanParameters.N)
                next_DEW = next_scan_param.get(ScanParameters.DYNAMIC_EXCLUSION_RT_TOL)
            except IndexError:  # if queue is empty, the next one is an MS1 scan by default
                next_level = 1
                next_N = None
                next_DEW = None

            current_level = scan.ms_level

            # get scan duration based on current and next level
            if current_level == 1 and next_level == 1:
                # special case: for the transition (1, 1), we want to get the scan duration from N=0 and DEW=0
                # because that's how we store it in the peak sampler object for the full scan data
                current_scan_duration = self.peak_sampler.scan_durations(current_level, next_level, 1,
                                                                         N=0, DEW=0)
            else:
                current_scan_duration = self.peak_sampler.scan_durations(current_level, next_level, 1,
                                                                         N=self.current_N, DEW=self.current_DEW)
            scan.scan_duration = current_scan_duration.flatten()[0]

            # increase simulator scan index and time
            self.idx += 1
            if self.schedule_file is None:
                self.time += scan.scan_duration
            else:
                self.time = self.schedule["targetTime"][self.idx]

            # keep track of the N and DEW values for the next scan if they have been changed by the Hybrid Controller
            if next_N is not None:
                self.current_N = next_N
                self.current_DEW = next_DEW
            return scan
        else:
            return None

    def add_to_queue(self, param):
        self.queue.append(param)

    def disable_repeating_scan(self):
        self.set_repeating_scan(None)

    def set_repeating_scan(self, params):
        self.repeating_scan_parameters = params

    def reset(self): # TODO: should reset other stuff initialised in the constructor too
        for key in self.event_dict:  # clear event handlers
            self.clear(key)
        self.time = 0  # reset internal time and index to 0
        self.idx = 0

    def exclude(self, mz, rt):  # TODO: make this faster?
        for x in self.exclusion_list:
            exclude_mz = x.from_mz <= mz <= x.to_mz
            exclude_rt = x.from_rt <= rt <= x.to_rt
            if exclude_mz and exclude_rt:
                self.logger.debug(
                    'Time {:.6f} Excluded precursor ion mz {:.4f} rt {:.2f} because of {}'.format(self.time, mz, rt, x))
                return True
        return False

    def _get_scan(self, scan_time, param):
        """
        Constructs a scan at a particular timepoint
        :param time: the timepoint
        :return: a mass spectrometry scan at that time
        """
        scan_mzs = []  # all the mzs values in this scan
        scan_intensities = []  # all the intensity values in this scan
        ms_level = param.get(ScanParameters.MS_LEVEL)
        isolation_windows = param.get(ScanParameters.ISOLATION_WINDOWS)
        scan_id = self.idx

        # for all chemicals that come out from the column coupled to the mass spec
        idx = self._get_chem_indices(scan_time)
        for i in idx:
            chemical = self.chemicals[i]

            # mzs is a list of (mz, intensity) for the different adduct/isotopes combinations of a chemical
            if self.add_noise:
                mzs = self._get_all_mz_peaks_noisy(chemical, scan_time, ms_level, isolation_windows)
            else:
                mzs = self._get_all_mz_peaks(chemical, scan_time, ms_level, isolation_windows)

            peaks = []
            if mzs is not None:
                chem_mzs = []
                chem_intensities = []
                for peak_mz, peak_intensity in mzs:
                    if peak_intensity > 0:
                        chem_mzs.append(peak_mz)
                        chem_intensities.append(peak_intensity)
                        p = Peak(peak_mz, scan_time, peak_intensity, ms_level)
                        peaks.append(p)

                scan_mzs.extend(chem_mzs)
                scan_intensities.extend(chem_intensities)

            # for benchmarking purpose
            if len(peaks) > 0:
                frag = FragmentationEvent(chemical, scan_time, ms_level, peaks, scan_id)
                self.fragmentation_events.append(frag)

        scan_mzs = np.array(scan_mzs)
        scan_intensities = np.array(scan_intensities)

        # Note: at this point, the scan duration is not set yet because we don't know what the next scan is going to be
        # We will set it later in the get_next_scan() method after we've notified the controller that this scan is produced.
        return Scan(scan_id, scan_mzs, scan_intensities, ms_level, scan_time,
                    scan_duration=None, isolation_windows=isolation_windows)

    def _get_chem_indices(self, query_rt):
        rtmin_check = self.chrom_min_rts <= query_rt
        rtmax_check = query_rt <= self.chrom_max_rts
        idx = np.nonzero(rtmin_check & rtmax_check)[0]
        return idx

    def _get_all_mz_peaks_noisy(self, chemical, query_rt, ms_level, isolation_windows):
        mz_peaks = self._get_all_mz_peaks(chemical, query_rt, ms_level, isolation_windows)
        if self.peak_sampler is None:
            return mz_peaks
        if mz_peaks is not None:
            noisy_mz_peaks = [(mz_peaks[i][0], self.peak_sampler.get_msn_noisy_intensity(mz_peaks[i][1], ms_level)) for i in range(len(mz_peaks))]
        else:
            noisy_mz_peaks = []
        noisy_mz_peaks += self.peak_sampler.get_noise_sample()
        return noisy_mz_peaks

    def _get_all_mz_peaks(self, chemical, query_rt, ms_level, isolation_windows):
        if not self._rt_match(chemical, query_rt):
            return None
        mz_peaks = []
        for which_isotope in range(len(chemical.isotopes)):
            for which_adduct in range(len(self._get_adducts(chemical))):
                mz_peaks.extend(
                    self._get_mz_peaks(chemical, query_rt, ms_level, isolation_windows, which_isotope, which_adduct))
        if mz_peaks == []:
            return None
        else:
            return mz_peaks

    def _get_mz_peaks(self, chemical, query_rt, ms_level, isolation_windows, which_isotope, which_adduct):
        # EXAMPLE OF USE OF DEFINITION: if we wants to do an ms2 scan on a chemical. we would first have ms_level=2 and the chemicals
        # ms_level =1. So we would go to the "else". We then check the ms1 window matched. It then would loop through
        # the children who have ms_level = 2. So we then go to second elif and return the mz and intensity of each ms2 fragment
        mz_peaks = []
        if ms_level == 1 and chemical.ms_level == 1:  # fragment ms1 peaks
            # returns ms1 peaks if chemical is has ms_level = 1 and scan is an ms1 scan
            if not (which_isotope > 0 and which_adduct > 0):
                # rechecks isolations window if not monoisotopic and "M + H" adduct
                if self._isolation_match(chemical, query_rt, isolation_windows[0], which_isotope, which_adduct):
                    intensity = self._get_intensity(chemical, query_rt, which_isotope, which_adduct)
                    mz = self._get_mz(chemical, query_rt, which_isotope, which_adduct)
                    mz_peaks.extend([(mz, intensity)])
        elif ms_level == chemical.ms_level:
            # returns ms2 fragments if chemical and scan are both ms2, 
            # returns ms3 fragments if chemical and scan are both ms3, etc, etc
            intensity = self._get_intensity(chemical, query_rt, which_isotope, which_adduct)
            mz = self._get_mz(chemical, query_rt, which_isotope, which_adduct)
            return [(mz, intensity)]
            # TODO: Potential improve how the isotope spectra are generated
        else:
            # check isolation window for ms2+ scans, queries children if isolation windows ok
            if self._isolation_match(chemical, query_rt, isolation_windows[chemical.ms_level - 1], which_isotope,
                                     which_adduct) and chemical.children is not None:
                for i in range(len(chemical.children)):
                    mz_peaks.extend(self._get_mz_peaks(chemical.children[i], query_rt, ms_level, isolation_windows,
                                                       which_isotope, which_adduct))
            else:
                return []
        return mz_peaks

    def _get_adducts(self, chemical):
        if chemical.ms_level == 1:
            return chemical.adducts
        else:
            return self._get_adducts(chemical.parent)

    def _rt_match(self, chemical, query_rt):
        return chemical.ms_level != 1 or chemical.chromatogram._rt_match(query_rt - chemical.rt)

    def _get_intensity(self, chemical, query_rt, which_isotope, which_adduct):
        if chemical.ms_level == 1:
            intensity = chemical.isotopes[which_isotope][1] * self._get_adducts(chemical)[which_adduct][1] * \
                        chemical.max_intensity
            return intensity * chemical.chromatogram.get_relative_intensity(query_rt - chemical.rt)
        else:
            return self._get_intensity(chemical.parent, query_rt, which_isotope, which_adduct) * \
                   chemical.parent_mass_prop * chemical.prop_ms2_mass

    def _get_mz(self, chemical, query_rt, which_isotope, which_adduct):
        if chemical.ms_level == 1:
            return (adduct_transformation(chemical.isotopes[which_isotope][0],
                                          self._get_adducts(chemical)[which_adduct][0]) +
                    chemical.chromatogram.get_relative_mz(query_rt - chemical.rt))
        else:
            ms1_parent = chemical
            while ms1_parent.ms_level != 1:
                ms1_parent = chemical.parent
            isotope_transformation = ms1_parent.isotopes[which_isotope][0] - ms1_parent.isotopes[0][0]
            # TODO: Needs improving
            return (adduct_transformation(chemical.isotopes[0][0],
                                         self._get_adducts(chemical)[which_adduct][0]) + isotope_transformation)

    def _isolation_match(self, chemical, query_rt, isolation_windows, which_isotope, which_adduct):
        # assumes list is formated like:
        # [(min_1,max_1),(min_2,max_2),...]
        for window in isolation_windows:
            if window[0] < self._get_mz(chemical, query_rt, which_isotope, which_adduct) <= window[1]:
                return True
        return False


class DsDAMassSpec(IndependentMassSpectrometer):

    def run(self, schedule, pbar=None):
        self.schedule = schedule
        self.time = schedule["targetTime"][0]
        self.fire_event(MassSpectrometer.ACQUISITION_STREAM_OPENING)

        try:
            last_ms1_id = 0
            while len(self.queue) != 0:
                scan_params = self.queue.pop(0)

                # make a scan
                target_time = scan_params.get(ScanParameters.TIME)
                scan = self._get_scan(target_time, scan_params)

                # set scan duration
                try:
                    next_time = self.queue[0].get(ScanParameters.TIME)
                except IndexError:
                    next_time = 1
                scan.scan_duration = next_time - target_time

                # update precursor scan id
                if scan.ms_level == 1:
                    last_ms1_id = scan.scan_id
                else:
                    precursor = scan_params.get(ScanParameters.PRECURSOR)
                    if precursor is not None:
                        precursor.precursor_scan_id = last_ms1_id
                        self.precursor_information[precursor].append(scan)

                # notify controller about this scan
                self.fire_event(self.MS_SCAN_ARRIVED, scan)

                # increase mass spec time
                self.idx += 1
                self.time += scan.scan_duration

                # print a progress bar if provided
                if pbar is not None:
                    elapsed = self.time
                    pbar.update(elapsed)
                    # TODO: fix error bar

        finally:
            self.fire_event(MassSpectrometer.ACQUISITION_STREAM_CLOSING)
            if pbar is not None:
                pbar.close()

# class ThermoFusionMassSpectrometer:

#     def __next__(self):
#         raise NotImplementedError()