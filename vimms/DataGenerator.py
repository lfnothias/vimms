import copy
import glob
import os

import math
import numpy as np
import pandas as pd
import pylab as plt
import pymzml
from sklearn.neighbors import KernelDensity

from vimms.Common import LoggerMixin, MZ, INTENSITY, RT, N_PEAKS, SCAN_DURATION, MZ_INTENSITY_RT


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


def filter_df(df, min_ms1_intensity, rt_range, mz_range):
    # filter by rt range
    if rt_range is not None:
        df = df[(df['rt'] > rt_range[0][0]) & (df['rt'] < rt_range[0][1])]

    # filter by mz range
    if mz_range is not None:
        df = df[(df['rt'] > mz_range[0][0]) & (df['rt'] < mz_range[0][1])]

    # filter by min intensity
    intensity_col = 'maxo'
    if min_ms1_intensity is not None:
        df = df[(df[intensity_col] > min_ms1_intensity)]
    return df


class DataSource(LoggerMixin):
    """
    A class to load and extract centroided peaks from CSV and mzML files.
    :param min_ms1_intensity: minimum ms1 intensity for filtering
    :param min_ms2_intensity: maximum ms2 intensity for filtering
    :param min_rt: minimum RT for filtering
    :param max_rt: maximum RT for filtering
    """

    def __init__(self):
        # A dictionary that stores the actual pymzml spectra for each filename
        self.file_spectra = {}  # key: filename, value: a dict where key is scan_number and value is spectrum

        # A dictionary to store the distribution on scan durations for each ms_level in each file
        self.file_scan_durations = {}  # key: filename, value: a dict with key ms level and value scan durations

        # pymzml parameters
        self.ms1_precision = 5e-6
        self.obo_version = '4.0.1'

        # xcms peak picking results, if any
        self.df = None

    def load_data(self, mzml_path, file_name=None):
        """
        Loads data and generate peaks from mzML files. The resulting peak objects will not have chromatographic peak
        shapes, because no peak picking has been performed yet.
        :param mzml_path: the input folder containing the mzML files
        :return: nothing, but the instance variable file_spectra and scan_durations are populated
        """
        file_scan_durations = {}  # key: filename, value: a dict where key is ms level and value is scan durations
        file_spectra = {}  # key: filename, value: a dict where key is scan_number and value is spectrum
        for filename in glob.glob(os.path.join(mzml_path, '*.mzML')):
            fname = os.path.basename(filename)
            if file_name is not None and fname != file_name:
                continue

            self.logger.info('Loading %s' % fname)
            file_spectra[fname] = {}
            file_scan_durations[fname] = {
                (1, 1): [],
                (1, 2): [],
                (2, 1): [],
                (2, 2): []
            }

            run = pymzml.run.Reader(filename, obo_version=self.obo_version,
                                    MS1_Precision=self.ms1_precision,
                                    extraAccessions=[('MS:1000016', ['value', 'unitName'])])
            for scan_no, scan in enumerate(run):
                # store scans
                file_spectra[fname][scan_no] = scan

                # store scan durations
                if scan_no == 0:
                    previous_level = scan['ms level']
                    old_rt = self._get_rt(scan)
                    continue
                rt = self._get_rt(scan)
                current_level = scan['ms level']
                rt_steps = file_scan_durations[fname]
                previous_duration = rt - old_rt
                rt_steps[(previous_level, current_level)].append(previous_duration)
                previous_level = current_level
                old_rt = rt

        self.file_scan_durations = file_scan_durations
        self.file_spectra = file_spectra

    def load_xcms_output(self, xcms_filename):
        self.df = pd.read_csv(xcms_filename)

    def plot_data(self, file_name, ms_level=1, min_rt=None, max_rt=None, max_data=100000):
        data_types = [MZ, INTENSITY, RT, N_PEAKS, SCAN_DURATION]
        for data_type in data_types:
            if data_type == SCAN_DURATION:
                X = self.get_scan_durations(file_name)
                self.plot_histogram(X, data_type)
            elif data_type == N_PEAKS:
                X = self.get_n_peaks(file_name, ms_level, min_rt=min_rt, max_rt=max_rt)
            else:
                X = self.get_data(data_type, file_name, ms_level, min_rt=min_rt, max_rt=max_rt, max_data=max_data)
                if data_type == INTENSITY:
                    X = np.log(X)
                self.plot_histogram(X, data_type)
                self.plot_boxplot(X, data_type)

    def plot_histogram(self, X, data_type, n_bins=100):
        """
        Makes a histogram plot on the distribution of the item of interest
        :param X: a numpy array
        :param bins: number of histogram bins
        :return: nothing. A plot is shown.
        """
        if data_type == SCAN_DURATION:
            rt_steps = X
            for key, rt_list in rt_steps.items():
                try:
                    bins = np.linspace(min(rt_list), max(rt_list), n_bins)
                    plt.figure()
                    plt.hist(rt_list, bins=bins)
                    plt.title(key)
                    plt.show()
                except ValueError:
                    continue
        else:
            plt.figure()
            _ = plt.hist(X, bins=n_bins)
            plt.plot(X[:, 0], np.full(X.shape[0], -0.01), '|k')
            plt.title('Histogram for %s -- shape %s' % (data_type, str(X.shape)))
            plt.show()

    def plot_boxplot(self, X, data_type):
        """
        Makes a boxplot on the distribution of the item of interest
        :param X: a numpy array
        :return: nothing. A plot is shown.
        """
        plt.figure()
        _ = plt.boxplot(X)
        plt.title('Boxplot for %s -- shape %s' % (data_type, str(X.shape)))
        plt.show()

    def plot_peak(self, peak):
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(peak.rt_values, peak.intensity_values)
        axarr[1].plot(peak.rt_values, peak.mz_values, linestyle='None', marker='o', markersize=1.0, color='b')

    def get_data(self, data_type, filename, ms_level, min_intensity=None,
                 min_rt=None, max_rt=None, log=False, max_data=100000):
        """
        Retrieves values as numpy array
        :param data_type: data_type is 'mz', 'rt', 'intensity' or 'n_peaks'
        :param filename: the mzml filename or None for all files
        :param ms_level: level 1 or 2
        :param min_intensity: minimum ms2 intensity for thresholding
        :param min_rt: minimum RT value for thresholding
        :param max_rt: max RT value for thresholding
        :param log: if true, the returned values will be logged
        :return: an Nx1 numpy array of all the values requested
        """
        # if xcms peak picking results are provided, use that instead
        if self.df is not None:
            self.logger.info('Using values from XCMS peaklist')

            # remove rows in the peak picked dataframe that are outside threshold values
            df = filter_df(self.df, min_intensity, [[min_rt, max_rt]], None)

            # extract the values we need
            if data_type == MZ:
                X = df['mz'].values
            elif data_type == RT:
                # we use rt for the starting value for the chemical to elute
                X = df['rt'].values
            elif data_type == INTENSITY:
                X = df['maxo'].values
            elif data_type == MZ_INTENSITY_RT:
                X = df[['mz', 'maxo', 'rt']].values

        else:  # else we get the values by reading from the scans in mzML files directly
            self.logger.info('Using values from scans')

            # get spectra from either one file or all files
            if filename is None:  # use all spectra
                all_spectra = []
                for f in self.file_spectra:
                    spectra_for_f = list(self.file_spectra[f].values())
                    all_spectra.extend(spectra_for_f)
            else:  # use spectra for that file only
                all_spectra = self.file_spectra[filename].values()

            # loop through spectrum and get all peaks above threshold
            values = []
            for spectrum in all_spectra:
                # if wrong ms level, skip this spectrum
                if spectrum.ms_level != ms_level:
                    continue

                # collect all valid Peak objects in a spectrum
                spectrum_peaks = []
                for mz, intensity in spectrum.peaks('raw'):
                    rt = self._get_rt(spectrum)
                    p = Peak(mz, rt, intensity, spectrum.ms_level)
                    if self._valid_peak(p, min_intensity, min_rt, max_rt):
                        spectrum_peaks.append(p)

                if data_type == MZ_INTENSITY_RT:  # used when fitting m/z, rt and intensity together for the manuscript
                    mzs = list(getattr(x, MZ) for x in spectrum_peaks)
                    intensities = list(getattr(x, INTENSITY) for x in spectrum_peaks)
                    rts = list(getattr(x, RT) for x in spectrum_peaks)
                    values.extend(list(zip(mzs, intensities, rts)))

                else:  # MZ, INTENSITY or RT separately
                    attrs = list(getattr(x, data_type) for x in spectrum_peaks)
                    values.extend(attrs)

            X = np.array(values)

        # log-transform if necessary
        if log:
            if data_type == MZ_INTENSITY_RT:  # just log the intensity part
                X[:, 1] = np.log(X[:, 1])
            else:
                X = np.log(X)

        # pick random samples
        try:
            idx = np.arange(len(X))
            rnd_idx = np.random.choice(idx, size=int(max_data), replace=False)
            sampled_X = X[rnd_idx]
        except ValueError:
            sampled_X = X

        # return values
        if data_type == MZ_INTENSITY_RT:
            return sampled_X  # it's already a Nx2 or Nx3 array
        else:
            # convert into Nx1 array
            return sampled_X[:, np.newaxis]

    def get_n_peaks(self, filename, ms_level, min_intensity=None, min_rt=None, max_rt=None):
        # get spectra from either one file or all files
        if filename is None:  # use all spectra
            all_spectra = []
            for f in self.file_spectra:
                spectra_for_f = list(self.file_spectra[f].values())
                all_spectra.extend(spectra_for_f)
        else:  # use spectra for that file only
            all_spectra = self.file_spectra[filename].values()

        # loop through spectrum and get all peaks above threshold
        values = []
        for spectrum in all_spectra:
            # if wrong ms level, skip this spectrum
            if spectrum.ms_level != ms_level:
                continue

            # collect all valid Peak objects in a spectrum
            spectrum_peaks = []
            for mz, intensity in spectrum.peaks('raw'):
                rt = self._get_rt(spectrum)
                p = Peak(mz, rt, intensity, spectrum.ms_level)
                if self._valid_peak(p, min_intensity, min_rt, max_rt):
                    spectrum_peaks.append(p)

            # collect the data points we need into a list
            n_peaks = len(spectrum_peaks)
            if n_peaks > 0:
                values.append(n_peaks)

        # convert into Nx1 array
        X = np.array(values)
        return X[:, np.newaxis]

    def get_scan_durations(self, fname):
        if fname is None:  # if no filename, then combine all the dictionary keys
            combined = None
            for f in self.file_scan_durations:
                if combined is None:  # copy first one
                    combined = copy.deepcopy(self.file_scan_durations[f])
                else:  # and extend with the subsequent ones
                    for key in combined:
                        combined[key].extend(self.file_scan_durations[f][key])
        else:
            combined = self.file_scan_durations[fname]
        return combined

    def _get_rt(self, spectrum):
        rt, units = spectrum.scan_time
        if units == 'minute':
            rt *= 60.0
        return rt

    def _valid_peak(self, peak, min_intensity, min_rt, max_rt):
        if min_intensity is not None and peak.intensity < min_intensity:
            return False
        elif min_rt is not None and peak.rt < min_rt:
            return False
        elif max_rt is not None and peak.rt > max_rt:
            return False
        else:
            return True


class PeakDensityEstimator(LoggerMixin):
    """A class to perform kernel density estimation for peak data by fitting m/z, RT and intensity together.
    Takes as input a DataSource class."""

    def __init__(self, min_ms1_intensity, min_ms2_intensity, min_rt, max_rt, plot=False):
        self.kdes = {}
        self.kernel = 'gaussian'
        self.min_ms1_intensity = min_ms1_intensity
        self.min_ms2_intensity = min_ms2_intensity
        self.min_rt = min_rt
        self.max_rt = max_rt
        self.plot = plot
        self.file_scan_durations = None

    def kde(self, data_source, filename, ms_level, bandwidth_mz_intensity_rt=1.0,
            bandwidth_n_peaks=1.0, bandwidth_scan_durations=0.01, max_data=100000):
        params = [
            {'data_type': MZ_INTENSITY_RT, 'bandwidth': bandwidth_mz_intensity_rt},
            {'data_type': N_PEAKS, 'bandwidth': bandwidth_n_peaks}
        ]

        # not really kde, but we store it anyway for sampling later
        self.file_scan_durations = data_source.get_scan_durations(filename)

        for param in params:
            data_type = param['data_type']

            # get data
            self.logger.debug('Retrieving %s values from %s' % (data_type, data_source))
            if data_type == N_PEAKS:
                X = data_source.get_n_peaks(filename, ms_level, min_intensity=min_intensity,
                                            min_rt=self.min_rt, max_rt=self.max_rt)
            else:
                min_intensity = self.min_ms1_intensity if ms_level == 1 else self.min_ms2_intensity
                log = True if data_type == MZ_INTENSITY_RT else False
                X = data_source.get_data(data_type, filename, ms_level, min_intensity=min_intensity,
                                         min_rt=self.min_rt, max_rt=self.max_rt, log=log, max_data=max_data)

            # fit kde
            bandwidth = param['bandwidth']
            kde = KernelDensity(kernel=self.kernel, bandwidth=bandwidth).fit(X)
            self.kdes[(data_type, ms_level)] = kde

            # plot if necessary
            self._plot(kde, X, data_type, filename, bandwidth)

    def sample(self, ms_level, n_sample):
        vals = self.kdes[(MZ_INTENSITY_RT, ms_level)].sample(n_sample)
        return vals

    def n_peaks(self, ms_level, n_sample):
        return self.kdes[(N_PEAKS, ms_level)].sample(n_sample)

    def scan_durations(self, previous_level, current_level, n_sample):
        key = (previous_level, current_level,)
        values = self.file_scan_durations[key]
        return np.random.choice(values, size=n_sample)

    def _plot(self, kde, X, data_type, filename, bandwidth):
        if self.plot:
            if data_type == MZ_INTENSITY_RT:
                self.logger.debug('3D plotting for %s not implemented' % MZ_INTENSITY_RT)
            else:
                fname = 'All' if filename is None else filename
                title = '%s density estimation for %s - bandwidth %.3f' % (data_type, fname, bandwidth)
                X_plot = np.linspace(np.min(X), np.max(X), 1000)[:, np.newaxis]
                log_dens = kde.score_samples(X_plot)
                plt.figure()
                plt.fill_between(X_plot[:, 0], np.exp(log_dens), alpha=0.5)
                plt.plot(X[:, 0], np.full(X.shape[0], -0.01), '|k')
                plt.title(title)
                plt.show()


class PeakSampler(LoggerMixin):
    """A class to sample peaks from a trained density estimator"""

    def __init__(self, density_estimator):
        self.density_estimator = density_estimator

    def sample(self, ms_level, n_peaks=None, min_mz=None, max_mz=None, min_rt=None, max_rt=None, min_intensity=None):
        if n_peaks is None:
            n_peaks = max(self.density_estimator.n_peaks(ms_level, 1).astype(int)[0][0], 0)

        peaks = []
        while len(peaks) < n_peaks:
            vals = self.density_estimator.sample(ms_level, 1)
            intensity = np.exp(vals[0, 1])
            mz = vals[0, 0]
            rt = vals[0, 2]
            p = Peak(mz, rt, intensity, ms_level)
            if self._is_valid(p, min_mz, max_mz, min_rt, max_rt, min_intensity):  # othwerise we just keep rejecting
                peaks.append(p)
        return peaks

    def _is_valid(self, peak, min_mz, max_mz, min_rt, max_rt, min_intensity):
        if peak.intensity < 0:
            return False
        if min_mz is not None and min_mz > peak.mz:
            return False
        if max_mz is not None and max_mz < peak.mz:
            return False
        if min_rt is not None and min_rt > peak.rt:
            return False
        if max_rt is not None and max_rt < peak.rt:
            return False
        if min_intensity is not None and min_intensity > peak.intensity:
            return False
        return True
