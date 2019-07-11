import numpy as np
import pandas as pd
import scipy.stats

from vimms.Chemicals import UnknownChemical
from vimms.Common import LoggerMixin, takeClosest, PROTON_MASS


class Chromatogram(object):

    def get_relative_intensity(self, query_rt):
        raise NotImplementedError()

    def get_relative_mz(self, query_rt):
        raise NotImplementedError()

    def _rt_match(self, rt):
        raise NotImplementedError()


class EmpiricalChromatogram(Chromatogram):
    """
    Empirical Chromatograms to be used within Chemicals
    """

    def __init__(self, rts, mzs, intensities, single_point_length=0.9):
        self.raw_rts = rts
        self.raw_mzs = mzs
        self.raw_intensities = intensities
        # ensures that all arrays are in sorted order
        if len(rts) > 1:
            p = rts.argsort()
            rts = rts[p]
            mzs = mzs[p]
            intensities = intensities[p]
        else:
            rts = np.array([rts[0] - 0.5 * single_point_length, rts[0] + 0.5 * single_point_length])
            mzs = np.array([mzs[0], mzs[0]])
            intensities = np.array([intensities[0], intensities[0]])
        # normalise arrays
        self.rts = rts - min(rts)
        self.mzs = mzs - np.mean(mzs)  # may want to just set this to 0 and remove from input
        self.intensities = intensities / max(intensities)
        # chromatogramDensityNormalisation(rts, intensities)

        self.min_rt = min(self.rts)
        self.max_rt = max(self.rts)

    def get_relative_intensity(self, query_rt):
        if not self._rt_match(query_rt):
            return None
        else:
            neighbours_which = self._get_rt_neighbours_which(query_rt)
            intensity_below = self.intensities[neighbours_which[0]]
            intensity_above = self.intensities[neighbours_which[1]]
            return intensity_below + (intensity_above - intensity_below) * self._get_distance(query_rt)

    def get_relative_mz(self, query_rt):
        if not self._rt_match(query_rt):
            return None
        else:
            neighbours_which = self._get_rt_neighbours_which(query_rt)
            mz_below = self.mzs[neighbours_which[0]]
            mz_above = self.mzs[neighbours_which[1]]
            return mz_below + (mz_above - mz_below) * self._get_distance(query_rt)

    def _get_rt_neighbours(self, query_rt):
        which_rt_below, which_rt_above = self._get_rt_neighbours_which(query_rt)
        rt_below = self.rts[which_rt_below]
        rt_above = self.rts[which_rt_above]
        return [rt_below, rt_above]

    def _get_rt_neighbours_which(self, query_rt):
        # find the max index of self.rts smaller than query_rt
        pos = np.where(self.rts <= query_rt)[0]
        which_rt_below = pos[-1]

        # take the min index of self.rts larger than query_rt
        pos = np.where(self.rts > query_rt)[0]
        which_rt_above = pos[0]
        return [which_rt_below, which_rt_above]

    def _get_distance(self, query_rt):
        rt_below, rt_above = self._get_rt_neighbours(query_rt)
        return (query_rt - rt_below) / (rt_above - rt_below)

    def _rt_match(self, query_rt):
        return self.min_rt < query_rt < self.max_rt

    def __eq__(self, other):
        if not isinstance(other, EmpiricalChromatogram):
            # don't attempt to compare against unrelated types
            return NotImplemented

        return np.array_equal(sorted(self.raw_mzs), sorted(other.raw_mzs)) and \
               np.array_equal(sorted(self.raw_rts), sorted(other.raw_rts)) and \
               np.array_equal(sorted(self.raw_intensities), sorted(other.raw_intensities))


# Make this more generalisable. Make scipy.stats... as input, However this makes it difficult to do the cutoff
class FunctionalChromatogram(Chromatogram):
    """
    Functional Chromatograms to be used within Chemicals
    """

    def __init__(self, distribution, parameters, cutoff=0.01):
        self.cutoff = cutoff
        self.mz = 0
        if distribution == "normal":
            self.distrib = scipy.stats.norm(parameters[0], parameters[1])
        elif distribution == "gamma":
            self.distrib = scipy.stats.gamma(parameters[0], parameters[1], parameters[2])
        elif distribution == "uniform":
            self.distrib = scipy.stats.uniform(parameters[0], parameters[1])
        else:
            raise NotImplementedError("distribution not implemented")
        self.min_rt = 0
        self.max_rt = self.distrib.ppf(1 - (self.cutoff / 2)) - self.distrib.ppf(self.cutoff / 2)

    def get_relative_intensity(self, query_rt):
        if self._rt_match(query_rt) == False:
            return None
        else:
            return (self.distrib.pdf(query_rt + self.distrib.ppf(self.cutoff / 2)) * (1 / (1 - self.cutoff)))

    def get_relative_mz(self, query_rt):
        if self._rt_match(query_rt) == False:
            return None
        else:
            return self.mz

    def _rt_match(self, query_rt):
        if query_rt < 0 or query_rt > self.distrib.ppf(1 - (self.cutoff / 2)) - self.distrib.ppf(self.cutoff / 2):
            return False
        else:
            return True


class ChromatogramCreator(LoggerMixin):
    def __init__(self, xcms_output):
        # load the chromatograms and sort by intensity ascending
        chromatograms, chemicals = self._load_chromatograms(xcms_output)
        max_intensities = np.array([max(c.raw_intensities) for c in chromatograms])
        idx = np.argsort(max_intensities)
        self.chromatograms = chromatograms[idx]
        self.max_intensities = max_intensities[idx]
        self.chemicals = chemicals

    def sample(self, intensity=None):
        """
        Samples a chromatogram
        :param intensity: the intensity to select the closest chromatogram
        :return: a Chromatogram object
        """
        if intensity == None:  # randomly sample chromatograms
            selected = np.random.choice(len(self.chromatograms), 1)[0]
        else:  # find the chromatogram closest to the intensity
            selected = takeClosest(self.max_intensities, intensity)
        return self.chromatograms[selected]

    def _load_chromatograms(self, xcms_output):
        """
        Load CSV file of chromatogram information exported by the XCMS script 'process_data.R'
        :param df_file: the input csv file exported by the script (in gzip format)
        :return: a list of Chromatogram objects
        """
        df = pd.read_csv(xcms_output, compression='gzip')
        peak_ids = df.id.unique()
        groups = df.groupby('id')
        chroms = []
        chems = []
        for i in range(len(peak_ids)):
            if i % 5000 == 0:
                self.logger.debug('Loading {} chromatograms'.format(i))

            # raise numpy warning as exception, see https://stackoverflow.com/questions/15933741/how-do-i-catch-a-numpy-warning-like-its-an-exception-not-just-for-testing
            with np.errstate(divide='raise'):
                try:
                    pid = peak_ids[i]
                    chrom, chem = self._get_xcms_chromatograms(groups, pid)
                    if len(chrom.rts) > 1:  # chromatograms should have more than one single data point
                        chroms.append(chrom)
                        chems.append(chem)
                except FloatingPointError:
                    self.logger.debug('Invalid chromatogram {}'.format(i))
                except ZeroDivisionError:
                    self.logger.debug('Invalid chromatogram {}'.format(i))

        self.logger.info('Created {} chromatograms'.format(i))
        return np.array(chroms), np.array(chems)

    def _get_xcms_chromatograms(self, groups, pid):
        selected = groups.get_group(pid)
        mz = self._get_value(selected, 'mz')
        rt = self._get_value(selected, 'rt')
        max_intensity = self._get_value(selected, 'maxo')
        rts = self._get_values(selected, 'rt_values')
        mzs = self._get_values(selected, 'mz_values')
        intensities = self._get_values(selected, 'intensity_values')
        assert len(rts) == len(mzs)
        assert len(rts) == len(intensities)
        chrom = EmpiricalChromatogram(rts, mzs, intensities)

        # TODO: this only works for positive mode data
        # Correct the positively charged ions by substracting the mass of a proton
        mz = mz - PROTON_MASS
        chem = UnknownChemical(mz, rt, max_intensity, chrom, None)
        return chrom, chem

    def _get_values(self, df, column_name):
        return df[column_name].values

    def _get_value(self, df, column_name):
        return self._get_values(df, column_name)[0]
