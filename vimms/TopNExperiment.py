import os
from collections import defaultdict

import numpy as np
import pandas as pd
import pymzml

from vimms.Chemicals import get_absolute_intensity
from vimms.Common import save_obj, create_if_not_exist, get_rt, find_nearest_index_in_array
from vimms.Controller import TopNController
from vimms.DataGenerator import DataSource, PeakDensityEstimator
from vimms.MassSpec import IndependentMassSpectrometer, FragmentationEvent
from vimms.Roi import make_roi, RoiToChemicalCreator


########################################################################################################################
# Data extraction methods
########################################################################################################################

def get_chemicals(mzML_file, mz_tol, min_ms1_intensity, start_rt, stop_rt, min_length=1):
    '''
    Extract ROI from an mzML file and turn them into UnknownChemical objects
    :param mzML_file: input mzML file
    :param mz_tol: mz tolerance for ROI extraction
    :param min_ms1_intensity: ROI will only be kept if it has one point above this threshold
    :param start_rt: start RT to extract ROI
    :param stop_rt: end RT to extract ROI
    :return: a list of UnknownChemical objects
    '''
    min_intensity = 0
    good_roi, junk = make_roi(mzML_file, mz_tol=mz_tol, mz_units='ppm', min_length=min_length,
                              min_intensity=min_intensity, start_rt=start_rt, stop_rt=stop_rt)

    # keep ROI that have at least one point above the minimum to fragment threshold
    keep = []
    for roi in good_roi:
        if np.count_nonzero(np.array(roi.intensity_list) > min_ms1_intensity) > 0:
            keep.append(roi)

    ps = None  # unused
    rtcc = RoiToChemicalCreator(ps, keep)
    chemicals = np.array(rtcc.chemicals)
    return chemicals


########################################################################################################################
# Codes to set up experiments
########################################################################################################################

def fit_densities(mzml_path, fragfile, min_rt, max_rt):
    ds = DataSource()
    ds.load_data(mzml_path, file_name=fragfile)
    kde_min_ms1_intensity = 0  # min intensity to be selected for kdes
    kde_min_ms2_intensity = 0
    densities = PeakDensityEstimator(kde_min_ms1_intensity, kde_min_ms2_intensity, min_rt, max_rt, plot=False)
    densities.kde(ds, fragfile, 2, bandwidth_mz_intensity_rt=1.0, bandwidth_n_peaks=1.0)
    return densities


def run_experiment(param):
    '''
    Runs a Top-N experiment
    :param param: the experimental parameters
    :return: the analysis name that has been successfully ran
    '''
    analysis_name = param['analysis_name']
    mzml_out = param['mzml_out']
    pickle_out = param['pickle_out']
    N = param['N']
    rt_tol = param['rt_tol']

    if os.path.isfile(mzml_out) and os.path.isfile(pickle_out):
        print('Skipping %s' % (analysis_name))
    else:
        print('Processing %s' % (analysis_name))
        density = param['density']
        if density is None:  # extract density from the fragmenatation file
            mzml_path = param['mzml_path']
            fragfiles = param['fragfiles']
            fragfile = fragfiles[(N, rt_tol,)]
            min_rt = param['min_rt']
            max_rt = param['max_rt']
            density = fit_densities(mzml_path, fragfile, min_rt, max_rt)

        mass_spec = IndependentMassSpectrometer(param['ionisation_mode'], param['data'], density)
        controller = TopNController(mass_spec, param['N'], param['isolation_window'],
                                    param['mz_tol'], param['rt_tol'], param['min_ms1_intensity'])
        controller.run(param['min_rt'], param['max_rt'], progress_bar=param['pbar'])
        controller.write_mzML(analysis_name, mzml_out)
        save_obj(controller, pickle_out)
        return analysis_name


def run_parallel_experiment(params):
    '''
    Runs experiments in parallel using iParallel library
    :param params: the experimental parameter
    :return: None
    '''
    import ipyparallel as ipp
    rc = ipp.Client()
    dview = rc[:]  # use all engines​
    with dview.sync_imports():
        pass

    analysis_names = dview.map_sync(run_experiment, params)
    for analysis_name in analysis_names:
        print(analysis_name)


def run_serial_experiment(params):
    '''
    Runs experiments serially
    :param params: the experimental parameter
    :return: None
    '''
    total = len(params)
    for i in range(len(params)):
        param = params[i]
        print('Processing \t%d/%d\t%s' % (i + 1, total, param['analysis_name']))
        run_experiment(param)


def get_params(experiment_name, Ns, rt_tols, mz_tol, isolation_window, ionisation_mode, data, density,
               min_ms1_intensity, min_rt, max_rt,
               out_dir, pbar, mzml_path=None, fragfiles=None):
    '''
    Creates a list of experimental parameters
    :param experiment_name: current experimental name
    :param Ns: possible values of N in top-N to test
    :param rt_tols: possible values of DEW to test
    :param mz_tol: Top-N controller parameter: the m/z window (ppm) to prevent the same precursor ion to be fragmented again
    :param isolation_window: Top-N controller parameter: the m/z window (ppm) to prevent the same precursor ion to be fragmented again
    :param ionisation_mode: Top-N controller parameter: either positive or negative
    :param data: chemicals to fragment
    :param density: trained densities to sample values during simulatin
    :param min_ms1_intensity: Top-N controller parameter: minimum ms1 intensity to fragment
    :param min_rt: start RT to simulate
    :param max_rt: end RT to simulate
    :param out_dir: output directory
    :param pbar: progress bar to update
    :return: a list of parameters
    '''
    create_if_not_exist(out_dir)
    print('N =', Ns)
    print('rt_tol =', rt_tols)
    params = []
    for N in Ns:
        for rt_tol in rt_tols:
            analysis_name = 'experiment_%s_N_%d_rttol_%d' % (experiment_name, N, rt_tol)
            mzml_out = os.path.join(out_dir, '%s.mzML' % analysis_name)
            pickle_out = os.path.join(out_dir, '%s.p' % analysis_name)
            param_dict = {
                'N': N,
                'mz_tol': mz_tol,
                'rt_tol': rt_tol,
                'min_ms1_intensity': min_ms1_intensity,
                'isolation_window': isolation_window,
                'ionisation_mode': ionisation_mode,
                'data': data,
                'density': density,
                'min_rt': min_rt,
                'max_rt': max_rt,
                'analysis_name': analysis_name,
                'mzml_out': mzml_out,
                'pickle_out': pickle_out,
                'pbar': pbar
            }
            if mzml_path is not None:
                param_dict['mzml_path'] = mzml_path
            if fragfiles is not None:
                param_dict['fragfiles'] = fragfiles
            params.append(param_dict)
    print('len(params) =', len(params))
    return params


########################################################################################################################
# Extract precursor information from mzML file
########################################################################################################################


def get_peaks(spectrum):
    mzs = spectrum.mz
    rts = [get_rt(spectrum)] * len(mzs)
    intensities = spectrum.i
    peaklist = np.stack([mzs, rts, intensities], axis=1)
    return peaklist


def find_precursor_ms1(precursor, last_ms1_peaklist, last_ms1_scan_no, isolation_window):
    precursor_mz = precursor['mz']
    precursor_intensity = precursor['i']

    # find mz in the last ms1 scan that fall within isolation window
    mzs = last_ms1_peaklist[:, 0]
    diffs = abs(mzs - precursor_mz) < isolation_window
    idx = np.nonzero(diffs)[0]

    if len(idx) == 0:  # should never happen!?
        raise ValueError('Cannot find precursor peak (%f, %f) in the last ms1 scan %d' %
                         (precursor_mz, precursor_intensity, last_ms1_scan_no))

    elif len(idx) == 1:  # only one is found
        selected_ms1_idx = idx[0]

    else:  # found multilple possible ms1 peak, select the largest intensity
        possible_ms1 = last_ms1_peaklist[idx, :]
        possible_intensities = possible_ms1[:, 2]
        closest = np.argmax(possible_intensities)
        selected_ms1_idx = idx[closest]

    selected_ms1 = last_ms1_peaklist[selected_ms1_idx, :]
    return selected_ms1, selected_ms1_idx


def find_precursor_peaks(precursor, last_ms1_peaklist, last_ms1_scan_no, isolation_window=0.5):
    selected_ms1, selected_ms1_idx = find_precursor_ms1(precursor, last_ms1_peaklist,
                                                        last_ms1_scan_no, isolation_window)
    selected_ms1_mz = selected_ms1[0]
    selected_ms1_rt = selected_ms1[1]
    selected_ms1_intensity = selected_ms1[2]
    res = [last_ms1_scan_no, selected_ms1_rt, selected_ms1_mz, selected_ms1_intensity]
    return res


def get_precursor_info(fragfile):
    run = pymzml.run.Reader(fragfile, obo_version='4.0.1',
                            MS1_Precision=5e-6,
                            extraAccessions=[('MS:1000016', ['value', 'unitName'])])

    last_ms1_peaklist = None
    last_ms1_scan_no = 0
    isolation_window = 0.5  # Dalton
    data = []
    for scan_no, scan in enumerate(run):
        if scan.ms_level == 1:  # save the last ms1 scan that we've seen
            last_ms1_peaklist = get_peaks(scan)
            last_ms1_scan_no = scan_no

        # TODO: it's better to use the "isolation window target m/z" field in the mzML file for matching
        precursors = scan.selected_precursors
        if len(precursors) > 0:
            assert len(precursors) == 1  # assume exactly 1 precursor peak for each ms2 scan
            precursor = precursors[0]

            try:
                scan_rt = get_rt(scan)
                precursor_mz = precursor['mz']
                precursor_intensity = precursor['i']
                res = find_precursor_peaks(precursor, last_ms1_peaklist, last_ms1_scan_no,
                                           isolation_window=isolation_window)
                row = [scan_no, scan_rt, precursor_mz, precursor_intensity]
                row.extend(res)
                data.append(row)
            except ValueError as e:
                print(e)
            except KeyError as e:
                continue  # sometimes we can't find the intensity value precursor['i'] in precursors

    columns = ['ms2_scan_id', 'ms2_scan_rt', 'ms2_precursor_mz', 'ms2_precursor_intensity',
               'ms1_scan_id', 'ms1_scan_rt', 'ms1_mz', 'ms1_intensity']
    df = pd.DataFrame(data, columns=columns)

    # select only rows where we are sure of the matching, i.e. the intensity values aren't too different
    df['intensity_diff'] = np.abs(df['ms2_precursor_intensity'] - df['ms1_intensity'])
    idx = (df['intensity_diff'] < 0.1)
    ms1_df = df[idx]
    return ms1_df


def _get_chem_indices(query_mz, query_rt, min_mzs, max_mzs, min_rts, max_rts):
    rtmin_check = min_rts < query_rt
    rtmax_check = query_rt < max_rts
    mzmin_check = min_mzs < query_mz
    mzmax_check = query_mz < max_mzs
    idx = np.nonzero(rtmin_check & rtmax_check & mzmin_check & mzmax_check)[0]
    return idx


def get_chem_to_frag_events(chemicals, ms1_df):
    # used for searching later
    min_rts = np.array([min(chem.chromatogram.raw_rts) for chem in chemicals])
    max_rts = np.array([max(chem.chromatogram.raw_rts) for chem in chemicals])
    min_mzs = np.array([min(chem.chromatogram.raw_mzs) for chem in chemicals])
    max_mzs = np.array([max(chem.chromatogram.raw_mzs) for chem in chemicals])

    # loop over each fragmentation event in ms1_df, attempt to match it to chemicals
    chem_to_frag_events = defaultdict(list)
    for idx, row in ms1_df.iterrows():
        query_rt = row['ms1_scan_rt']
        query_mz = row['ms1_mz']
        query_intensity = row['ms1_intensity']
        scan_id = row['ms2_scan_id']

        chem = None
        idx = _get_chem_indices(query_mz, query_rt, min_mzs, max_mzs, min_rts, max_rts)
        if len(idx) == 1:  # single match
            chem = chemicals[idx][0]

        elif len(
                idx) > 1:  # multiple matches, find the closest in intensity to query_intensity at the time of fragmentation
            matches = chemicals[idx]
            possible_intensities = np.array([get_absolute_intensity(chem, query_rt) for chem in matches])
            closest = find_nearest_index_in_array(possible_intensities, query_intensity)
            chem = matches[closest]

        # create frag event for the given chem
        if chem is not None:
            ms_level = 2
            peaks = []  # we don't know which ms2 peaks are linked to this chem object
            # key = get_key(chem)
            frag_event = FragmentationEvent(chem, query_rt, ms_level, peaks, scan_id)
            chem_to_frag_events[chem].append(frag_event)
    return dict(chem_to_frag_events)


def get_N_rt_tol_from_qcb_filename(fragfile):
    base = os.path.basename(fragfile)
    base = os.path.splitext(base)[0]
    tokens = base.split('_')
    N = int(tokens[1][1:])
    rt_tol = int(tokens[2][3:])
    return N, rt_tol
