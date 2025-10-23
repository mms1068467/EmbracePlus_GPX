"""
HRV-Analysis methods from the association AURA-healthcare.
Includes time-domain and frequency-domain measurements with plotting functions
Github: https://github.com/Aura-healthcare/hrv-analysis
"""
# $ pip install hrv-analysis
#from hrvanalysis import get_time_domain_features, get_frequency_domain_features, plot_psd

import hrvanalysis
import numpy as np
import pandas as pd


def time_domain_features(data, timeframe=10):
    time_domain = data.copy()
    time_domain['SDNN'] = [np.nan] * len(data)
    time_domain['SDSD'] = [np.nan] * len(data)
    time_domain['RMSSD'] = [np.nan] * len(data)
    time_domain['NNI50'] = [np.nan] * len(data)
    time_domain['PNNI50'] = [np.nan] * len(data)
    time_domain['NNI20'] = [np.nan] * len(data)
    time_domain['PNNI20'] = [np.nan] * len(data)
    time_domain['CVSD'] = [np.nan] * len(data)
    time_domain['CVNNI'] = [np.nan] * len(data)
    counter = 0
    i = 0
    # for i in range(0,len(data['IBI'])):
    while i < len(data['IBI']) - 1:
        counter += 1

        if counter == timeframe:
            # print("timeframe index: ", i-(timeframe-1), ":", i+1)
            time_domain_features = hrvanalysis.get_time_domain_features(data.IBI[i - (timeframe - 1):i + 1].tolist())
            time_domain.loc[i - timeframe, 'SDNN'] = time_domain_features['sdnn']
            time_domain.loc[i - timeframe, 'SDSD'] = time_domain_features['sdsd']
            time_domain.loc[i - timeframe, 'RMSSD'] = time_domain_features['rmssd']
            time_domain.loc[i - timeframe, 'NNI50'] = time_domain_features['nni_50']
            time_domain.loc[i - timeframe, 'PNNI50'] = time_domain_features['pnni_50']
            time_domain.loc[i - timeframe, 'NNI20'] = time_domain_features['nni_20']
            time_domain.loc[i - timeframe, 'PNNI20'] = time_domain_features['pnni_20']
            time_domain.loc[i - timeframe, 'CVSD'] = time_domain_features['cvsd']
            time_domain.loc[i - timeframe, 'CVNNI'] = time_domain_features['cvnni']

            i = i - (timeframe - 2)
            counter = 1
        i += 1

    # convert time_iso into time_millis for interpolation
    time_domain['time_iso'] = time_domain["time_iso"].astype(np.int64) / int(1e9)

    # Interpolation with time_millis which has already the right timestamp
    data = time_domain.set_index('time_iso')
    data = data.interpolate(method="linear", order=1)
    time_domain = data.reset_index()

    time_domain.time_iso = pd.to_datetime(time_domain.time_iso, unit='s')

    return time_domain


def frequency_domain_features(data, timeframe=20):
    frequency_domain = data.copy()
    frequency_domain['VLF'] = [np.nan] * len(data)
    frequency_domain['LF'] = [np.nan] * len(data)
    frequency_domain['HF'] = [np.nan] * len(data)
    frequency_domain['LF_HF_RATIO'] = [np.nan] * len(data)
    frequency_domain['LFNU'] = [np.nan] * len(data)
    frequency_domain['HFNU'] = [np.nan] * len(data)

    counter = 0
    i = 0
    # for i in range(0,len(data['IBI'])):
    while i < len(data['IBI']) - 1:
        counter += 1

        if counter == timeframe:
            frequency_domain_features = hrvanalysis.get_frequency_domain_features(data.IBI[i - (timeframe - 1):i + 1].tolist(),
                                                                      sampling_frequency=1)
            frequency_domain.loc[i - timeframe, 'VLF'] = frequency_domain_features['vlf']
            frequency_domain.loc[i - timeframe, 'LF'] = frequency_domain_features['lf']
            frequency_domain.loc[i - timeframe, 'HF'] = frequency_domain_features['hf']
            frequency_domain.loc[i - timeframe, 'LF_HF_RATIO'] = frequency_domain_features['lf_hf_ratio']
            frequency_domain.loc[i - timeframe, 'LFNU'] = frequency_domain_features['lfnu']
            frequency_domain.loc[i - timeframe, 'HFNU'] = frequency_domain_features['hfnu']

            i = i - (timeframe - 2)
            counter = 1
        i += 1

    # convert time_iso into time_millis for interpolation
    frequency_domain['time_iso'] = frequency_domain["time_iso"].astype(np.int64) / int(1e9)

    # Interpolation with time_millis which has already the right timestamp
    data = frequency_domain.set_index('time_iso')
    data = data.interpolate(method="linear", order=1)
    frequency_domain = data.reset_index()

    frequency_domain.time_iso = pd.to_datetime(frequency_domain.time_iso, unit='s')

    return frequency_domain


def low_frequency_domain(data, timeframe=20):
    frequency_domain = data.copy()
    frequency_domain['LF'] = [np.nan] * len(data)

    counter = 0
    i = 0
    # for i in range(0,len(data['IBI'])):
    while i < len(data['IBI']) - 1:
        counter += 1

        if counter == timeframe:
            # print("timeframe index: ", i-(timeframe-1), ":", i+1)
            frequency_domain_features = hrvanalysis.get_frequency_domain_features(data.IBI[i - (timeframe - 1):i + 1].tolist(),
                                                                      sampling_frequency=1)
            frequency_domain.loc[i - timeframe, 'LF'] = frequency_domain_features['lf']

            i = i - (timeframe - 2)
            # print("new i", i, "old i", i + (timeframe-1))

            counter = 1
        i += 1

    # convert time_iso into time_millis for interpolation
    frequency_domain['time_iso'] = frequency_domain["time_iso"].astype(np.int64) / int(1e9)

    # Interpolation with time_millis which has already the right timestamp
    data = frequency_domain.set_index('time_iso')
    data = data.interpolate(method="linear", order=1)
    frequency_domain = data.reset_index()

    frequency_domain.time_iso = pd.to_datetime(frequency_domain.time_iso, unit='s')

    return frequency_domain


# --------------------------------------------------------------------------------------------------------------------
# ----------------------------------AURA - HEALTHCARE FREQUENCY FEATURES (extracted)-----------------------------------
# --------------------------------------------------------------------------------------------------------------------

from typing import List, Tuple
from collections import namedtuple
from scipy import interpolate
from scipy import signal
from astropy.stats import LombScargle

# Frequency Methods name
WELCH_METHOD = "welch"
LOMB_METHOD = "lomb"

# Named Tuple for different frequency bands
VlfBand = namedtuple("Vlf_band", ["low", "high"])
LfBand = namedtuple("Lf_band", ["low", "high"])
HfBand = namedtuple("Hf_band", ["low", "high"])


# ----------------- FREQUENCY DOMAIN FEATURES ----------------- #


def _get_frequency_domain_features(nn_intervals: List[float], method: str = WELCH_METHOD,
                                  sampling_frequency: int = 4, interpolation_method: str = "linear",
                                  vlf_band: namedtuple = VlfBand(0.003, 0.04),
                                  lf_band: namedtuple = LfBand(0.04, 0.15),
                                  hf_band: namedtuple = HfBand(0.15, 0.40)) -> dict:
    """
    Returns a dictionary containing frequency domain features for HRV analyses.
    To our knowledge, you might use this function on short term recordings, from 2 to 5 minutes  \
    window.
    Parameters
    ---------
    nn_intervals : list
        list of Normal to Normal Interval
    method : str
        Method used to calculate the psd. Choice are Welch's FFT or Lomb method.
    sampling_frequency : int
        Frequency at which the signal is sampled. Common value range from 1 Hz to 10 Hz,
        by default set to 4 Hz. No need to specify if Lomb method is used.
    interpolation_method : str
        kind of interpolation as a string, by default "linear". No need to specify if Lomb
        method is used.
    vlf_band : tuple
        Very low frequency bands for features extraction from power spectral density.
    lf_band : tuple
        Low frequency bands for features extraction from power spectral density.
    hf_band : tuple
        High frequency bands for features extraction from power spectral density.
    Returns
    ---------
    frequency_domain_features : dict
        Dictionary containing frequency domain features for HRV analyses. There are details
        about each features below.
    Notes
    ---------
    Details about feature engineering...
    - **total_power** : Total power density spectral
    - **vlf** : variance ( = power ) in HRV in the Very low Frequency (.003 to .04 Hz by default). \
    Reflect an intrinsic rhythm produced by the heart which is modulated primarily by sympathetic \
    activity.
    - **lf** : variance ( = power ) in HRV in the low Frequency (.04 to .15 Hz). Reflects a \
    mixture of sympathetic and parasympathetic activity, but in long-term recordings, it reflects \
    sympathetic activity and can be reduced by the beta-adrenergic antagonist propanolol.
    - **hf**: variance ( = power ) in HRV in the High Frequency (.15 to .40 Hz by default). \
    Reflects fast changes in beat-to-beat variability due to parasympathetic (vagal) activity. \
    Sometimes called the respiratory band because it corresponds to HRV changes related to the \
    respiratory cycle and can be increased by slow, deep breathing (about 6 or 7 breaths per \
    minute) and decreased by anticholinergic drugs or vagal blockade.
    - **lf_hf_ratio** : lf/hf ratio is sometimes used by some investigators as a quantitative \
    mirror of the sympatho/vagal balance.
    - **lfnu** : normalized lf power.
    - **hfnu** : normalized hf power.
    References
    ----------
    .. [1] Heart rate variability - Standards of measurement, physiological interpretation, and \
    clinical use, Task Force of The European Society of Cardiology and The North American Society \
    of Pacing and Electrophysiology, 1996
    .. [2] Signal Processing Methods for Heart Rate Variability - Gari D. Clifford, 2002
    """

    # ----------  Handle pandas series  ---------- #

    nn_intervals = list(nn_intervals)

    # ----------  Compute frequency & Power spectral density of signal  ---------- #
    freq, psd = _get_freq_psd_from_nn_intervals(nn_intervals=nn_intervals, method=method,
                                                sampling_frequency=sampling_frequency,
                                                interpolation_method=interpolation_method,
                                                vlf_band=vlf_band, hf_band=hf_band)

    # ---------- Features calculation ---------- #
    frequency_domain_features = _get_features_from_psd(freq=freq, psd=psd,
                                                       vlf_band=vlf_band,
                                                       lf_band=lf_band,
                                                       hf_band=hf_band)

    return frequency_domain_features


def _get_freq_psd_from_nn_intervals(nn_intervals: List[float], method: str = WELCH_METHOD,
                                    sampling_frequency: int = 4,
                                    interpolation_method: str = "linear",
                                    vlf_band: namedtuple = VlfBand(0.003, 0.04),
                                    hf_band: namedtuple = HfBand(0.15, 0.40)) -> Tuple:
    """
    Returns the frequency and power of the signal.
    Parameters
    ---------
    nn_intervals : list
        list of Normal to Normal Interval
    method : str
        Method used to calculate the psd. Choice are Welch's FFT or Lomb method.
    sampling_frequency : int
        Frequency at which the signal is sampled. Common value range from 1 Hz to 10 Hz,
        by default set to 7 Hz. No need to specify if Lomb method is used.
    interpolation_method : str
        Kind of interpolation as a string, by default "linear". No need to specify if Lomb
        method is used.
    vlf_band : tuple
        Very low frequency bands for features extraction from power spectral density.
    hf_band : tuple
        High frequency bands for features extraction from power spectral density.
    Returns
    ---------
    freq : list
        Frequency of the corresponding psd points.
    psd : list
        Power Spectral Density of the signal.
    """

    timestamp_list = _create_timestamp_list(nn_intervals)

    if method == WELCH_METHOD:
        # ---------- Interpolation of signal ---------- #
        funct = interpolate.interp1d(x=timestamp_list, y=nn_intervals, kind=interpolation_method)

        timestamps_interpolation = _create_interpolated_timestamp_list(nn_intervals, sampling_frequency)
        nni_interpolation = funct(timestamps_interpolation)

        # ---------- Remove DC Component ---------- #
        nni_normalized = nni_interpolation - np.mean(nni_interpolation)

        #  --------- Compute Power Spectral Density  --------- #
        freq, psd = signal.welch(x=nni_normalized, fs=sampling_frequency, window='hann',
                                 nfft=4096)

    elif method == LOMB_METHOD:
        freq, psd = LombScargle(timestamp_list, nn_intervals,
                                normalization='psd').autopower(minimum_frequency=vlf_band[0],
                                                               maximum_frequency=hf_band[1])
    else:
        raise ValueError("Not a valid method. Choose between 'lomb' and 'welch'")

    return freq, psd


def _create_timestamp_list(nn_intervals: List[float]) -> List[float]:
    """
    Creates corresponding time interval for all nn_intervals
    Parameters
    ---------
    nn_intervals : list
        List of Normal to Normal Interval.
    Returns
    ---------
    nni_tmstp : list
        list of time intervals between first NN-interval and final NN-interval.
    """
    # Convert in seconds
    nni_tmstp = np.cumsum(nn_intervals) / 1000

    # Force to start at 0
    return nni_tmstp - nni_tmstp[0]


def _create_interpolated_timestamp_list(nn_intervals: List[float], sampling_frequency: int = 7) -> List[float]:
    """
    Creates the interpolation time used for Fourier transform's method
    Parameters
    ---------
    nn_intervals : list
        List of Normal to Normal Interval.
    sampling_frequency : int
        Frequency at which the signal is sampled.
    Returns
    ---------
    nni_interpolation_tmstp : list
        Timestamp for interpolation.
    """
    time_nni = _create_timestamp_list(nn_intervals)
    # Create timestamp for interpolation
    nni_interpolation_tmstp = np.arange(0, time_nni[-1], 1 / float(sampling_frequency))
    return nni_interpolation_tmstp


def _get_features_from_psd(freq: List[float], psd: List[float], vlf_band: namedtuple = VlfBand(0.003, 0.04),
                           lf_band: namedtuple = LfBand(0.04, 0.15),
                           hf_band: namedtuple = HfBand(0.15, 0.40)) -> dict:
    """
    Computes frequency domain features from the power spectral decomposition.
    Parameters
    ---------
    freq : array
        Array of sample frequencies.
    psd : list
        Power spectral density or power spectrum.
    vlf_band : tuple
        Very low frequency bands for features extraction from power spectral density.
    lf_band : tuple
        Low frequency bands for features extraction from power spectral density.
    hf_band : tuple
        High frequency bands for features extraction from power spectral density.
    Returns
    ---------
    freqency_domain_features : dict
        Dictionary containing frequency domain features for HRV analyses. There are details
        about each features given below.
    """

    # Calcul of indices between desired frequency bands
    vlf_indexes = np.logical_and(freq >= vlf_band[0], freq < vlf_band[1])
    lf_indexes = np.logical_and(freq >= lf_band[0], freq < lf_band[1])
    hf_indexes = np.logical_and(freq >= hf_band[0], freq < hf_band[1])

    # Integrate using the composite trapezoidal rule
    lf = np.trapz(y=psd[lf_indexes], x=freq[lf_indexes])
    hf = np.trapz(y=psd[hf_indexes], x=freq[hf_indexes])

    # total power & vlf : Feature often used for  "long term recordings" analysis
    vlf = np.trapz(y=psd[vlf_indexes], x=freq[vlf_indexes])
    total_power = vlf + lf + hf

    lf_hf_ratio = lf / hf
    lfnu = (lf / (lf + hf)) * 100
    hfnu = (hf / (lf + hf)) * 100

    freqency_domain_features = {
        'lf': lf,
        'hf': hf,
        'lf_hf_ratio': lf_hf_ratio,
        'lfnu': lfnu,
        'hfnu': hfnu,
        'total_power': total_power,
        'vlf': vlf
    }

    return freqency_domain_features
