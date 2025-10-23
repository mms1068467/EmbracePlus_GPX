"""
Time-Domain Features calculation from the IBI data
SDNN, RMSDD und PNN50 Formel can be found at:
https://knowledge.time2tri.me/en/articles/main-hrv-parameters-part-1-time-related-parameters
"""

import pandas as pd
import numpy as np
import math
import plotly.express as px


# -------------------------------SDNN--------------------------------------
# Calculation 1: SDNN (standard derivation of the IBI signal)
# Mechanism: Sympathetic & Parasympathetic
def sdnn_hrv(data, timeframe=10):
    sdnn_data = data.copy()
    sdnn_data['SDNN'] = [np.nan] * len(data)
    avg = data['IBI'].mean()

    sum = 0
    counter = 0
    i = 0
    # for i in range(0,len(data['IBI'])):
    while i < len(data['IBI']) - 1:
        sum += (data.loc[i, 'IBI'] - avg) ** 2

        counter += 1
        # timeframe-1 in if-condition because index starts at 0
        if counter == timeframe:
            # print(i % (timeframe-1) == 0, f'{i} % {timeframe}-1')
            sdnn = math.sqrt((1 / (timeframe - 1)) * sum)
            print("sdnn", sdnn)
            sdnn_data.loc[i - (timeframe / 2), 'SDNN'] = sdnn

            i = i - (timeframe - 1)
            print("new i", i, "old i", i + (timeframe - 2))

            counter = 0
            sum = 0
        i += 1

    # convert time_iso into time_millis for interpolation
    sdnn_data['time_iso'] = sdnn_data["time_iso"].astype(np.int64) / int(1e9)

    # Interpolation with time_millis which has already the right timestamp
    data = sdnn_data.set_index('time_iso')
    data = data.interpolate(method="linear", order=1)
    sdnn_data = data.reset_index()

    sdnn_data.time_iso = pd.to_datetime(sdnn_data.time_iso, unit='s')

    return sdnn_data


# -------------------------------RMSSD--------------------------------------
# Calculation 2: RMSSD (Root Mean Sum of Squared Distance)
# Mechanism: Parasympathetic
def rmssd_hrv(data, timeframe=10):
    rmssd_data = data.copy()
    rmssd_data['RMSSD'] = [np.nan] * len(data)

    sum = 0
    counter = 0
    i = 0
    # for i in range(0,len(data['IBI'])-1):
    while i < len(data['IBI']) - 2:
        sum += (data.loc[i + 1, 'IBI'] - data.loc[i, 'IBI']) ** 2
        counter += 1
        # timeframe-1 in if-condition because index starts at 0
        if counter == timeframe:
            print(i % (timeframe - 1) == 0, f'{i} % {timeframe}-1')
            rmssd = math.sqrt((1 / (timeframe - 1)) * sum)
            print("rmssd", rmssd)
            rmssd_data.loc[i - (timeframe / 2), 'RMSSD'] = rmssd

            i = i - (timeframe - 1)
            print("new i", i, "old i", i + (timeframe - 2))

            counter = 0
            sum = 0
        i += 1

    # convert time_iso into time_millis for interpolation
    rmssd_data['time_iso'] = rmssd_data["time_iso"].astype(np.int64) / int(1e9)

    # Interpolation with time_millis which has already the right timestamp
    data = rmssd_data.set_index('time_iso')
    data = data.interpolate(method="linear", order=1)
    rmssd_data = data.reset_index()

    rmssd_data.time_iso = pd.to_datetime(rmssd_data.time_iso, unit='s')

    return rmssd_data


# -------------------------------ln of RMSSD--------------------------------------
# Calculation 2: logarithm natural RMSSD (Root Mean Sum of Squared Distance)
def ln_rmssd_hrv(data, timeframe=10):
    ln_rmssd_data = data.copy()
    ln_rmssd_data['ln_RMSSD'] = np.log(rmssd_hrv(data, timeframe))

    return ln_rmssd_data


# -------------------------------PNN50--------------------------------------
# Calculation 3: pNN50
# pNN50 is the fraction of all pairs of consecutive RR intervals that differ from each other by more than 50 ms.
# Mechanism: Parasympathetic
def pNN50_hrv(data, timeframe=10):
    pNN50_data = data.copy()
    pNN50_data['pNN50'] = [np.nan] * len(data)

    sum = 0
    counter = 0
    i = 0
    while i < len(data['HRV']) - 1:
        sum += 1 if data.loc[i, 'HRV'] > 50 else 0
        counter += 1
        # timeframe-1 in if-condition because index starts at 0
        if counter == timeframe - 1:
            # print(i % (timeframe-1) == 0, f'{i} % {timeframe}-1')
            pNN50 = (sum / (timeframe - 1)) * 100
            print("pNN50", pNN50)
            # print(i,": data stored:", i-timeframe+1)
            pNN50_data.loc[i - timeframe + 1, 'pNN50'] = pNN50

            i = i - (timeframe - 2)
            print("new i", i, "old i", i + (timeframe - 2))

            counter = 0
            sum = 0
        i += 1

    # convert time_iso into time_millis for interpolation
    pNN50_data['time_iso'] = pNN50_data["time_iso"].astype(np.int64) / int(1e9)

    # Interpolation with time_millis which has already the right timestamp
    data = pNN50_data.set_index('time_iso')
    data = data.interpolate(method="linear", order=1)
    pNN50_data = data.reset_index()

    pNN50_data.time_iso = pd.to_datetime(pNN50_data.time_iso, unit='s')

    return pNN50_data


# -------------------------------SDNNI--------------------------------------
# Calculation 4: SDNN INDEX (SDNNI)
# Mean of every SDNN frame
def sdnni_hrv(data, timeframe=20):
    sdnni_data = data.copy()
    sdnni_data['SDNNI'] = [np.nan] * len(data)

    for i in range(0, len(data['SDNN']), timeframe):
        sdnni_data.loc[i + (timeframe / 2), 'SDNNI'] = data[i:i + timeframe - 1]['SDNN'].mean()

    # convert time_iso into time_millis for interpolation
    sdnni_data['time_iso'] = sdnni_data["time_iso"].astype(np.int64) / int(1e9)

    # Interpolation with time_millis which has already the right timestamp
    data = sdnni_data.set_index('time_iso')
    data = data.interpolate(method="linear", order=1)
    sdnni_data = data.reset_index()

    sdnni_data.time_iso = pd.to_datetime(sdnni_data.time_iso, unit='s')

    return sdnni_data


def plotting_time_domain_features(data: pd.DataFrame, title: str = "Interactive Data incl. Stress Moments",
                                  saveHTML: bool = False, htmlName: str = "time_domain_features_incl_stress.html"):
    plot_data_form = pd.melt(data, id_vars=['time_iso'],
                             value_vars=['GSR', 'ST', 'IBI', 'HRV', 'RMSSD', 'SDNN', 'pNN50', 'SDNNI', 'ln_RMSSD',
                                         'stress'],
                             value_name='signal [ms]')

    # select signals that we want to display
    Signal_subset = plot_data_form.query(
        "variable == 'IBI' or variable == 'SDNN' or variable == 'pNN50' or variable == 'RMSSD' "
        "or variable == 'SDNNI' or variable == 'ln_RMSSD'")

    # create interactive plot where color where signals are presented in different colors
    fig = px.line(data_frame=Signal_subset, x='time_iso', y='signal [ms]', color='variable',
                  title=title)

    # adding stress moments
    for i in data.index:
        if data.stress[i] == 1:
            fig.add_vline(data['time_iso'][i], line_color='black')

    fig.show()
    if saveHTML:
        fig.write_html(htmlName)
