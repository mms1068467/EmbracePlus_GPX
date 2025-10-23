"""
Main preprocessing script
"""
import datetime

import numpy as np
import pandas as pd

from preprocessing import utilities
from preprocessing import data_loader as dl
from preprocessing import preprocess_signals as pps


# TODO - empty location dataframe in sqlite file: Lab 3, Session 1, Phone 5 --> check if phone was recording anything for this

# eDiary Data containing physiological data measurements

#full_path_prefix = ""
full_path_prefix = "C:/Users/MM/Desktop/Uni Salzburg/P.hD/ZGis/Human Sensing/MOS_Detection/MOS_Detection_new/"

lab_mo = "Moritz-test/zgis_phone_6_2022-05-06T1124.sqlite"

# Session 1
lab_1_1 = "LabData1-3/lab1/raw_data/20180627_084235_phone_1.sqlite3"

# Session 2
lab_2_3_1 = "LabData1-3/lab2/raw_data/Session_3/phone_1/eDiary/sqlite/20190213_133621.sqlite3"
lab_2_5_1 = "LabData1-3/lab2/raw_data/Session_5/phone_1/eDiary/sqlite/20190213_133621.sqlite3"

# Session 3
lab_3_3_1 = "LabData1-3/lab3/raw_data/Session3/2021-04-21T0941_1/zgis_phone_1_2021-04-21T0941.sqlite"

# Session 4
lab_4_2_1 = "Labtest4/Session2/2022-02-16T1112_1/zgis_phone_1_2022-02-16T1112.sqlite"


# TU Test:
#physio1 = "Tests/Physio_1/tuwien_106_2022-03-11T1246.sqlite"
#physio3 = "Tests/Physio_3/tuwien_108_2022-03-12T0853.sqlite"
#physio4 = "Tests/Physio_4/tuwien_109_2022-03-12T1011.sqlite"

# create full path to file
filename = full_path_prefix + lab_3_3_1

# Ground Truth Data --> labeled stress moments (0/1)
#labels_file = "LabData1-3/lab3/usable/lab3_3_1.csv"
#labels = full_path_prefix + labels_file

# columns: TimeNum, GSR, ST, time, stress --> only time column (ISO format) is relevant
#gt = pd.read_csv(labels)

#print("Ground truth: ", gt)

# load in physiological Data from eDiary App sqlite file

#### GSR
GSR_cluster, GSR_raw = dl.get_ediary_data(filename = filename, phys_signal = "GSR")
print(GSR_raw)
#GSR_raw_prep = pps.preprocess_GSR_ST(clustered_data = GSR_cluster, raw_data = GSR_raw, phys_signal="GSR")
#print("GSR raw prepprocess \n", GSR_raw_prep)


# all in one
GSR = pps.GSR_preprocessing(GSR_cluster = GSR_cluster,
                            GSR_raw = GSR_raw,
                            phys_signal = "GSR")


#### ST
ST_cluster, ST_raw = dl.get_ediary_data(filename = filename, phys_signal = "ST")
print(ST_raw)

ST = pps.ST_preprocessing(ST_cluster = ST_cluster,
                          ST_raw = ST_raw,
                          phys_signal = "ST")


#### IBI
IBI_raw = dl.get_ediary_data(filename = filename, phys_signal = "IBI")
IBI_raw_formatted = "" #format IBI raw
IBI = pps.IBI_preprocessing(IBI_raw)


#### HRV ---> get HRV from preprocessed IBIs
HRV = pps.HRV_preprocessing(IBI)

print("GSR: ", GSR)
print("ST: ", ST)
print("IBI: ", IBI)
print("HRV: ", HRV)

merged_data = pps.merge_signals(GSR, ST, IBI, HRV, merge_col = 'time_iso')
print("Final preprocessed and merged dataset: \n", merged_data.head(30))


#### ECG
#ECG = dl.get_ediary_data(filename = filename, phys_signal = "ECG")

# TODO - divide IBI by 1000 and then visualize GSR, ST & IBI on one plot

#### Human Sensing - Visualization

from HS_Visualization import plot_signals as ps

long_data = ps.convert_signals_to_long_format(merged_data, id_variables = ['time_iso'],
                                              id_var_name = 'variable',
                                              value_variables = ['GSR', 'ST', 'IBI', 'HRV'],
                                              value_name = 'value')

print("Data in long format \n", long_data)







# plotting function with detected MOS markers:

import plotly.express as px
import plotly.graph_objects as go

#plot_data_form = pd.melt(lab2_5_1, id_vars=['time_iso'],
#                         value_vars=['GSR', 'ST', 'IBI', 'HRV', 'stress', 'detectedMOS'],
#                         value_name='signal')

# select signals that we want to display
# TODO - add
Signal_subset = long_data.query("variable == 'ST'")

# create interactive plot where color where signals are presented in different colors
fig = px.line(data_frame=Signal_subset, x='time_iso', y='value', color='variable',
              title="Interactive Data incl. Stress Moments and detected MOS")
fig.add_trace(go.Scatter(x = GSR_raw['time_iso'], y = GSR_raw['value_real']))
fig.show()

Signal_subset = long_data.query("variable == 'GSR'")
fig2 = px.line(data_frame=Signal_subset, x='time_iso', y='value', color='variable',
              title="Interactive Data incl. Stress Moments and detected MOS")
fig2.add_trace(go.Scatter(x = ST_raw['time_iso'], y = ST_raw['value_real']))
fig2.show()


Signal_subset = long_data.query("variable == 'GSR' or variable == 'ST'")
fig3 = px.line(data_frame=Signal_subset, x='time_iso', y='value', color='variable',
              title="Interactive Data incl. Stress Moments and detected MOS")
fig3.show()





# adding stress moments
#for i in lab2_5_1.index:
#    if lab2_5_1.stress[i] == 1:
#        fig.add_vline(lab2_5_1['time_iso'][i], line_color='black')

# adding detected stress moments
#for i in gt.index:
#    if gt.stress[i] == 1:
#        # fig.add_vline(lab2_5_1['time_iso'][i], line_color = 'red')
#        fig.add_trace(go.Scatter(x=[filename['time_iso'][i]],
#                                 y=[filename['HRV'][i]],
#                                 mode='markers',
#                                 marker=dict(color='red', size=10),
#                                 showlegend=False))

