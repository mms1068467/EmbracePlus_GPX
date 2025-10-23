interval_dic = {'GSR': '1000', 'ST': '1000', 'IBI': '1000', 'ECG': '1000', 'IBI_E4': '1000', 'PPG': '1000', 'HR': '1000', 'ACC': '1000', 'ACC_E4': '1000'}  # sampling interval
platform_id_dic = {'GSR': '3', 'ST': '3', 'IBI_E4': '3', 'PPG': '3', 'IBI': '2', 'ECG': '2', 'HR':'2', 'ACC': '2', 'ACC_E4': '3'} # signal lookup table for platform_id
sensor_id_dic = {'GSR': '7', 'ST': '3', 'IBI_E4': '22', 'PPG': '21', 'IBI': '16', 'ECG': '15', 'HR':'1', 'ACC': '17', 'ACC_E4': '17'}  # signal lookup table for sensor_id

freq_dic = {'GSR': 4, 'ST': 4, 'IBI': 1, 'IBI_E4': 1, 'PPG': 64, 'ECG': 250, 'HR': 1, 'ACC_E4': 16}  # frequency lookup table (in Hz)
cluster_size_dic = {'GSR': 6, 'ST': 8, 'IBI': 1}  # values per cluster

# dictionaries of known filter parameter
butter_filter_order_dic = {'GSR': 1, 'ST': 2}  # low-, and high-pass butterworth filters - have actually the same order

# defining low- and high-pass cutoff frequencies for filtering GSR and ST measurements (based on literature)
# w = cutoff Freq. / (Sampling Freq / 2) -> (Sampling Freq / 2) = Niquist Theorem
lowpass_cutoff_freq_dic = {'GSR': 1 / (4 / 2), 'ST': 0.1/ (4 / 2)}  # GSR 0.5 as second option
highpass_cutoff_freq_dic = {'GSR': 0.05 / (4 / 2), 'ST': 0.01/ (4 / 2)}