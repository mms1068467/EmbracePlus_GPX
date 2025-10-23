import pandas as pd
from avro.datafile import DataFileReader
from avro.io import DatumReader

import matplotlib.pyplot as plt
from datetime import datetime
import json

from preprocessing.preprocess_signals import filter_signals
from MOS_rules_EDA_individualized import MOS_main_filepath

import os

def find_avro_files(FOLDER_PATH: str):
    # find all sqlite files in SQLITE_PATH
    avro_files = []
    # check if specified path is already an .sqlite file
    if os.path.isfile(FOLDER_PATH):
        avro_files.append(FOLDER_PATH)

    else:
        for root, dirs, files in os.walk(FOLDER_PATH):
            # print("Root: \n", root)
            # print("Directory: \n", dirs)
            # print('Files: \n', files)
            for file in files:
                if file.endswith('.avro'):
                    avro_files.append(os.path.join(root, file))

    print(f"Found {len(avro_files)} files with extension '.avro'. \n")
    return avro_files



def find_filtered_data_files(FOLDER_PATH: str):
    # find all sqlite files in SQLITE_PATH
    filtered_data_files = []
    # check if specified path is already an .sqlite file
    if os.path.isfile(FOLDER_PATH):
        avro_files.append(FOLDER_PATH)

    else:
        for root, dirs, files in os.walk(FOLDER_PATH):
            # print("Root: \n", root)
            # print("Directory: \n", dirs)
            # print('Files: \n', files)
            for file in files:
                if "Filtered-Data-Part" in file:
                    filtered_data_files.append(os.path.join(root, file))

    print(f"Found {len(filtered_data_files)} filtered files with extension '.csv'. \n")
    return filtered_data_files


#avro_file_paths = find_avro_files(r"C:\Users\b1081018\Desktop\Sensors\Empatica-EmbracePlus\EmbracePlus_NK-3YK3K151PX")

## Read Avro File

def read_avro_file(filepath: str):

    reader = DataFileReader(open(filepath, "rb"), DatumReader())
    schema = json.loads(reader.meta.get('avro.schema').decode('utf-8'))

    data = []

    for datum in reader:
        data = datum
    reader.close()

    #print("Schema looks like this: \n")
    #print(schema)

    return data, schema


# Load data from S3 bucket by name and key
def get_avro_file_from_s3(bucket_name, key):
    # Todo: Download file from S3 bucket into a in-memory file object
    client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    response = client.get_object(Bucket=bucket_name, Key=key)

    return response

# Read avro file from S3 bucket into a pandas dataframe
def read_avro_file_io(response):
    # Write the response to an in-memory stream
    memory_stream = io.BytesIO(response['Body'].read())

    reader = DataFileReader(memory_stream, DatumReader())

    df_out = preprocess_avro_file(reader)
    
    return df_out

# preprocess avro file
def preprocess_avro_file(reader):
    for data in reader:
        
        tmp = pd.DataFrame(data["rawData"]["temperature"])
        tmp["timestampStart"] = pd.date_range(pd.to_datetime(tmp["timestampStart"][0]*1000, unit="ns"), periods=len(tmp), freq="1s")
        tmp.rename(columns={"timestampStart": "time_iso", "samplingFrequency":"samplingFrequencyST", "values": "ST"}, inplace=True)
        tmp = filter_signals(tmp, "ST")
        
        eda = pd.DataFrame(data["rawData"]["eda"])
        eda["timestampStart"] = pd.date_range(pd.to_datetime(eda["timestampStart"][0]*1000, unit="ns"), periods=len(eda), freq="250ms")
        eda.rename(columns={"timestampStart": "time_iso", "samplingFrequency":"samplingFrequencyEDA", "values": "GSR"}, inplace=True)
        eda = filter_signals(eda, "GSR")
        
        resampled_eda = eda.set_index("time_iso").resample("1s").mean().reset_index()
        resampled_tmp = tmp.set_index("time_iso").resample("1s").ffill().reset_index()
        df_out = pd.merge(resampled_eda, resampled_tmp, on="time_iso")
        
        df_out["TimeNum"] = df_out["time_iso"].astype("int64")/1e9

        return df_out



# data, schema = read_avro_file(r"C:\Users\b1081018\Desktop\Sensors\Empatica-EmbracePlus\EmbracePlus_NK-3YK3K151PX\raw_data\v6\1-1-NK_1675258533.avro")




def get_accelerometer_data(avro_file_data: pd.DataFrame):

    accelerometer_data = avro_file_data["rawData"]["accelerometer"]
    start_time = avro_file_data['rawData']['accelerometer']['timestampStart']

    return accelerometer_data, start_time

def get_gyroscope_data(avro_file_data: pd.DataFrame):

    gyroscope_data = avro_file_data["rawData"]["gyroscope"]
    start_time = avro_file_data['rawData']['gyroscope']['timestampStart']

    return gyroscope_data, start_time


def get_eda_data(avro_file_data: pd.DataFrame):

    eda_data = avro_file_data["rawData"]["eda"]
    start_time = avro_file_data['rawData']['eda']['timestampStart']
    samplingFrequency = avro_file_data['rawData']['eda']['samplingFrequency']
    values = avro_file_data['rawData']['eda']['values']

    return eda_data, start_time, samplingFrequency

def get_st_data(avro_file_data: pd.DataFrame):

    st_data = avro_file_data["rawData"]["temperature"]
    start_time = avro_file_data['rawData']['temperature']['timestampStart']
    samplingFrequency = avro_file_data['rawData']['temperature']['samplingFrequency']

    return st_data, start_time, samplingFrequency

def get_bvp_data(avro_file_data: pd.DataFrame):

    bvp_data = avro_file_data["rawData"]["bvp"]
    start_time = avro_file_data['rawData']['bvp']['timestampStart']
    samplingFrequency = avro_file_data['rawData']['bvp']['samplingFrequency']

    return bvp_data, start_time, samplingFrequency

def get_tags(avro_file_data: pd.DataFrame):

    tags = avro_file_data["rawData"]["tags"]

    return tags

def get_steps(avro_file_data: pd.DataFrame):

    steps = avro_file_data["rawData"]["steps"]

    return steps

def get_timezone(avro_file_data: pd.DataFrame):
    return avro_file_data['timezone']


def get_participant_data(avro_file_data: pd.DataFrame):
    participantID = avro_file_data['enrollment']["participantID"]
    siteID = avro_file_data['enrollment']["siteID"]
    studyID = avro_file_data['enrollment']["studyID"]
    organizationID = avro_file_data['enrollment']["organizationID"]

    # data['enrollment'] lists other attributes

def get_start_timestamp(avro_file_data: pd.DataFrame):
    tags = avro_file_data["rawData"]["timestampStart"]

#print(f" EDA data start time: {data['rawData']['eda']['timestampStart']} \n")
#print(f" EDA raw data: {data['rawData']['eda']['values']} \n")
#print(f" Skin Temperature data start time: {data['rawData']['temperature']['timestampStart']} \n")
#print(f"Skin Temperature raw data: {data['rawData']['temperature']['values']} \n")
#print(f"Inertial Measurement raw data: {data['rawData']['InertialMeasurement']} \n")
#print(data['rawData']['bvp'])


#print(schema)


def fill_missing_timestamp_gaps(data: pd.DataFrame, timestamp_col_name: str = 'time_iso', resampling_interval: str = '1s'):

    # fill time stamp gaps
    df =  data.copy()

    df_filled = (df.assign(date=df[timestamp_col_name].dt.date)   #create new col 'date' from the timestamp
                .set_index(timestamp_col_name)              #set timestamp as index
                .groupby('date')                     #groupby for each date
                .apply(lambda x: x.resample(resampling_interval)  #apply resampling for 1 minute from start time to end time for that date
                    .ffill())                     #ffill values
                .reset_index('date', drop=True)      #drop index 'date' that was created by groupby
                .drop('date', axis = 1)                      #drop 'date' column created before
                .reset_index()                       #reset index to get back original 2 cols
            )

    return df_filled


##### MOS Detection Algorithm Wrapper class

#### Wrapper for MOS model #### 
class BaseModel():
    def __init__(self, model) -> None:
        self.model = model 
        
    def predict(self, data):
        pass

class MOSModel(BaseModel):
    def __init__(self, model) -> None:
        super().__init__(model)

    def get_mos_output(self, data, MOSpercentage):
        mos_output, mos_count = MOS_main_filepath(data, MOSpercentage = MOSpercentage)
        return mos_output
    
    def predict(self, data, MOSpercentage = 1.25):
        test, _ = MOS_main_filepath(data, MOSpercentage = MOSpercentage)
        return test["detectedMOS"].values