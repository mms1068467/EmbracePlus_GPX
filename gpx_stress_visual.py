import streamlit as st
from streamlit_folium import st_folium
import folium
from folium import plugins

import sqlite3
import pathlib
import os
import pandas as pd



import math
import datetime
from datetime import datetime, timezone
from io import BytesIO
import random
import time

import plotly.express as px
import plotly.graph_objects as go

import gpxpy
import geopandas as gpd
from shapely.geometry import Point


from streamlit_folium import folium_static
from folium.plugins import MarkerCluster

def create_map():
    if 'map' not in st.session_state or st.session_state.map is None:
        stress_map = create_folium_map_with_track_and_stress(gdf_merged.dropna(subset=['geometry']), mos_threshold=0.75)
        
        st.session_state.map = stress_map  # Save the map in the session state
    return st.session_state.map

def show_map():
    stress_map = create_map()  # Get or create the map
    folium_static(stress_map)



def save_uploadedfile(uploaded_file, path: str):
    with open(os.path.join(path, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return st.success("Saved file: {} to {}".format(uploaded_file.name, path))


def read_gpx(file_path):
    with open(file_path, 'r') as gpx_file:
        gpx_data = gpxpy.parse(gpx_file)
    return gpx_data

def print_gpx_contents(gpx_data):
    print("GPX Version:", gpx_data.version)
    print("GPX Creator:", gpx_data.creator)
    print("\nTracks:")
    for track in gpx_data.tracks:
        print(f" Track name: {track.name}")
        for segment in track.segments:
            print(f"  Segment with {len(segment.points)} points")
            for point in segment.points:
                print(f"   Point at ({point.latitude}, {point.longitude}), elevation={point.elevation}, time={point.time}")

    print("\nWaypoints:")
    for waypoint in gpx_data.waypoints:
        print(f" Waypoint {waypoint.name} at ({waypoint.latitude}, {waypoint.longitude})")

    print("\nRoutes:")
    for route in gpx_data.routes:
        print(f" Route name: {route.name}")
        for point in route.points:
            print(f"  Route point at ({point.latitude}, {point.longitude})")

def create_gdf(gpx_data):
    points = []
    times = []

    for track in gpx_data.tracks:
        for segment in track.segments:
            for point in segment.points:
                points.append(Point(point.longitude, point.latitude))
                times.append(point.time)

    df = pd.DataFrame({'timestamp': times})
    gdf = gpd.GeoDataFrame(df, geometry=points)
    gdf.set_crs(epsg=4326, inplace=True)  # WGS84 lat/lon
    return gdf

def print_metadata(gpx_data):
    # Calculate sampling rate roughly by averaging time delta between points
    all_times = []
    for track in gpx_data.tracks:
        for segment in track.segments:
            for point in segment.points:
                if point.time:
                    all_times.append(point.time)
    all_times.sort()

    if len(all_times) > 1:
        deltas = [(all_times[i+1] - all_times[i]).total_seconds() for i in range(len(all_times)-1)]
        avg_sampling_rate = sum(deltas) / len(deltas)
        print(f"\nEstimated average sampling interval: {avg_sampling_rate:.2f} seconds")
    else:
        print("\nNot enough timestamp data to calculate sampling rate")

    # Additional metadata from GPX extensions or other attributes can be extracted here,
    # but that depends on the GPX data and device. We'll print the raw extensions if available:
    print("\nMetadata and Extensions:")
    if gpx_data.extensions:
        print(gpx_data.extensions)
    else:
        print("No extensions metadata found")

def extract_all_metadata(gpx_data):
    """
    Extracts all metadata recursively from GPX object including extensions,
    returning a dict with keys and values.
    """
    def recursive_extract(obj):
        metadata = {}
        # Extract all attributes except private and callable
        for attr in dir(obj):
            if attr.startswith('_') or callable(getattr(obj, attr)):
                continue
            try:
                val = getattr(obj, attr)
                # If val is a GPX element with extensions, recurse
                if hasattr(val, '__class__') and 'gpxpy' in str(val.__class__):
                    metadata[attr] = recursive_extract(val)
                else:
                    metadata[attr] = val
            except Exception:
                continue
        return metadata

    meta = recursive_extract(gpx_data)
    return meta


def create_folium_map_with_track(gdf):
    if gdf.empty:
        print("GeoDataFrame is empty, cannot create map.")
        return None

    # Center map on mean location
    mean_lat = gdf.geometry.y.mean()
    mean_lon = gdf.geometry.x.mean()

    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=13)

    # Extract lat/lon tuples in order
    lat_lon = [(point.y, point.x) for point in gdf.geometry]

    # Add a PolyLine for the track
    folium.PolyLine(lat_lon, color="blue", weight=5, opacity=0.7).add_to(m)

    # Optional: add popup markers at each point with timestamp info
    for idx, row in gdf.iterrows():
        timestamp = row['timestamp']
        popup_text = f"Time: {timestamp}" if timestamp else "No timestamp"
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=3,
            color='red',
            fill=True,
            fill_color='red',
            popup=popup_text
        ).add_to(m)

    return m

def create_folium_map_with_track_and_stress(gdf, mos_threshold=1.5):
    #if gdf.empty:
    #    print("GeoDataFrame is empty, cannot create map.")
    #    return None

    # Ensure geometry is all Points
    #if not all(gdf.geometry.geom_type == "Point"):
    #    print("Geometry must be of type Point.")
    #    return None

    # Use .x and .y on the geometry series
    mean_lat = gdf.geometry.y.mean()
    mean_lon = gdf.geometry.x.mean()

    m = folium.Map(location=[mean_lat, mean_lon], zoom_start=13)

    # Create list of (lat, lon) tuples
    lat_lon = list(zip(gdf.geometry.y, gdf.geometry.x))

    # Add track line
    folium.PolyLine(lat_lon, color="blue", weight=5, opacity=0.7).add_to(m)

    gdf_mos = gdf[gdf["MOS_Score"] > mos_threshold]
    gdf_no_mos = gdf[gdf["MOS_Score"] <= mos_threshold]

    gdf_mos = gdf_merged[gdf_merged["MOS_Score"] > 1.5].dropna(subset=['geometry'])


    # Add conditionally colored markers
    """
    for idx, row in gdf_no_mos.iterrows():
        point = row.geometry
        timestamp = row.get('timestamp')
        mos_score = row.get('MOS_Score')

        popup_text = f"Time: {timestamp}<br>MOS_Score: {mos_score}"


        folium.CircleMarker(
            location=[point.y, point.x],
            radius=4,
            color="green",
            fill=True,
            fill_color="green",
            fill_opacity=0.8,
            popup=popup_text
        ).add_to(m)

    """
    
    for idx, row in gdf_mos.iterrows():
        point = row.geometry
        timestamp = row.get('timestamp')
        mos_score = row.get('MOS_Score')

        popup_text = f"Time: {timestamp}<br>MOS_Score: {mos_score}"


        folium.CircleMarker(
            location=[point.y, point.x],
            radius=7,
            color='red',
            fill=True,
            fill_color="red",
            fill_opacity=0.8,
            popup=popup_text
        ).add_to(m)

    return m


import numpy as np

def merge_on_closest_timestamp(df1, df2, df1_time_col: str = "time_iso", df2_time_col: str = "timestamp"):

    df1 = df1.rename(columns={"time_iso": "timestamp"})
    # Explicitly set both timezones to pandas UTC (standardized)
    df1['timestamp'] = df1['timestamp'].dt.tz_convert('UTC')
    df2['timestamp'] = df2['timestamp'].dt.tz_convert('UTC')

    # Sort both DataFrames by the timestamp
    df1 = df1.sort_values('timestamp')
    df2 = df2.sort_values('timestamp')

    # Perform asof merge in both directions and find the closest match
    merged_forward = pd.merge_asof(df1, df2, on='timestamp', direction='forward')
    merged_backward = pd.merge_asof(df1, df2, on='timestamp', direction='backward')

    # Compute time differences
    merged_forward['forward_diff'] = (merged_forward['timestamp'] - df1['timestamp']).abs()
    merged_backward['backward_diff'] = (merged_backward['timestamp'] - df1['timestamp']).abs()

    return merged_backward



st.header("Select a .gpx track file (.gpx extension):")

path = pathlib.Path(__file__).parent.resolve()
st.markdown("---")

######## File uploader ########

uploaded_db_file = st.file_uploader("Drag and drop your .gpx file here...", type=["gpx"])
st.info("Upload .gpx file")

# the main, branching part of the application
if uploaded_db_file is not None:
    try:
        save_uploadedfile(uploaded_file=uploaded_db_file, path=path)
        st.write(f"Saving file {uploaded_db_file} to: {path}")
        
        st.sidebar.title("Visualise Track and stress")

        gpx = read_gpx(os.path.join(path, uploaded_db_file.name))
        gdf = create_gdf(gpx)

        save = False

        # Create and save folium map with track line and points
        folium_map = create_folium_map_with_track(gdf)

        

        add_stress = st.sidebar.checkbox("Please tick this box if you want to add stress events to the map:")

        if add_stress:

            uploaded_stress_file = st.file_uploader("Drag and drop your .gpx file here...", type=["csv"])
            st.info("Now upload a .csv stress file")

            # the main, branching part of the application
            if uploaded_stress_file is not None:
                try:
                    save_uploadedfile(uploaded_file=uploaded_stress_file, path=path)
                    st.write(f"Saving file {uploaded_stress_file} to: {path}")
                

                    stress_df = pd.read_csv(os.path.join(path, uploaded_stress_file.name))
                    stress_df["time_iso"] = pd.to_datetime(stress_df["time_iso"])

                    time_span = gdf.timestamp.max() - gdf.timestamp.min()

                    df_merged = merge_on_closest_timestamp(df1 = stress_df, df2=gdf)
                    df_merged_nona = df_merged.dropna()
                    gdf_merged = gpd.GeoDataFrame(df_merged, geometry=df_merged.geometry, crs = "EPSG:4326")

                    #stress_map = create_folium_map_with_track_and_stress(gdf_merged.dropna(subset=['geometry']), mos_threshold=0.75)
                    #st_map2 = st_folium(stress_map, width=1000)

                    show_map()


                except:
                    pass
            
            else:
                st_map = st_folium(folium_map, width=1000)
    except:
        pass