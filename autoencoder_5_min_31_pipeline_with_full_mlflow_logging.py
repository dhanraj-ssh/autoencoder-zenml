# - Requires a ZenML stack with an MLflow experiment tracker registered as 'mlflow_tracker'.
# - Key additions are marked with "### MLflow" comments.

# ==========================
# Imports
# ==========================
import os
import io
import tempfile
import pickle
import datetime as dt
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from zenml import pipeline, step#, Model
from zenml.client import Client
from pydantic import BaseModel

import mlflow
import mlflow.sklearn
import mlflow.tensorflow

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    auc,
    precision_recall_curve,
)

import tensorflow as tf
import keras

import requests
import json
from io import StringIO
import hashlib
from datetime import datetime, timedelta, timezone
import time

# ==========================
# Config
# ==========================
class TrainingConfig(BaseModel):
    """Configuration for the training pipeline."""
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 100
    max_depth: int = 10
    experiment_name: str = "autoencoder_5min_31_exp"  # <- used with MLflow
    model_name: str = "autoencoder_5min_31"

# If you want to enforce an experiment name globally (works well with local MLflow):
mlflow.set_experiment(TrainingConfig().experiment_name)  # ### MLflow

variables_used = {
    "ME_CSW_IN_T": "ME COPT COND CSW IN TEMP",
    "ME_CYL_OIL_IN_T": "ME CYL. L.O IN TEMP.",
    "ME_EXH_GAS_1_OUT_T": "ME EXH. GAS OUT TEMP.CYL. NO.1",
    "ME_EXH_GAS_2_OUT_T": "ME EXH. GAS OUT TEMP.CYL. NO.2",
    "ME_EXH_GAS_3_OUT_T": "ME EXH. GAS OUT TEMP.CYL. NO.3",
    "ME_EXH_GAS_4_OUT_T": "ME EXH. GAS OUT TEMP.CYL. NO.4",
    "ME_EXH_GAS_5_OUT_T": "ME EXH. GAS OUT TEMP.CYL. NO.5",
    "ME_EXH_GAS_6_OUT_T": "ME EXH. GAS OUT TEMP.CYL. NO.6",
    "ME_EXH_GAS_TC_IN_T": "ME T/C 1 EXH. GAS IN TEMP.",
    "ME_EXH_GAS_TC_OUT_T": "ME T/C 1  EXH. GAS OUT TEMP.",
    "ME_FO_IN_P": "ME F.O IN PRESS",
    "ME_FO_IN_T": "ME F.O. IN TEMP.",
    "ME_JACKET_CFW_1_OUT_T": "ME J.C.W OUT HIGH TEMP SLD.CYL.1",
    "ME_JACKET_CFW_2_OUT_T": "ME J.C.W OUT HIGH TEMP SLD.CYL.2",
    "ME_JACKET_CFW_3_OUT_T": "ME J.C.W OUT HIGH TEMP SLD.CYL.3",
    "ME_JACKET_CFW_4_OUT_T": "ME J.C.W OUT HIGH TEMP SLD.CYL.4",
    "ME_JACKET_CFW_5_OUT_T": "ME J.C.W OUT HIGH TEMP SLD.CYL.5",
    "ME_JACKET_CFW_6_OUT_T": "ME J.C.W OUT HIGH TEMP SLD.CYL.6",
    "ME_JACKET_CFW_IN_P": "ME J.C.W IN PRESS",
    "ME_MAIN_LO_IN_P": "ME L.O IN PRESS",
    "ME_MAIN_LO_IN_T": "ME L.O IN TEMP.",
    "ME_SCAV_AIR_RECEIV_P": "SCAV. AIR PRESS IN AIR RECEIVER",
    "ME_SCAV_AIR_RECEIV_T": "ME SCAV. AIR TEMP.  IN SCAV.",
    "ME_SCAV_AIR_1_BOX_T_SD": "ME SCAV. AIR FIRE DET. TEMP. HIGH PISTON CYL. NO.1 SLD",
    "ME_SCAV_AIR_2_BOX_T_SD": "ME SCAV. AIR FIRE DET. TEMP. HIGH PISTON CYL. NO.2 SLD",
    "ME_SCAV_AIR_3_BOX_T_SD": "ME SCAV. AIR FIRE DET. TEMP. HIGH PISTON CYL. NO.3 SLD",
    "ME_SCAV_AIR_4_BOX_T_SD": "ME SCAV. AIR FIRE DET. TEMP. HIGH PISTON CYL. NO.4 SLD",
    "ME_SCAV_AIR_5_BOX_T_SD": "ME SCAV. AIR FIRE DET. TEMP. HIGH PISTON CYL. NO.5 SLD",
    "ME_SCAV_AIR_6_BOX_T_SD": "ME SCAV. AIR FIRE DET. TEMP. HIGH PISTON CYL. NO.6 SLD",
    "Shaft_Power": "ME SHAFT POWER",
    "Shaft_RPM": "ME RPM",
    "dataTime": "dataTime",
}

# ==========================
# Steps
# ==========================
@step
def download_data(client_name: str, vessel_name: str, client_id: int) -> pd.DataFrame:
    def hash_string(data):
        sha256_hash = hashlib.sha256()
        sha256_hash.update(data.encode())
        return sha256_hash.hexdigest()
    
    def trigger_api(client_name, vessel_name, sql_query, API):
        upload_to_gcp = "false"
        
        string = f"client_name_{client_name}_vessel_name_{vessel_name}_upload_to_gcp_{upload_to_gcp}_query_{sql_query}_salt_mnzxvy&h$B)(KUI+7b5b670%6klkjbB=lkasjdf"
    
        token = hash_string(string)
        # print(token)
    
        upload_to_gcp = False
    
        headers = {
            "Content-Type": "application/json",
            "from-api": "true",
            "X-tenant-id": client_name,
            "x-request-id": "asdkfjalsdkf"
            
        }
        
    
        body = {
        "vessel_name": vessel_name,
        "client_name": client_name,
        "token": token,
        "query": sql_query,
        "upload_to_gcp": upload_to_gcp
        }
    
        try:
            response = requests.post(API, headers=headers, data=json.dumps(body))  # Make a GET request with parameters
            if response.status_code == 200:
                print(f"API response")
                # Use StringIO to create a file-like object from the string data
                csv_data = StringIO(response.text)
                
                # Read the CSV data into a DataFrame
                df = pd.read_csv(csv_data, low_memory=False)
                return df  # Return the response for further use if needed
            else:
                print(f"Failed to retrieve data from API. Status code: {response.text}")
                return None
        except requests.RequestException as e:
            print(f"Request failed: {e}")
            return None
    
    
    #client_name = "aesm"
    #vessel_name = "LOWLANDS ORANGE"
    upload_to_gcp = "false"
    #client_id = 8
    mlflow.log_param("client_name", client_name)
    mlflow.log_param("vessel_name", vessel_name)
    mlflow.log_param("client_id", client_id)
    
    url = "https://www.smartshipweb.com/prod/api/v2/extract" # modified url provided by Amol(Developer)
    
    
    start_date = datetime(2024, 6, 1)
    end_date = datetime(2025, 7, 25)
    step = timedelta(days=15)
    
    data_list = []
    final_df = pd.DataFrame()
    # Loop over time chunks
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + step - timedelta(seconds=1), end_date)
    
        start_str = current_start.strftime("%Y-%m-%d %H:%M:%S")
        end_str = current_end.strftime("%Y-%m-%d %H:%M:%S")
    
        print(f"Fetching data from {start_str} to {end_str}...")
    
        # Construct query inside the loop
        query = f'''SELECT
            name AS "vesselName",
            mappingname AS "vesselDASId",
            to_char(packettime, 'yyyy/mm/DD HH24:MI:SS') AS "dataTime",
            (packetdata ->> 'AF26') AS "ME SUPPLY LINE TEMP",
            (packetdata ->> 'AF35') AS "BOILER FUEL SUPPLY RATE",
            (packetdata ->> 'GPGLL_1') AS "GPS Latitude",
            (packetdata ->> 'GPGLL_2') AS "GPS Lat Direction N/S",
            (packetdata ->> 'GPGLL_3') AS "GPS Longitude",
            (packetdata ->> 'GPGLL_4') AS "GPS Long Direction E/W",
            (packetdata ->> 'GPGLL_5') AS "GPS Fix taken",
            (packetdata ->> 'GPGLL_6') AS "GPS Data Active",
            (packetdata ->> 'GPVTG_5') AS "GPS VTG SOG",
            (packetdata ->> 'GPRMB_4') AS "GPS From Waypoint ID",
            (packetdata ->> 'GPRMB_5') AS "GPS To Waypoint ID",
            (packetdata ->> 'GPRMB_10') AS "GPS Range to Destination",
            (packetdata ->> 'GPRMB_11') AS "GPS BRG to destination",
            (packetdata ->> 'GPRMB_13') AS "GPS Arrival Circle Entered",
            (packetdata ->> 'GPABP_13') AS "GPS Autopilot Heading",
            (packetdata ->> 'HEHDT_1') AS "Gyro Heading Degress",
            (packetdata ->> 'HEHDT_2') AS "Gyro Heading True Relative",
            (packetdata ->> 'TIROT_1') AS "Rate of Turn from Turn Sensor",
            (packetdata ->> 'TIROT_2') AS "ROT Status from Turn Sensor",
            (packetdata ->> 'WIMWV_1') AS "Anemo Wind Angle",
            (packetdata ->> 'WIMWV_2') AS "Anemo Wind Reference",
            (packetdata ->> 'WIMWV_3') AS "Anemo Wind Speed",
            (packetdata ->> 'WIMWV_4') AS "Anemo Wind Speed Unit",
            (packetdata ->> 'VDVHW_5') AS "Speed of vessel relative to the water",
            (packetdata ->> 'VDVLW_1') AS "Total Distance Traveled through Water",
            (packetdata ->> 'VDVLW_3') AS "Total Distance Traveled since Reset",
            (packetdata ->> 'GPVTG_1') AS "GPS Course",
            (packetdata ->> 'AIVDO_Latitude') AS "AIS Latitude",
            (packetdata ->> 'AIVDO_Latitude_Direction') AS "AIS Latitude Direction",
            (packetdata ->> 'AIVDO_Longitude') AS "AIS Longitude",
            (packetdata ->> 'AIVDO_Longitude_Direction') AS "AIS Longitude Direction",
            (packetdata ->> 'AIVDO_Course') AS "AIS Course",
            (packetdata ->> 'AIVDO_Speed') AS "AIS Speed Over Ground",
            (packetdata ->> 'stormGlassCurrentSpeed') AS "Current Speed",
            (packetdata ->> 'stormGlassWaveDirection') AS "Storm Glass Wave Direction",
            (packetdata ->> 'stormGlassWaveHeight') AS "Storm Glass Wave Height",
            (packetdata ->> 'stormGlassSwellHeight') AS "Storm Glass Swell Height",
            (packetdata ->> 'stormGlassWindSpeed') AS "Wind Speed",
            (packetdata ->> 'stormGlassWindDirection') AS "Storm Glass Wind Direction",
            (packetdata ->> 'stormGlassSwellPeriod') AS "Storm Glass Swell Period",
            (packetdata ->> 'stormGlassWindWaveHeight') AS "Storm Glass Wind Wave Height",
            (packetdata ->> 'stormGlassWindWavePeriod') AS "Storm Glass Wind Wave Period",
            (packetdata ->> 'SDDPT_1') AS "Depth  Below Draft",
            (packetdata ->> 'AM01') AS "ME F.O IN PRESS",
            (packetdata ->> 'AM02') AS "ME P.C.O OUT TEMP.CYL. NO.1",
            (packetdata ->> 'AM03') AS "ME P.C.O OUT TEMP.CYL. NO.2",
            (packetdata ->> 'AM04') AS "ME P.C.O OUT TEMP.CYL. NO.3",
            (packetdata ->> 'AM05') AS "ME P.C.O OUT TEMP.CYL. NO.4",
            (packetdata ->> 'AM06') AS "ME P.C.O OUT TEMP.CYL. NO.5",
            (packetdata ->> 'AM07') AS "ME P.C.O OUT TEMP.CYL. NO.6",
            (packetdata ->> 'AM21') AS "ME EXH. GAS OUT TEMP.CYL. NO.1",
            (packetdata ->> 'AM22') AS "ME EXH. GAS OUT TEMP.CYL. NO.2",
            (packetdata ->> 'AM23') AS "ME EXH. GAS OUT TEMP.CYL. NO.3",
            (packetdata ->> 'AM24') AS "ME EXH. GAS OUT TEMP.CYL. NO.4",
            (packetdata ->> 'AM25') AS "ME EXH. GAS OUT TEMP.CYL. NO.5",
            (packetdata ->> 'AM26') AS "ME EXH. GAS OUT TEMP.CYL. NO.6",
            (packetdata ->> 'AM32') AS "ME T/C 1  EXH. GAS OUT TEMP.",
            (packetdata ->> 'AM50') AS "ME L.O IN PRESS",
            (packetdata ->> 'AM57') AS "ME T/C 1 L.O OUT TEMP.",
            (packetdata ->> 'AM64') AS "ME F.O. IN TEMP.",
            (packetdata ->> 'AM76') AS "ME CYL C.W. IN PRESS",
            (packetdata ->> 'AM83') AS "ME CONTROL AIR PRESS",
            (packetdata ->> 'AM96') AS "ME T/C 1 EXH. GAS IN TEMP.",
            (packetdata ->> 'AM253') AS "ME START AIR PRESS",
            (packetdata ->> 'AM408') AS "ME EXH. GAS TEMP. MEAN VALUE",
            (packetdata ->> 'AM271') AS "STERN TUBE  AFT BRG TEMP.",
            (packetdata ->> 'AM272') AS "SCAV. AIR PRESS IN AIR RECEIVER",
            (packetdata ->> 'AM300') AS "ME J.C.W IN PRESS",
            (packetdata ->> 'AM301') AS "ME J.C.W OUT TEMP.CYL.1",
            (packetdata ->> 'AM302') AS "ME J.C.W OUT TEMP.CYL.2",
            (packetdata ->> 'AM303') AS "ME J.C.W OUT TEMP.CYL.3",
            (packetdata ->> 'AM304') AS "ME J.C.W OUT TEMP.CYL.4",
            (packetdata ->> 'AM305') AS "ME J.C.W OUT TEMP.CYL.5",
            (packetdata ->> 'AM306') AS "ME J.C.W OUT TEMP.CYL.6",
            (packetdata ->> 'AM336') AS "ME L.O IN TEMP.",
            (packetdata ->> 'AM337') AS "ME CYL. L.O IN TEMP.",
            (packetdata ->> 'AM352') AS "ME SCAV. AIR TEMP.  IN SCAV.",
            (packetdata ->> 'AM357') AS "ME EXH. GAS DEV. TEMP. LIM",
            (packetdata ->> 'AM403') AS "ME THRUST SEGMENT TEMP.",
            (packetdata ->> 'AM409') AS "MAIN AIR RESERVIOR NO.1  PRE.",
            (packetdata ->> 'AM410') AS "MAIN AIR RESERVIOR NO.2  PRE.",
            (packetdata ->> 'AM419') AS "ME COPT COND CSW IN TEMP",
            (packetdata ->> 'AM451') AS "ME EXH GAS REST TEMP",
            (packetdata ->> 'AM542') AS "ME SCAV. AIR FIRE DET. TEMP. HIGH PISTON CYL. NO.1 SLD",
            (packetdata ->> 'AM543') AS "ME SCAV. AIR FIRE DET. TEMP. HIGH PISTON CYL. NO.2 SLD",
            (packetdata ->> 'AM544') AS "ME SCAV. AIR FIRE DET. TEMP. HIGH PISTON CYL. NO.3 SLD",
            (packetdata ->> 'AM545') AS "ME SCAV. AIR FIRE DET. TEMP. HIGH PISTON CYL. NO.4 SLD",
            (packetdata ->> 'AM546') AS "ME SCAV. AIR FIRE DET. TEMP. HIGH PISTON CYL. NO.5 SLD",
            (packetdata ->> 'AM547') AS "ME SCAV. AIR FIRE DET. TEMP. HIGH PISTON CYL. NO.6 SLD",
            (packetdata ->> 'AM548') AS "ME MAIN & THRUST BRG L.O. IN LOW PRESS SLD",
            (packetdata ->> 'AM551') AS "ME J.C.W IN LOW PRESS. SLD",
            (packetdata ->> 'AM552') AS "ME J.C.W OUT HIGH TEMP SLD.CYL.1",
            (packetdata ->> 'AM553') AS "ME J.C.W OUT HIGH TEMP SLD.CYL.2",
            (packetdata ->> 'AM554') AS "ME J.C.W OUT HIGH TEMP SLD.CYL.3",
            (packetdata ->> 'AM555') AS "ME J.C.W OUT HIGH TEMP SLD.CYL.4",
            (packetdata ->> 'AM556') AS "ME J.C.W OUT HIGH TEMP SLD.CYL.5",
            (packetdata ->> 'AM557') AS "ME J.C.W OUT HIGH TEMP SLD.CYL.6",
            (packetdata ->> 'AM573') AS "ME P.C.O OUT TEMP CYL 1. SLD",
            (packetdata ->> 'AM574') AS "ME P.C.O OUT TEMP CYL 2. SLD",
            (packetdata ->> 'AM575') AS "ME P.C.O OUT TEMP CYL 3. SLD",
            (packetdata ->> 'AM577') AS "ME P.C.O OUT TEMP CYL 5. SLD",
            (packetdata ->> 'AM578') AS "ME P.C.O OUT TEMP CYL 6. SLD",
            (packetdata ->> 'AM593') AS "STERN TUBE AFT. BRG TEMP. HIGH",
            (packetdata ->> 'AM594') AS "ME EXH. GAS OUT TEMP.CYL. NO.1 SLD",
            (packetdata ->> 'AM595') AS "ME EXH. GAS OUT TEMP.CYL. NO.2 SLD",
            (packetdata ->> 'AM596') AS "ME EXH. GAS OUT TEMP.CYL. NO.3 SLD",
            (packetdata ->> 'AM597') AS "ME EXH. GAS OUT TEMP.CYL. NO.4 SLD",
            (packetdata ->> 'AM598') AS "ME EXH. GAS OUT TEMP.CYL. NO.5 SLD",
            (packetdata ->> 'AM599') AS "ME EXH. GAS OUT TEMP.CYL. NO.6 SLD",
            (packetdata ->> 'AM609') AS "STERN TUBE  AFT BRG TEMP. SLD",
            (packetdata ->> 'AM610') AS "STERN TUBE AFT BRG TEMP. SLD",
            (packetdata ->> 'AM757') AS "ME EXH. GAS AVERAGE TEMP.",
            (packetdata ->> 'DM2') AS "ME BRIDDG CONTROL",
            (packetdata ->> 'DM7') AS "ME START BLOCKED",
            (packetdata ->> 'DM8') AS "ME EMERGENCY STOP",
            (packetdata ->> 'DM9') AS "ME OVERSPEED",
            (packetdata ->> 'DM15') AS "ME WRONG WAY ALARM",
            (packetdata ->> 'DM18') AS "ME RUN",
            (packetdata ->> 'DM22') AS "ME CRITICAL RPM",
            (packetdata ->> 'DM26') AS "ME START FAIL/BLOCK",
            (packetdata ->> 'DM35') AS "ME AXIAL VIBRATION HIGH",
            (packetdata ->> 'DM42') AS "ME T/C LO INLET PRESS. TOO LOW",
            (packetdata ->> 'DM53') AS "ME SHUTDOWN CANCEL",
            (packetdata ->> 'DM72') AS "ME L.O. FILTER DIFFERENTIAL PRESS.. HIGH",
            (packetdata ->> 'DM73') AS "ME L.O. INLET PRESS. LOW",
            (packetdata ->> 'DM76') AS "ME 1 START UP PUMP MOTOR COMM",
            (packetdata ->> 'DM77') AS "ME 2 START UP PUMP MOTOR COMM",
            (packetdata ->> 'DM82') AS "ME MAIN BEARING & P.C.O. PRESS LOW LOW",
            (packetdata ->> 'DM84') AS "ME SCAV. BOX DRAIN TANK LEVEL HIGHI",
            (packetdata ->> 'DM96') AS "ME EXHAUST VALVE SPRING AIR PRESS LOW LOW",
            (packetdata ->> 'DM97') AS "ME THRUST BEARING FORE SIDE TEMP.TOO HIGH SHUTDOWN",
            (packetdata ->> 'DM99') AS "ME STERN TUBE AFT. BEARING TEMP HIGH HIGH",
            (packetdata ->> 'DM102') AS "ME EXH. GAS OUTLET TEMP HIGH HIGH",
            (packetdata ->> 'DM107') AS "ME SAFETY SYSTEM ABNORMAL",
            (packetdata ->> 'DM108') AS "ME SAFETY SYSTEM POWER FAIL",
            (packetdata ->> 'DM109') AS "ME ELECT GOV. SYSTEM ABNORMAL",
            (packetdata ->> 'DM110') AS "ME TELEGRAPH SYSTEM ABNORMAL",
            (packetdata ->> 'DM111') AS "ME TELEGRAPH SYSTEM POWER FAIL",
            (packetdata ->> 'DM114') AS "ME CONTROL SYSTEM ABNORMAL",
            (packetdata ->> 'DM115') AS "ME&DG OIL MIST DETECT FAIL",
            (packetdata ->> 'DM116') AS "ME & DG OIL MIST HIGH",
            (packetdata ->> 'DM117') AS "ME AXIAL VIBRATION SYSTEM FAIL",
            (packetdata ->> 'DM119') AS "ME FO VISCOSITY HIGH",
            (packetdata ->> 'DM121') AS "ME SAFETY SYSTEM POWER FAIL",
            (packetdata ->> 'DM122') AS "ME ELECT GOV SYSTEM ABNORMAL",
            (packetdata ->> 'DM123') AS "ME CYL CRANKCASE OIL MIST HIGH",
            (packetdata ->> 'DM138') AS "ME PCO NON-FLOW CYL 1",
            (packetdata ->> 'DM139') AS "ME PCO NON-FLOW CYL 2",
            (packetdata ->> 'DM140') AS "ME PCO NON-FLOW CYL 3",
            (packetdata ->> 'DM141') AS "ME PCO NON-FLOW CYL 4",
            (packetdata ->> 'DM142') AS "ME PCO NON-FLOW CYL 5",
            (packetdata ->> 'DM143') AS "ME PCO NON-FLOW CYL 6",
            (packetdata ->> 'DM145') AS "ME PCO NON-FLOW CYL 1 SLD",
            (packetdata ->> 'DM146') AS "ME PCO NON-FLOW CYL 2 SLD",
            (packetdata ->> 'DM147') AS "ME PCO NON-FLOW CYL 3 SLD",
            (packetdata ->> 'DM148') AS "ME PCO NON-FLOW CYL 4 SLD",
            (packetdata ->> 'DM149') AS "ME PCO NON-FLOW CYL 5 SLD",
            (packetdata ->> 'DM150') AS "ME PCO NON-FLOW CYL 6 SLD",
            (packetdata ->> 'DM157') AS "ME AUX. BLOWER NO.1 RUNNING",
            (packetdata ->> 'DM158') AS "ME AUX. BLOWER NO.2 RUNNING",
            (packetdata ->> 'DM173') AS "ME JACKET C.W. INLET PRESS. LOW",
            (packetdata ->> 'DM177') AS "ME JACKET C.F.W. PUMP NO.1 RUNNING",
            (packetdata ->> 'DM178') AS "ME JACKET C.F.W. PUMP NO.2 RUNNING",
            (packetdata ->> 'DM181') AS "ME MAIN JACKET C.F.W. TEMP. HIGH ALARM",
            (packetdata ->> 'DM200') AS "ME FO SUP. UNIT NO.1 SUPPLY PUMP",
            (packetdata ->> 'DM201') AS "ME FO SUP. UNIT NO.2 SUPPLY PUMP",
            (packetdata ->> 'DM202') AS "ME FO SUP. UNIT NO.1 CIRC. PUMP",
            (packetdata ->> 'DM203') AS "ME FO SUP. UNIT NO.2 CIRC. PUMP",
            (packetdata ->> 'DM206') AS "ME SLOW DOWN",
            (packetdata ->> 'DM208') AS "ME ORDER PRINTER POWER FAIL",
            (packetdata ->> 'DM256') AS "ME SLD PRE-WARNING",
            (packetdata ->> 'DM782') AS "ME BRIDGE CONTROL",
            (packetdata ->> 'DM830') AS "ME SCAVE. AIR TEMP HIGH HIGH/FIRE No.16  SLD",
            (packetdata ->> 'AF16') AS "AE IN F.O Volume flow",
            (packetdata ->> 'AF17') AS "AE OUT F.O Volume flow",
            (packetdata ->> 'AF29') AS "AE SUPPLY LINE TEMP",
            (packetdata ->> 'AF30') AS "AE RETURN LINE TEMP",
            (packetdata ->> 'AF32') AS "ME+AE COMMON SUPPLY LINE TEMP",
            (packetdata ->> 'AF37') AS "BOILER F.O Volume flow",
            (packetdata ->> 'AF40') AS "BOILER SUPPLY LINE TEMP",
            (packetdata ->> 'AM20') AS "ME RPM",
            (packetdata ->> 'AM266') AS "ME SHAFT TORQUE",
            (packetdata ->> 'AM267') AS "ME SHAFT POWER",
            (packetdata ->> 'DM189') AS "ME HYD. CON. OIL PUMP NO.1 FAULT",
            (packetdata ->> 'DM192') AS "ME HYD. CON. OIL PUMP NO.2 FAULT",
            (packetdata ->> 'DM835') AS "ME P.C.O. TEMP HIGHI SLD",
            (packetdata ->> 'DX1555') AS "ME HYDR OIL LEAK HIGH TRIP(A)",
            (packetdata ->> 'AF1') AS "ME FUEL SUPPLY RATE",
            (packetdata ->> 'VDVBW_1') AS "Longitudinal water speed",
            (packetdata ->> 'stormGlassSwellDirection') AS "Storm Glass Swell Direction",
            (packetdata ->> 'stormGlassCurrentDirection') AS "Storm Glass Current Direction",
            (packetdata ->> 'AM540') AS "ME THRUST BRG FORE SIDE HIGH TEMP SLD.",
            (packetdata ->> 'AM576') AS "ME P.C.O OUT TEMP CYL 4. SLD",
            (packetdata ->> 'DM16') AS "ME ENGINE STARTING FAILED",
            (packetdata ->> 'DM120') AS "ME SAFETY SYSTEM ABNORMAL",
            (packetdata ->> 'DM255') AS "ME SLD CANCEL",
            (packetdata ->> 'DX1556') AS "ME HYDR OIL LEAK HIGH TRIP(B)",
            (packetdata ->> 'DX1139') AS "FO SLUDGE TANK L",
            (packetdata ->> 'AF22') AS "ME+AE FUEL CONSUMPTION RATE",
            (packetdata ->> 'AF9') AS "AE FUEL SUPPLY RATE",
            (packetdata ->> 'AF10') AS "AE FUEL RETURN RATE",
            (packetdata ->> 'AF44') AS "BOILER FUEL CONSUMPTION RATE",
            (packetdata ->> 'AF83') AS "ME+AE FUEL SUPPLY FLOW COUNTER",
            (packetdata ->> 'AF67') AS "AE FUEL SUPPLY FLOW COUNTER",
            (packetdata ->> 'AF69') AS "AE FUEL RETURN FLOW COUNTER",
            (packetdata ->> 'AF71') AS "BOILER FUEL SUPPLY FLOW COUNTER"
        FROM
            shipping_db.highfrequencydata_temp
        JOIN
            shipping_db.ship
        ON
            shipping_db.highfrequencydata_temp.vesselid = shipping_db.ship.id
        WHERE
            vesselid IN ({client_id})
            AND packettime BETWEEN '{start_str}' AND '{end_str}'
        ORDER BY
            packettime ASC
        LIMIT 45000'''
    
        try:
            # Call your API function
            start = datetime.now()
            df = trigger_api(client_name, vessel_name, query, url)
            print(datetime.now() - start)
    
            #data_list.append(df)
            final_df = pd.concat([final_df, df])
    
            # Save output
            # filename = f"vessel_data_ASIA_UNITY_{current_start.strftime('%Y%m%d')}_{current_end.strftime('%Y%m%d')}.csv"
            # df.to_csv(filename, index=False)
        except Exception as e:
            print(f"Not able to find any data points for this period. Error: {e}")
    
    
        # Move to next step
        current_start = current_end + timedelta(seconds=1)
        time.sleep(10)
        print('Sleeping the script for one minute', end='\n')

    print(f"Total rows fetched: {len(final_df)}")

    final_df.set_index('dataTime', inplace=True)
    final_df = final_df.apply(pd.to_numeric, errors='coerce')   
    return final_df



@step
def data_loader() -> pd.DataFrame:
    """Load the dataset."""
    path = "../5_min_lowlands_orange_st_param_highfreq_temp.csv"
    df = pd.read_csv(path, low_memory=False)
    # ### MLflow
    mlflow.set_tag("component", "data_loader")
    mlflow.log_param("data_path", os.path.basename(path))
    mlflow.log_metric("rows_loaded", int(len(df)))
    mlflow.log_metric("cols_loaded", int(df.shape[1]))
    return df


@step
def data_preprocessor(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset."""
    mlflow.set_tag("component", "data_preprocessor")  # ### MLflow

    before_rows = len(df)
    df.reset_index(inplace=True)
    input_df = df[list(variables_used.values())].copy()
    input_df['dataTime'] = pd.to_datetime(input_df['dataTime'])
    input_df = input_df.dropna(how='all', subset=input_df.columns.drop('dataTime'))
    input_df.drop_duplicates(inplace=True)

    # Heuristic split by sampling cadence
    diffs = input_df['dataTime'].diff().dt.seconds
    input_df_1min = input_df[diffs < 250]
    input_df_15min = input_df[diffs > 250]

    input_df_1min = input_df_1min.set_index('dataTime')
    min1_resampled = (
        input_df_1min.resample('5T').apply(lambda x: x.sample(1, random_state=42) if len(x) > 0 else None)
        .dropna(how='all')
    )

    input_df_15min = input_df_15min.set_index('dataTime')
    updated_df = pd.concat([input_df_15min, min1_resampled])
    updated_df = updated_df.dropna(subset=['ME F.O. IN TEMP.', 'ME RPM'])
    updated_df.sort_index(ascending=True, inplace=True)

    # ### MLflow
    mlflow.log_metric("rows_before_preprocess", int(before_rows))
    mlflow.log_metric("rows_after_preprocess", int(len(updated_df)))
    mlflow.log_metric("unique_timestamps", int(updated_df.index.nunique()))
    mlflow.log_param("resample_rule", "5T")

    return updated_df


@step
def preprocess_remove_sensor_errors(updated_df: pd.DataFrame) -> pd.DataFrame:
    """Remove sensor errors from the dataset."""
    mlflow.set_tag("component", "sensor_cleaning")  # ### MLflow

    updated_df = updated_df.apply(pd.to_numeric, errors='coerce')
    range_dict = {
        'ME COPT COND CSW IN TEMP': (0, 70),
        'ME CYL. L.O IN TEMP.': (0, 90),
        'ME EXH. GAS OUT TEMP.CYL. NO.1': (0.1, 600),
        'ME EXH. GAS OUT TEMP.CYL. NO.2': (0.1, 600),
        'ME EXH. GAS OUT TEMP.CYL. NO.3': (0.1, 600),
        'ME EXH. GAS OUT TEMP.CYL. NO.4': (0.1, 600),
        'ME EXH. GAS OUT TEMP.CYL. NO.5': (0.1, 600),
        'ME EXH. GAS OUT TEMP.CYL. NO.6': (0.1, 600),
        'ME T/C 1 EXH. GAS IN TEMP.': (0, 600),
        'ME T/C 1  EXH. GAS OUT TEMP.': (0, 500),
        'ME F.O IN PRESS': (0, 10),
        'ME F.O. IN TEMP.': (0, 150),
        'ME J.C.W OUT HIGH TEMP SLD.CYL.1': (0, 90),
        'ME J.C.W OUT HIGH TEMP SLD.CYL.2': (0, 90),
        'ME J.C.W OUT HIGH TEMP SLD.CYL.3': (0, 90),
        'ME J.C.W OUT HIGH TEMP SLD.CYL.4': (0, 90),
        'ME J.C.W OUT HIGH TEMP SLD.CYL.5': (0, 90),
        'ME J.C.W OUT HIGH TEMP SLD.CYL.6': (0, 90),
        'ME J.C.W IN PRESS': (0, 0.7),
        'ME L.O IN PRESS': (0, 0.6),
        'ME L.O IN TEMP.': (0, 80),
        'SCAV. AIR PRESS IN AIR RECEIVER': (-1, 3.5),
        'ME SCAV. AIR TEMP.  IN SCAV.': (0, 60),
        'ME SCAV. AIR FIRE DET. TEMP. HIGH PISTON CYL. NO.1 SLD': (15, 130),
        'ME SCAV. AIR FIRE DET. TEMP. HIGH PISTON CYL. NO.2 SLD': (15, 130),
        'ME SCAV. AIR FIRE DET. TEMP. HIGH PISTON CYL. NO.3 SLD': (15, 130),
        'ME SCAV. AIR FIRE DET. TEMP. HIGH PISTON CYL. NO.4 SLD': (15, 130),
        'ME SCAV. AIR FIRE DET. TEMP. HIGH PISTON CYL. NO.5 SLD': (15, 130),
        'ME SCAV. AIR FIRE DET. TEMP. HIGH PISTON CYL. NO.6 SLD': (15, 130),
        'ME SHAFT POWER': (0, 9000),
        'ME RPM': (-100, 99),
    }

    total_replaced = 0
    for col, (min_val, max_val) in range_dict.items():
        out_of_range_mask = (updated_df[col] < min_val) | (updated_df[col] > max_val)
        count = int(out_of_range_mask.sum())
        total_replaced += count
        updated_df.loc[out_of_range_mask, col] = np.nan
        # ### MLflow
        mlflow.log_metric(f"oor_count__{col}", count)
        mlflow.log_param(f"range__{col}", f"[{min_val}, {max_val}]")

    mlflow.log_metric("total_values_replaced", int(total_replaced))  # ### MLflow
    return updated_df


@step
def steady_state_extraction(updated_df: pd.DataFrame) -> pd.DataFrame:
    mlflow.set_tag("component", "steady_state_extraction")  # ### MLflow

    updated_df = updated_df.sort_values(by=updated_df.index.name or 'dataTime')

    def steady_state_extraction_core(series, distance_threshold, alpha=0.2, L=20, step=1):
        series_smoothed = series.ewm(alpha=alpha).mean()
        windows = []
        for start in range(0, len(series_smoothed) - L + 1, step):
            window = series_smoothed.values[start:start+L]
            windows.append(window)

        seq_labels = []
        for window in windows:
            model = AgglomerativeClustering(n_clusters=None, distance_threshold=distance_threshold)
            states = model.fit_predict(window.reshape(-1, 1))
            if len(np.unique(states)) == 1:
                seq_labels.append(1)
            else:
                r = len(np.unique(states))
                C = np.zeros((r, r), dtype=int)
                for t in range(len(states)-1):
                    i, j = states[t], states[t+1]
                    C[i, j] += 1
                P = np.zeros_like(C, dtype=float)
                for i in range(r):
                    row_sum = C[i].sum()
                    if row_sum > 0:
                        P[i] = C[i] / row_sum
                import cv2
                B = (P > 0).astype(np.uint8)
                num_labels, labels_img = cv2.connectedComponents(B, connectivity=4)
                counts = np.bincount(labels_img.flatten())[1:]
                main_label = np.argmax(counts) + 1 if counts.size > 0 else 0
                seq_labels.append(main_label)

        T = len(series_smoothed)
        final_labels = np.zeros(T, dtype=int)
        for t in range(T):
            i_min = max(0, t - L + 1)
            i_max = min(t, len(windows) - 1)
            labels_here = [seq_labels[i] for i in range(i_min, i_max+1)]
            final_labels[t] = labels_here[0] if (len(labels_here) > 0 and all(l == labels_here[0] for l in labels_here)) else 0
        return final_labels

    distance_threshold = 700
    alpha = 0.2
    L = 20
    step_sz = 1
    labels_me_load = steady_state_extraction_core(updated_df['ME SHAFT POWER'], distance_threshold, alpha, L, step_sz)

    steady_df = updated_df[labels_me_load != 0]

    # ### MLflow
    mlflow.log_param("ssd_distance_threshold", distance_threshold)
    mlflow.log_param("ssd_ewma_alpha", alpha)
    mlflow.log_param("ssd_window_length", L)
    mlflow.log_param("ssd_window_step", step_sz)
    mlflow.log_metric("steady_state_fraction", float(len(steady_df) / max(1, len(updated_df))))
    mlflow.log_metric("steady_rows", int(len(steady_df)))

    return steady_df


@step
def further_filtering(steady_df: pd.DataFrame) -> pd.DataFrame:
    """Further filter the dataset."""
    mlflow.set_tag("component", "further_filtering")  # ### MLflow

    before = len(steady_df)
    power_thr = 3500
    cyl1_temp_thr = 260

    mlflow.log_param("power_threshold", power_thr)
    mlflow.log_param("cyl1_temp_threshold", cyl1_temp_thr)

    steady_df = steady_df[steady_df['ME SHAFT POWER'] > power_thr]
    steady_df = steady_df[(steady_df['ME EXH. GAS OUT TEMP.CYL. NO.1'] > cyl1_temp_thr)]
    steady_df = steady_df.dropna()

    scaler_pca = StandardScaler()
    X_scaled_pca = scaler_pca.fit_transform(steady_df)

    n_comp = 15
    pca_15 = PCA(n_components=n_comp)
    X_pca_15 = pca_15.fit_transform(X_scaled_pca)

    X_reconstructed = pca_15.inverse_transform(X_pca_15)
    reconstruction_error_pca = np.mean((X_scaled_pca - X_reconstructed)**2, axis=1)

    perc_99 = float(np.percentile(reconstruction_error_pca, 99))
    steady_df = steady_df[reconstruction_error_pca <= perc_99]

    # ### MLflow
    mlflow.log_param("pca_n_components", n_comp)
    mlflow.log_metric("rows_before_filtering", int(before))
    mlflow.log_metric("rows_after_filtering", int(len(steady_df)))
    mlflow.log_metric("pca_99th_percentile_error", perc_99)

    # Save PCA explained variance for inspection
    evr = getattr(pca_15, 'explained_variance_ratio_', None)
    if evr is not None:
        mlflow.log_dict({"explained_variance_ratio": evr.tolist()}, artifact_file="pca/explained_variance_ratio.json")

    return steady_df


@step
def data_standardization(steady_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """Standardize the dataset."""
    mlflow.set_tag("component", "data_standardization")  # ### MLflow

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(steady_df)

    X_train, X_valid = train_test_split(X_scaled, train_size=0.8, random_state=42)

    # ### MLflow: Save scaler + basic stats
    with tempfile.TemporaryDirectory() as td:
        sc_path = os.path.join(td, "scaler_minmax.pkl")
        with open(sc_path, 'wb') as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact(sc_path, artifact_path="preprocessing")
    mlflow.log_metric("n_train", int(len(X_train)))
    mlflow.log_metric("n_valid", int(len(X_valid)))

    return X_train, X_valid, scaler


@step  # (enable_cache=False)
def model_trainer(X_train: np.ndarray, X_valid: np.ndarray) -> tf.keras.Model:
    """Train the autoencoder model."""
    mlflow.set_tag("component", "model_trainer")  # ### MLflow

    # Enable autologging for Keras/TensorFlow to capture epochs, params, and artifacts automatically
    mlflow.tensorflow.autolog(log_models=False)  # we'll log the model manually for clarity

    tf.random.set_seed(42)

    @keras.saving.register_keras_serializable()
    class SamplingLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(SamplingLayer, self).__init__(**kwargs)
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        def get_config(self):
            return super(SamplingLayer, self).get_config()

    @keras.saving.register_keras_serializable()
    def sampling_function(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    def create_variational_autoencoder(input_dim, latent_dim=8):
        encoder_inputs = tf.keras.Input(shape=(input_dim,))
        x = tf.keras.layers.Dense(128, activation="relu")(encoder_inputs)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
        z = SamplingLayer(name="sampling")([z_mean, z_log_var])
        decoder_inputs = tf.keras.Input(shape=(latent_dim,))
        x = tf.keras.layers.Dense(32, activation="relu")(decoder_inputs)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        decoder_outputs = tf.keras.layers.Dense(input_dim, activation="sigmoid")(x)
        encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        decoder = tf.keras.Model(decoder_inputs, decoder_outputs, name="decoder")
        outputs = decoder(encoder(encoder_inputs)[2])
        vae = tf.keras.Model(encoder_inputs, outputs, name="vae")
        # simple MSE loss; KL handled via structure if needed
        def vae_loss(inputs, outputs):
            reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
            z_mean_batch, z_log_var_batch, _ = encoder(inputs)  # FIX: encoder returns (z_mean, z_log_var, z)
            kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var_batch - tf.square(z_mean_batch) - tf.exp(z_log_var_batch))
            return reconstruction_loss + kl_loss * 0.1
        vae.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=vae_loss)
        return vae, encoder, decoder

    INPUT_DIM = X_train.shape[1]
    autoencoder, encoder, decoder = create_variational_autoencoder(INPUT_DIM)

    autoencoder.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=0.01),
        loss="huber",
        metrics=["mse", "mae"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=10, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(dt.datetime.now().strftime("%Y%m%d-%H%M%S") + '_best_autoencoder.keras', save_best_only=True),
    ]

    log_dir = "logs/autoencoder/" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # ### MLflow: log training params
    mlflow.log_param("input_dim", int(INPUT_DIM))
    mlflow.log_param("optimizer", "AdamW")
    mlflow.log_param("lr", 1e-4)
    mlflow.log_param("weight_decay", 0.01)
    mlflow.log_param("loss", "huber")
    mlflow.log_param("metrics", "mse,mae")
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("epochs", 50)
    mlflow.log_param("es_patience", 30)
    mlflow.log_param("rlr_factor", 0.3)
    mlflow.log_param("rlr_patience", 10)

    history = autoencoder.fit(
        X_train, X_train,
        epochs=10,
        batch_size=16,
        validation_data=(X_valid, X_valid),
        callbacks=[tensorboard_callback] + callbacks,
        verbose=1,
    )

    # ### MLflow: log learning curves
    for k, v in history.history.items():
        for epoch, val in enumerate(v):
            mlflow.log_metric(k, float(val), step=epoch)

    # Evaluate & log
    train_loss = autoencoder.evaluate(X_train, X_train, verbose=0)
    val_loss = autoencoder.evaluate(X_valid, X_valid, verbose=0)
    mlflow.log_metric("final_train_loss", float(train_loss if np.isscalar(train_loss) else train_loss[0]))
    mlflow.log_metric("final_val_loss", float(val_loss if np.isscalar(val_loss) else val_loss[0]))

    # Log small reconstruction sample
    reconstructions = autoencoder.predict(X_valid[:10])
    re_err = float(np.mean((X_valid[0] - reconstructions[0])**2))
    mlflow.log_metric("sample_reconstruction_error", re_err)

    # Save textual model summary
    stream = io.StringIO()
    autoencoder.summary(print_fn=lambda x: stream.write(x + "\n"))
    mlflow.log_text(stream.getvalue(), artifact_file="autoencoder_summary.txt")

    # Log model
    mlflow.keras.log_model(autoencoder, name="autoencoder")

    return autoencoder


@step
def model_evaluator(model: tf.keras.Model, X_scaled: np.ndarray) -> Dict[str, Any]:
    """Evaluate the trained model."""
    mlflow.set_tag("component", "model_evaluator")  # ### MLflow

    reconstructions = model.predict(X_scaled)
    reconstruction_error = np.mean((X_scaled - reconstructions)**2, axis=1)

    # Summary stats
    mlflow.log_metric("recon_error_mean", float(np.mean(reconstruction_error)))
    mlflow.log_metric("recon_error_std", float(np.std(reconstruction_error)))
    mlflow.log_metric("recon_error_p95", float(np.percentile(reconstruction_error, 95)))

    # Histogram plot
    fig = plt.figure(figsize=(7, 5))
    plt.hist(reconstruction_error, bins=50)
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("MSE")
    plt.ylabel("Count")
    mlflow.log_figure(fig, artifact_file="plots/reconstruction_error_hist.png")
    plt.close(fig)

    return {"reconstruction_error": reconstruction_error}


@step(enable_cache=False)
def evaluate_autoencoder_step(
    excel_path: str,
    variables_used: Dict[str, str],
    final_df1: pd.DataFrame,
    scaler,  # sklearn scaler
    autoencoder: Any,  # keras model
    drop_cols: Optional[list] = None,
    anomaly_col: str = "Anomaly Reason",
    baseline_threshold: Optional[float] = None,
    normal_sample_size: int = 3000,
) -> Dict[str, Union[float, int, str]]:
    """Evaluate the autoencoder and log PR tradeoffs to MLflow."""
    mlflow.set_tag("component", "evaluation")  # ### MLflow
    mlflow.set_tag("stage", "test")

    test = pd.read_excel(excel_path)
    anomalies = test[test[anomaly_col].notna()].copy()
    if variables_used:
        anomalies = anomalies.rename(columns=variables_used)

    n = min(normal_sample_size, len(final_df1))
    normals = final_df1.sample(n=n, random_state=42).copy()

    common_cols = list(set(anomalies.columns).intersection(set(normals.columns)))
    if anomaly_col not in common_cols and anomaly_col in anomalies.columns:
        common_cols += [anomaly_col]
    anomalies = anomalies[common_cols].copy()
    normals = normals[[c for c in common_cols if c != anomaly_col]].copy()
    normals[anomaly_col] = np.nan

    test_data = pd.concat([anomalies, normals], ignore_index=True)

    if drop_cols is None:
        drop_cols = ['me_load', "ME SCAV AIR BOX TEMP (°C)"]
    actually_dropped = [c for c in drop_cols if c in test_data.columns]
    test_data = test_data.drop(columns=actually_dropped, errors='ignore')

    test_data[anomaly_col] = test_data[anomaly_col].apply(lambda x: 0 if pd.isna(x) else 1)
    y_true = test_data[anomaly_col].astype(int).values

    # Ensure order matches training data
    X = test_data.drop(columns=[anomaly_col])
    X = X[final_df1.columns]
    X_scaled = scaler.transform(X)

    # ### MLflow params from evaluation config
    mlflow.log_param("eval_excel", os.path.basename(excel_path))
    mlflow.log_param("eval_normal_sample_size", int(n))
    mlflow.log_param("eval_drop_cols", ",".join(actually_dropped))
    if baseline_threshold is not None:
        mlflow.log_param("baseline_threshold", float(baseline_threshold))

    # Inference & PR
    x_pred = autoencoder.predict(X_scaled)
    reconstruction_error = np.mean(np.abs(X_scaled - x_pred), axis=1)

    precisions, recalls, thresholds = precision_recall_curve(y_true, reconstruction_error)
    pr_auc = auc(recalls, precisions)

    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-12)
    best_idx = int(np.argmax(f1_scores))
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    best_precision = float(precisions[best_idx])
    best_recall = float(recalls[best_idx])

    y_pred_best = (reconstruction_error > best_threshold).astype(int)
    cm_best = confusion_matrix(y_true, y_pred_best)
    tn, fp, fn, tp = cm_best.ravel()
    accuracy_best = float((tp + tn) / cm_best.sum())

    cls_report_best = classification_report(y_true, y_pred_best, digits=4)

    # Baseline metrics
    baseline_metrics = {}
    if baseline_threshold is not None:
        y_pred_base = (reconstruction_error > baseline_threshold).astype(int)
        cm_base = confusion_matrix(y_true, y_pred_base)
        tn_b, fp_b, fn_b, tp_b = cm_base.ravel()
        precision_b = float(tp_b / (tp_b + fp_b + 1e-12))
        recall_b = float(tp_b / (tp_b + fn_b + 1e-12))
        f1_b = float(2 * precision_b * recall_b / (precision_b + recall_b + 1e-12))
        accuracy_b = float((tp_b + tn_b) / cm_base.sum())
        cls_report_base = classification_report(y_true, y_pred_base, digits=4)
        baseline_metrics = {
            "baseline_threshold": float(baseline_threshold),
            "baseline_precision": precision_b,
            "baseline_recall": recall_b,
            "baseline_f1": f1_b,
            "baseline_accuracy": accuracy_b,
            "baseline_tn": int(tn_b),
            "baseline_fp": int(fp_b),
            "baseline_fn": int(fn_b),
            "baseline_tp": int(tp_b),
            "baseline_classification_report": cls_report_base,
        }

    # ### MLflow metrics
    mlflow.log_metric("pr_auc", float(pr_auc))
    mlflow.log_metric("best_threshold", best_threshold)
    mlflow.log_metric("best_precision", best_precision)
    mlflow.log_metric("best_recall", best_recall)
    mlflow.log_metric("best_f1", best_f1)
    mlflow.log_metric("best_accuracy", accuracy_best)
    mlflow.log_metric("tn", int(tn))
    mlflow.log_metric("fp", int(fp))
    mlflow.log_metric("fn", int(fn))
    mlflow.log_metric("tp", int(tp))

    for k, v in baseline_metrics.items():
        if isinstance(v, (int, float)):
            mlflow.log_metric(k, v)

    # Text artifacts
    mlflow.log_text(cls_report_best, artifact_file="reports/classification_report_best.txt")
    if baseline_metrics.get("baseline_classification_report"):
        mlflow.log_text(baseline_metrics["baseline_classification_report"], artifact_file="reports/classification_report_baseline.txt")

    # Curves as CSV artifacts
    pr_df = pd.DataFrame({
        "threshold": np.r_[thresholds, np.nan],
        "precision": precisions,
        "recall": recalls,
    })
    with tempfile.TemporaryDirectory() as td:
        pr_path = os.path.join(td, "pr_curve_points.csv")
        pr_df.to_csv(pr_path, index=False)
        mlflow.log_artifact(pr_path, artifact_path="curves")

    # Plots
    fig1 = plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1], label="Recall")
    if baseline_threshold is not None:
        plt.axvline(baseline_threshold, linestyle="--", label=f"Baseline = {baseline_threshold}")
    plt.axvline(best_threshold, linestyle=":", label=f"Best = {best_threshold:.4f}")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision and Recall vs Threshold")
    plt.legend()
    plt.grid(True)
    mlflow.log_figure(fig1, artifact_file="plots/precision_recall_vs_threshold.png")
    plt.close(fig1)

    fig2 = plt.figure(figsize=(6, 6))
    im = plt.imshow(cm_best, interpolation="nearest")
    plt.title("Confusion Matrix (Best Threshold)")
    plt.colorbar(im)
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Normal (0)", "Anomaly (1)"])
    plt.yticks(tick_marks, ["Normal (0)", "Anomaly (1)"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm_best[i, j], ha="center", va="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    mlflow.log_figure(fig2, artifact_file="plots/confusion_matrix_best.png")
    plt.close(fig2)

    # raw errors as CSV
    with tempfile.TemporaryDirectory() as td:
        err_path = os.path.join(td, "reconstruction_errors.csv")
        pd.DataFrame({"reconstruction_error": reconstruction_error, "y_true": y_true}).to_csv(err_path, index=False)
        mlflow.log_artifact(err_path, artifact_path="data")

    result = {
        "pr_auc": float(pr_auc),
        "best_threshold": best_threshold,
        "best_precision": best_precision,
        "best_recall": best_recall,
        "best_f1": best_f1,
        "best_accuracy": accuracy_best,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    if baseline_threshold is not None:
        result.update({
            "baseline_threshold": float(baseline_threshold),
            **{k: v for k, v in baseline_metrics.items() if k in {
                "baseline_precision", "baseline_recall", "baseline_f1", "baseline_accuracy"
            }},
        })

    return result


@pipeline#(enable_cache=True, experiment_tracker="mlflow_tracker")
def Autoencoder_5min_31_pipeline():
    """Define the training pipeline."""
    df = download_data('aesm', 'LOWLANDS ORANGE', 8)
    # df = data_loader()
    updated_df = data_preprocessor(df)
    updated_df = preprocess_remove_sensor_errors(updated_df)
    steady_df = steady_state_extraction(updated_df)
    filtered_df = further_filtering(steady_df)
    X_train, X_valid, scaler = data_standardization(filtered_df)
    model = model_trainer(X_train, X_valid)
    evaluation_results = model_evaluator(model, X_valid)
    results = evaluate_autoencoder_step(
        excel_path="../../Lowland Orange CBM Data For Model Building.xlsx",
        variables_used=variables_used,
        final_df1=filtered_df,
        scaler=scaler,
        autoencoder=model,
        drop_cols=['me_load', "ME SCAV AIR BOX TEMP (°C)"],
        anomaly_col="Anomaly Reason",
        baseline_threshold=None,
        normal_sample_size=3000,
    )
    return model, evaluation_results, results


if __name__ == "__main__":
    # Run the pipeline
    Autoencoder_5min_31_pipeline()
