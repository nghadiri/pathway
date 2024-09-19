import pandas as pd
import pm4py
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.alpha import algorithm as alpha_miner
#from pm4py.visualization.petrinet import visualizer as pn_visualizer
from pm4py.visualization.process_tree import visualizer as pt_visualizer
import os

csv_path = os.path.join(os.getcwd(), 'data', '4_sepsis_care_analysis_dataset.csv')

# Step 1: Load the CSV files
sepsis_data = pd.read_csv(csv_path)

# Step 2: Prepare the data for process mining
# Ensure the timestamp is in the correct format
sepsis_data['event_time'] = pd.to_datetime(sepsis_data['event_time'])

# Rename columns to match pm4py expectations: case_id (subject_id), event_name, timestamp
sepsis_data.rename(columns={
    'subject_id': 'case_id',
    'event_type': 'event_name',
    'event_time': 'time:timestamp'
}, inplace=True)

# Filter out rows where event_name is NaN (optional, if needed)
sepsis_data.dropna(subset=['event_name'], inplace=True)

# Convert the dataframe to an event log
sepsis_data = dataframe_utils.convert_timestamp_columns_in_df(sepsis_data)
event_log = log_converter.apply(sepsis_data)

# Step 3: Mine the process model using the Alpha Miner algorithm
net, initial_marking, final_marking = alpha_miner.apply(event_log)

# Step 4: Visualize the process model
#gviz = pn_visualizer.apply(net, initial_marking, final_marking)
#pn_visualizer.view(gviz)
gviz = pt_visualizer.apply(net)
pt_visualizer.view(gviz)
