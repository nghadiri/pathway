import pandas as pd
import pm4py
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.visualization.dfg import visualizer as dfg_visualization
from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.visualization.heuristics_net import visualizer as hn_visualizer
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.visualization.process_tree import visualizer as pt_visualizer
from pm4py.statistics.traces.generic.log import case_statistics
import os

csv_path = os.path.join(os.getcwd(), 'data', '4_sepsis_care_analysis_dataset.csv')

df = pd.read_csv(csv_path)

# Convert datetime columns
df['sepsis_onset_time'] = pd.to_datetime(df['sepsis_onset_time'])
df['event_time'] = pd.to_datetime(df['event_time'])

# Calculate time since sepsis onset
df['time_since_onset'] = (df['event_time'] - df['sepsis_onset_time']).dt.total_seconds() / 3600

# Filter events within 24 hours of sepsis onset
df_24h = df[df['time_since_onset'] <= 24]

# Prepare the event log
event_log = pm4py.format_dataframe(
    df_24h,
    case_id='hadm_id',
    activity_key='event_type',
    timestamp_key='event_time'
)

# Discovery Algorithms

# 1. Directly-Follows Graph
dfg = dfg_discovery.apply(event_log)
dfg_gviz = dfg_visualization.apply(dfg, log=event_log, variant=dfg_visualization.Variants.FREQUENCY)
dfg_visualization.save(dfg_gviz, "sepsis_dfg.png")

# 2. Heuristics Net
heu_net = heuristics_miner.apply_heu(event_log, parameters={
    heuristics_miner.Variants.CLASSIC.value.Parameters.DEPENDENCY_THRESH: 0.5
})
hn_gviz = hn_visualizer.apply(heu_net)
hn_visualizer.save(hn_gviz, "sepsis_heuristics_net.png")

# 3. Process Tree
tree = inductive_miner.apply(event_log)
tree_gviz = pt_visualizer.apply(tree)
pt_visualizer.save(tree_gviz, "sepsis_process_tree.png")

# Performance Analysis

# 1. Case Duration
case_durations = case_statistics.get_case_durations(event_log)
avg_case_duration = sum(case_durations) / len(case_durations)
print(f"Average case duration: {avg_case_duration:.2f} hours")

# 2. Activity Frequency
activities = df_24h['event_type'].value_counts()
print("\nTop 10 most frequent activities:")
print(activities.head(10))

# 3. Bottleneck Analysis
dfg_performance = dfg_discovery.apply(event_log, variant=dfg_discovery.Variants.PERFORMANCE)
dfg_perf_gviz = dfg_visualization.apply(dfg_performance, log=event_log, variant=dfg_visualization.Variants.PERFORMANCE)
dfg_visualization.save(dfg_perf_gviz, "sepsis_performance_dfg.png")

print("\nProcess mining analysis complete. Please check the generated PNG files for visualizations.")