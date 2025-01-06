from gtda.diagrams import Amplitude
from tqdm import tqdm
import snelweg.datatools2 as dt
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from gtda.graphs import GraphGeodesicDistance
from gtda.homology import FlagserPersistence
from spacetime.TDA_trial import specific_graph_old
import plotly.io as pio

pio.renderers.default = 'browser'

try:
    print(data)
except Exception as e:
    data = dt.NDW_handler('snelweg/ochtend_alles/A20Gouda_alles_ochtend.csv')


def graph_forward_info_speed(subgraph, speeds):
    """
    :param subgraph: (sub)graph on which to calculate
    :param speeds: measured speeds
    :return: complete (directed) graph with time distances
    """
    time_graph = subgraph.copy()
    for out, to, data in time_graph.edges(data=True):
        data['weight'] = 60 * 2 * data['weight'] / (speeds[out] + speeds[to])
    return time_graph


date_index = 98
date_df = data.dfs[date_index]
n = len(date_df)
t_index = 0
target = 'RWS01_MONIBAS_0201hrr0413ra'
labels, g = specific_graph_old(data)
subg = g.subgraph(range(14))
test_speeds = [{label: date_df['speed_' + labels[label]].to_numpy()[i] for label in labels} for i in range(n)]
test_intensities = [{label: date_df['intensity_' + labels[label]].to_numpy()[i] for label in labels} for i in range(n)]

d0, d1 = [], []
for i in tqdm([120]):
    subd = graph_forward_info_speed(subg, test_speeds[i])
    # adj_mat = nx.adjacency_matrix(subd)
    adj_mat = nx.floyd_warshall_numpy(subd)

    flagser = FlagserPersistence()
    # flagser.fit_transform_plot([adj_mat])

    x_ggd = GraphGeodesicDistance(directed=True, unweighted=False).fit_transform([adj_mat])
    dgms = flagser.fit_transform_plot(x_ggd)
    amp = Amplitude(n_jobs=6)
    amps = amp.fit_transform([dgms])
    d0.append(amps[0][0])
    d1.append(amps[0][1])

fig, ax = plt.subplots()

fig.suptitle(f'Forward, {dt.df_date(data.dfs, date_index)}, i={date_index}')
ax.plot(date_df['speed_' + target].to_numpy(), label='Speed')
ax2 = ax.twinx()
# ax2.plot(d0, label='Dim0 homology amplitude')
ax2.plot(d1, 'r--', label='Dim1 homology amplitude')

s = date_df['speed_' + target].to_numpy()
plt.figure()
plt.plot(s[1:]-s[:-1])
