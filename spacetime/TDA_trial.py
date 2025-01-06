import gudhi as gd
import numpy as np
import networkx as nx
import pandas as pd
import snelweg.datatools2 as dt
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
from ripser import Rips
import warnings
import persim

plt.rcParams.update({'text.usetex': False})
matplotlib.use('Qt5Agg')
import seaborn
seaborn.set_theme(style='whitegrid')


def specific_graph(data):
    def scrap_1(x):
        if "_1" == x[-2:]:
            return x[:-2]
        else:
            return x

    nodes = [(i, {"name": loc}) for i, loc in enumerate(data.locations)]
    A20_locs = [loc for loc in nodes if "0201" in loc[1]['name'][:-6]]
    A16_locs = [loc for loc in nodes if "0161" in loc[1]['name'][:-6]]
    A13_locs = [loc for loc in nodes if "0131" in loc[1]['name'][:-6]]
    vwy = [loc for loc in nodes if "vwy" in loc[1]['name'][:-6]][0]
    A15 = [loc for loc in nodes if "015" in loc[1]['name'][:-6]][0]
    geo = [loc for loc in nodes if "GEO" in loc[1]['name'][:-6]][0]
    edges = []
    # t0 = data.dfs[df_index][t_index:t_index+1]
    for loc0, loc1 in zip(A20_locs[:-1], A20_locs[1:]):
        dist = abs(int(scrap_1(loc0[1]['name'])[-6:-2]) - int(scrap_1(loc1[1]['name'])[-6:-2])) / 10
        # avgspeed = (t0['speed_' + loc0[1]['name']] + t0['speed_' + loc1[1]['name']]).iloc[0] / 2
        edges.append((loc0[0], loc1[0], dist))  # distance in minutes
    for loc0, loc1 in zip(A13_locs[:-1], A13_locs[1:]):
        dist = abs(int(loc0[1]['name'][-6:-2]) - int(loc1[1]['name'][-6:-2])) / 10
        # avgspeed = (t0['speed_' + loc0[1]['name']] + t0['speed_' + loc1[1]['name']]).iloc[0] / 2
        edges.append((loc0[0], loc1[0], dist))  # distance in minutes
    for loc0, loc1 in zip(A16_locs[:-1], A16_locs[1:]):
        dist = abs(int(scrap_1(loc0[1]['name'])[-6:-2]) - int(scrap_1(loc1[1]['name'])[-6:-2])) / 10
        # avgspeed = (t0['speed_' + loc0[1]['name']] + t0['speed_' + loc1[1]['name']]).iloc[0] / 2
        edges.append((loc1[0], loc0[0], dist))  # distance in minutes
    # edges.append((A16_locs[1][0], A16_locs[0][0],  # A16 linkjes
    #               12 * abs(int(A16_locs[0][1]['name'][-6:-2]) - int(A16_locs[1][1]['name'][-6:-2])) /
    #               (t0['speed_' + A16_locs[0][1]['name']] + t0['speed_' + A16_locs[1][1]['name']]).iloc[0]))
    # edges.append((A16_locs[1][0], vwy[0],  # A16 en afrit
    #               12 * abs(int(A16_locs[0][1]['name'][-6:-2]) - int(A16_locs[1][1]['name'][-6:-2])) /
    #               (t0['speed_' + vwy[1]['name']] + t0['speed_' + A16_locs[1][1]['name']]).iloc[0]))
    # edges.append((A15[0], vwy[0],  # A15 naar afrit :(
    #               12 * abs(int(A16_locs[0][1]['name'][-6:-2]) - int(A16_locs[1][1]['name'][-6:-2])) /
    #               (t0['speed_' + A15[1]['name']] + t0['speed_' + vwy[1]['name']]).iloc[0]))
    # edges.append((geo[0], A20_locs[1][0],  # Geo naar A20
    #               12 * abs(int(A20_locs[0][1]['name'][-6:-2]) - int(A20_locs[1][1]['name'][-6:-2])) /
    #               (t0['speed_' + geo[1]['name']] + t0['speed_' + A20_locs[1][1]['name']]).iloc[0]))
    # edges.append((A13_locs[-1][0], A20_locs[3][0],  # A13 naar A20
    #               12 * 24 /
    #               (t0['speed_' + A13_locs[-1][1]['name']] + t0['speed_' + A20_locs[3][1]['name']]).iloc[0]))
    # edges.append((A16_locs[0][0], A20_locs[4][0],  # A16 naar A20 maar dan oeps
    #               12 * 84 /
    #               (t0['speed_' + A16_locs[0][1]['name']] + t0['speed_' + A20_locs[4][1]['name']]).iloc[0]))

    edges.append((A16_locs[-1][0], vwy[0],  # A16 naar naastweg
                  abs(int(A16_locs[-1][1]['name'][-6:-2]) - int(A16_locs[-2][1]['name'][-6:-2])) / 10))
    edges.append((vwy[0], A16_locs[0][0],  # naastweg weer naar A16
                  abs(int(A16_locs[-2][1]['name'][-6:-2]) - int(A16_locs[0][1]['name'][-6:-2])) / 10))
    edges.append((A15[0], vwy[0],  # A15 naar afrit
                  abs(int(A16_locs[0][1]['name'][-6:-2]) - int(A16_locs[-2][1]['name'][-6:-2])) / 10))
    edges.append((geo[0], A20_locs[1][0],  # Geo naar A20
                  abs(int(A20_locs[0][1]['name'][-6:-2]) - int(A20_locs[1][1]['name'][-6:-2])) / 10))
    edges.append((A13_locs[-1][0], A20_locs[3][0],  # A13 naar A20
                  2.4))
    edges.append((A16_locs[0][0], A20_locs[5][0],  # A16 naar A20
                  2.8))

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    return nx.get_node_attributes(G, 'name'), G


def specific_graph_old(data):
    def scrap_1(x):
        if "_1" == x[-2:]:
            return x[:-2]
        else:
            return x

    nodes = [(i, {"name": loc}) for i, loc in enumerate(data.locations)]
    A20_locs = [loc for loc in nodes if "0201" in loc[1]['name'][:-6]]
    A16_locs = [loc for loc in nodes if "0161" in loc[1]['name'][:-6]]
    A13_locs = [loc for loc in nodes if "0131" in loc[1]['name'][:-6]]
    vwy = [loc for loc in nodes if "vwy" in loc[1]['name'][:-6]][0]
    A15 = [loc for loc in nodes if "015" in loc[1]['name'][:-6]][0]
    geo = [loc for loc in nodes if "GEO" in loc[1]['name'][:-6]][0]
    edges = []
    # t0 = data.dfs[df_index][t_index:t_index+1]
    for loc0, loc1 in zip(A20_locs[:-1], A20_locs[1:]):
        dist = abs(int(scrap_1(loc0[1]['name'])[-6:-2]) - int(scrap_1(loc1[1]['name'])[-6:-2])) / 10
        # avgspeed = (t0['speed_' + loc0[1]['name']] + t0['speed_' + loc1[1]['name']]).iloc[0] / 2
        edges.append((loc0[0], loc1[0], dist))  # distance in minutes
    for loc0, loc1 in zip(A13_locs[:-1], A13_locs[1:]):
        dist = abs(int(loc0[1]['name'][-6:-2]) - int(loc1[1]['name'][-6:-2])) / 10
        # avgspeed = (t0['speed_' + loc0[1]['name']] + t0['speed_' + loc1[1]['name']]).iloc[0] / 2
        edges.append((loc0[0], loc1[0], dist))  # distance in minutes
    for loc0, loc1 in zip(A16_locs[:-1], A16_locs[1:]):
        dist = abs(int(scrap_1(loc0[1]['name'])[-6:-2]) - int(scrap_1(loc1[1]['name'])[-6:-2])) / 10
        # avgspeed = (t0['speed_' + loc0[1]['name']] + t0['speed_' + loc1[1]['name']]).iloc[0] / 2
        edges.append((loc1[0], loc0[0], dist))  # distance in minutes
    edges.append((A16_locs[1][0], A16_locs[0][0],  # A16 linkjes
                  abs(int(A16_locs[0][1]['name'][-6:-2]) - int(A16_locs[1][1]['name'][-6:-2])) / 10))
    edges.append((A16_locs[1][0], vwy[0],  # A16 en afrit
                  abs(int(A16_locs[0][1]['name'][-6:-2]) - int(A16_locs[1][1]['name'][-6:-2])) / 10))
    edges.append((A15[0], vwy[0],  # A15 naar afrit :(
                  abs(int(A16_locs[0][1]['name'][-6:-2]) - int(A16_locs[1][1]['name'][-6:-2])) / 10))
    edges.append((geo[0], A20_locs[1][0],  # Geo naar A20
                  abs(int(A20_locs[0][1]['name'][-6:-2]) - int(A20_locs[1][1]['name'][-6:-2])) / 10))
    edges.append((A13_locs[-1][0], A20_locs[3][0],  # A13 naar A20
                  2.4))
    edges.append((A16_locs[0][0], A20_locs[4][0],  # A16 naar A20 maar dan oeps
                  8.4))
    edges.append((vwy[0], A20_locs[4][0],  # A16 naar A20 maar dan oeps
                  8.4))

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    return nx.get_node_attributes(G, 'name'), G


if __name__ == "__main__":
    data = dt.NDW_handler('spacetime/TDA_datasets/TDA_spacetime.csv')
    rips = Rips(maxdim=2)

    window_size = 10

    for i in [0]:
        date_df = data.dfs[30 * i]
        n = len(date_df) - 2 * window_size
        m = len(data.locations)
        if n < 0:
            continue
        w_dists = np.empty((n, 1))

        date_df_speed = date_df[['speed_' + loc for loc in data.locations]].to_numpy().flatten()
        date_df_intensity = date_df[['intensity_' + loc for loc in data.locations]].to_numpy().flatten()
        date_df_flat = np.vstack([date_df_speed, date_df_intensity]).T

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            for j in tqdm(range(n)):
                dgm1 = rips.fit_transform(date_df_flat[m * j:m * (j + window_size), :])
                dgm2 = rips.fit_transform(date_df_flat[m * (j + window_size):m * (j + 2 * window_size), :])
                w_dists[j] = persim.wasserstein(dgm1[0], dgm2[0], matching=False)

        fig, ax = plt.subplots(figsize=(8, 6))
        # ax.set_title(f"{dt.df_date(data.dfs, 30 * i)}")
        ax.plot(np.arange(2 * window_size, len(date_df)), date_df['speed_RWS01_MONIBAS_0201hrr0413ra'].to_numpy()[2*window_size: len(date_df)], label="Speed")
        ax.set_xlim(2*window_size, len(date_df))
        ax.set_xlabel("Minute")
        ax.set_ylabel("Speed (km/h)")
        ax2 = ax.twinx()
        ax2.plot(np.arange(2 * window_size, len(date_df)), w_dists, 'r', label="Wasserstein distance")
        ax2.set_ylabel("Wasserstein distance")
        fig.legend(loc='upper center', bbox_to_anchor=(0.5,0.5,0.5,0.35))
        # plt.savefig(f"spacetime/wassersteinplots/in{window_size}_{dt.df_date(data.dfs, 30 * i)}.png")
