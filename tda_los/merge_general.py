import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import numpy as np
import plotly.io as pio
import tda_los.animodule as animodule
import snelweg.datatools2 as dt
from spacetime.local_st_gen import Spacetime
import pandas as pd
import importlib
import seaborn as sns
importlib.reload(animodule)
from tda_los.animodule import ScatDiagAnim, fast_dgms_from_pointclouds

pio.renderers.default = 'browser'
matplotlib.use('Qt5Agg')

def get_line(t_index, rpath, speeds, graph, interp=.05, scale=200, gain=1):
    path = rpath
    d_max = nx.path_weight(graph, path, weight='weight')
    xy = np.array([[0, d_max]])
    for i in range(len(path)-1):
        dd = g[path[i]][path[i+1]]['weight']
        speed = speeds[i+1]
        speed = gain * (speed - 100) + 100
        delta = np.array([dd / speed, -dd])
        dr = np.sqrt(np.sum(delta ** 2))
        if interp == 'intensity':
            interp_dr = scale / (data_df.iloc[t_index]['intensity_' + path[i+1]] + 1e-1)
            interp_dr = max(0.05, min(interp_dr, 1))
            xy = np.append(xy, np.linspace(xy[-1, :], xy[-1, :] + delta, int(dr / interp_dr)), axis=0)
        elif isinstance(interp, dict):
            interp_dr = scale / (interp[path[i+1]] + 1e-1)
            interp_dr = max(0.05, min(interp_dr, 1))
            xy = np.append(xy, np.linspace(xy[-1, :], xy[-1, :] + delta, int(dr / interp_dr)), axis=0)
        else:
            xy = np.append(xy, np.linspace(xy[-1, :], xy[-1, :] + delta, int(dr / interp)), axis=0)
    return xy


def integer_floor(x, integer=2):
    return np.floor(x * integer) / integer


def shift_line(line, offset=0, resolution=2):
    where = np.argmin(
        np.abs(line[:, 1] - (np.max(integer_floor(line[:, 1], integer=resolution)) - offset / resolution)))
    dt = line[where, 0]
    return line - np.array([dt, 0])


def get_lines(t_index, rpath, speeds, graph, interp=.05, resolution=2, scale=200, gain=1):
    maxline = get_line(t_index, rpath, speeds, graph, interp, scale=scale, gain=gain)
    max_d = np.max(integer_floor(maxline[:, 1], integer=resolution))
    if isinstance(interp, dict):
        interp_min = scale/max(interp.values())
    else:
        interp_min = interp
    pres = [shift_line(maxline, offset=i, resolution=resolution) for i in range(int(max_d * resolution))]
    pres = [line[np.all(line >= 0, axis=1), :] for line in pres]
    xy = np.concatenate(pres, axis=0)
    xy = xy[np.all(xy >= 0, axis=1), :]
    maxes = np.max(xy, axis=0)
    xy_prelink = xy.copy()
    xy = np.append(xy, np.linspace(np.array([0, 0]), np.array([0, maxes[1]]), int(maxes[1] / interp_min)), axis=0)
    xy = np.append(xy, np.linspace(np.array([0, 0]), np.array([maxes[0], 0]), int(maxes[0] / interp_min)), axis=0)
    xy = np.unique(xy, axis=0)
    return xy, pres


nodes = ['fake_0458ra', 'RWS01_MONIBAS_0041hrr0463ra', 'RWS01_MONIBAS_0041hrr0466ra', 'RWS01_MONIBAS_0041hrr0478ra',
         'RWS01_MONIBAS_0041hrr0493ra', 'RWS01_MONIBAS_0041hrr0498ra', 'RWS01_MONIBAS_0041hrr0504ra',
         'RWS01_MONIBAS_0131hrr0064ra', 'RWS01_MONIBAS_0131hrr0067ra', 'RWS01_MONIBAS_0131hrr0072ra']
edges = [(nodes[0], nodes[1], .5), (nodes[1], nodes[2], 0.3), (nodes[2], nodes[3], 1.2), (nodes[3], nodes[4], 1.5),
         (nodes[4], nodes[5], 0.5), (nodes[5], nodes[6], 0.6), (nodes[3], nodes[7], 1), (nodes[7], nodes[8], .3),
         (nodes[8], nodes[9], .6)]
g = nx.DiGraph()
g.add_nodes_from(nodes)
g.add_weighted_edges_from(edges)
g = g.reverse()
starts = [node for node, degree in g.in_degree if degree == 0]
mergers = [node for node, degree in g.in_degree if degree > 1]
target = [node for node, degree in g.out_degree if degree == 0][0]
# paths = {s: nx.shortest_path(g, source=s, target=target) for s in starts}
paths = [nx.shortest_path(g, source=s, target=target) for s in starts]

def merge_point(path1, path_list):
    for i in path1:
        em = [i in p for p in path_list]
        if max(em):
            return em.index(True), i


def merge(interp, speeds, paths, plot=True, scale=200):
    paths_lines = []
    paths = sorted(paths, key=lambda x: nx.path_weight(g, x, weight='weight'), reverse=True)  # sort by descending sensor distance to target
    init_lines = get_lines(0, paths[0], [speeds[i] for i in paths[0]], g, interp=interp, scale=scale)[1]
    init_lines = [np.pad(e, [(0,0),(0,len(paths)-1)], mode='constant') for e in init_lines]
    line_starts = [e[0, :] for e in init_lines] + [np.zeros(len(paths)+1)]
    line_ends = [e[-1, :] for e in init_lines] + [np.zeros(len(paths)+1)]
    if isinstance(interp, dict):
        min_interp = scale/max(interp.values())
    elif isinstance(interp, float):
        min_interp = interp
    print(line_starts)
    paths_lines.append(init_lines)
    sensor_locs = {node: np.append(np.array([nx.shortest_path_length(g, target=target, source=node, weight='weight')]),
                                   np.zeros(len(paths)-1)) for node in g.nodes}
    paths_used = [paths[0]]
    interps = [np.linspace(e1, e2, int(np.sqrt(np.sum((e1-e2)**2))/min_interp)) for e1, e2 in zip(line_starts[:-1], line_starts[1:])]
    interps += [np.linspace(e1, e2, int(np.sqrt(np.sum((e1-e2)**2))/min_interp)) for e1, e2 in zip(line_ends[:-1], line_ends[1:])]
    for i, path in enumerate(paths[1:]):
        path_lines = get_lines(0, path, [speeds[i] for i in path], g, interp=interp, scale=scale)[1]
        li, m = merge_point(path, paths_used)
        print(f"Line {i+1} of sorted paths merges with line {li} at node {m}")
        for sensor in path[:path.index(m)]:
            sensor_locs[sensor][0] -= np.sum(sensor_locs[m])
            sensor_locs[sensor][[0, i+1]] = sensor_locs[sensor][[i+1, 0]]
            sensor_locs[sensor] += sensor_locs[m]
        for j, line in enumerate(path_lines):
            line[:, 1] -= np.sum(sensor_locs[m])
            shift_j = len(paths_lines[li])-len(path_lines)
            z = np.zeros((line.shape[0], len(paths)+1))
            z[:, [0, i+2]] = line
            z += np.append(np.array([0]), sensor_locs[m])
            z = z[np.all(z[:, 1:] >= 0, axis=1), :]
            attaching_line = paths_lines[li][j+shift_j]
            attaching_index = np.argmin(np.sum((attaching_line[:, 1:]-sensor_locs[m])**2, axis=1))
            attaching_time = attaching_line[attaching_index, 0]
            if z.shape[0] > 0:
                # print(attaching_time - z[-1, 0])
                z[:, 0] += attaching_time - z[-1, 0]
            path_lines[j] = z
        paths_used.append(path)
        paths_lines.append(path_lines)
        lap = paths_lines[li][len(paths_lines)+shift_j]
        lt = lap[np.argmin(np.sum((lap[:, 1:]-sensor_locs[m])**2, axis=1)), 0]
        path_lines = [x for x in path_lines if x.shape[0] > 0]
        line_ends = [e[0, :] for e in path_lines] + [np.append(0, sensor_locs[m])]
        interps += [np.linspace(e1, e2, int(np.sqrt(np.sum((e1-e2)**2))/min_interp)) for e1, e2 in zip(line_ends[:-1], line_ends[1:])]
        ie1 = path_lines[-1][0, :]
        ie2 = np.append(0, sensor_locs[m])
        # interps += [np.linspace(ie1, ie2, int(np.sqrt(np.sum((ie1-ie2)**2))/min_interp))]
    return paths_lines, interps

test_get_lines = get_lines(0, paths[0], len(g.nodes)*[1], g)[1]
test_speeds = {i: 1.0 for i in g.nodes}
test_speeds['RWS01_MONIBAS_0041hrr0478ra'] = 1/3
test_merge, apps = merge(0.1, test_speeds, paths)
tm = np.concatenate([np.concatenate(x) for x in test_merge]+apps)
plt.figure()
# ax = plt.subplot()
# ax.plot(tm[:, 0], tm[:, 1], linestyle="", marker=".")
ax = plt.subplot(projection='3d')
ax.plot(tm[:, 0], tm[:, 1], tm[:, 2], linestyle="", marker=".")
