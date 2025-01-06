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

ystarts = np.arange(1, 8)


def gen_xy(shiftx, shiftlines):
    x = np.array([])
    y = np.array([])
    closed = True
    for y0 in ystarts:
        if y0 in shiftlines and shiftx is not None:
            tshiftx = shiftx - shiftlines.index(y0)
            n2 = 10 * (y0 - 1)
            x = np.append(x, np.append(np.linspace(0, 2 + tshiftx, 10), np.linspace(2 + tshiftx + 1 / n2, y0, n2)))
        else:
            x = np.append(x, np.linspace(0, y0, y0 * 10))
        y = np.append(y, np.linspace(y0, 0, y0 * 10))
    if closed:
        x = np.append(x, np.append(np.repeat(0, 10 * max(ystarts)), np.linspace(0, max(ystarts), 10 * max(ystarts))))
        y = np.append(y, np.append(np.linspace(0, max(ystarts), 10 * max(ystarts)), np.repeat(0, 10 * max(ystarts))))
    return np.vstack([x, y]).T


g = nx.Graph()
n = list(range(7))
g.add_nodes_from(n)
g.add_weighted_edges_from([(i, i + 1, 1) for i in n[:-1]])
st = Spacetime(g, np.repeat(1, len(n)), np.repeat(1, len(n)), [6], 7, factor=1)
lst = st.local_spacetime_alt([1, 1 / 4, 1, 1, 1, 1, 1], 6, interp=.1, mode='prev')
xy = np.array([i for j in lst.values() for i in j])
xy = np.append(xy, np.linspace((0, 0), (np.max(xy[:, 0]), 0), int(np.max(xy[:, 0]) / .1)), axis=0)
xy = np.append(xy, np.linspace((0, 0), (0, np.max(xy[:, 1])), int(np.max(xy[:, 1]) / .1)), axis=0)
xy = np.unique(xy, axis=0)
plt.scatter(xy[:, 0], xy[:, 1], s=1)


def gen_lsta(location, speed, nodes, interp=.1):
    target = len(nodes)-1
    speeds = np.ones(len(nodes))
    speeds[location] = speed
    lst = st.local_spacetime_alt(speeds, target, interp=interp, mode='prev')
    xy = np.array([i for j in lst.values() for i in j])
    xy = np.append(xy, np.linspace((0, 0), (np.max(xy[:, 0]), 0), int(np.max(xy[:, 0]) / .1)), axis=0)
    xy = np.append(xy, np.linspace((0, 0), (0, np.max(xy[:, 1])), int(np.max(xy[:, 1]) / .1)), axis=0)
    xy = np.unique(xy, axis=0)
    return xy


gxy = gen_xy(2, [6, 5])


# plt.scatter(gxy[:, 0], gxy[:, 1], s=1)
# plt.xlabel("Time")
# plt.ylabel("Distance")
# persistence = VietorisRipsPersistence(homology_dimensions=[0, 1], n_jobs=6)
# rc = gd.RipsComplex(points=gxy)
# st = rc.create_simplex_tree(max_dimension=2)
# st.compute_persistence()
# dgm = st.persistence_intervals_in_dimension(1)
# gd.plot_persistence_diagram(dgm)

# TODO: filmpjes analyseren voor verschillende locs
# TODO: analyseren met vertakkingen


def create_and_save_anim(loc, minspeed, nodes, base_path='tda_los/filmpjes/', name='recht', close=True):
    shiftxs = np.linspace(1, minspeed, 100)
    clouds = [gen_lsta(loc, shiftx, nodes) for shiftx in shiftxs]
    dgms = fast_dgms_from_pointclouds(clouds, n_cpus=12)
    anim = ScatDiagAnim(clouds, dgms, s=1)
    anim.anim.save(base_path+name+f"_{loc}_{minspeed}.gif", writer='pillow', fps=25)
    if close:
        plt.close('all')

locs = n
minspeeds = len(locs) * [1 / 5]
# for loc, minspeed in zip(locs, minspeeds):
#     create_and_save_anim(loc, minspeed, n)


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
         'RWS01_MONIBAS_0131hrr0064ra', 'RWS01_MONIBAS_0131hrr0067ra']
edges = [(nodes[0], nodes[1], .5), (nodes[1], nodes[2], 0.3), (nodes[2], nodes[3], 1.2), (nodes[3], nodes[4], 1.5),
         (nodes[4], nodes[5], 0.5), (nodes[5], nodes[6], 0.6), (nodes[2], nodes[7], 3), (nodes[7], nodes[8], .3)]
g = nx.DiGraph()
g.add_nodes_from(nodes)
g.add_weighted_edges_from(edges)
g = g.reverse()
st = Spacetime(g, {node: 1 for node in nodes}, {node: 1 for node in nodes}, [n for n in nodes[:-1]], len(nodes), factor=1)
lst = st.local_spacetime_alt({node: 1 for node in nodes}, nodes[0], interp=.1, mode='prev')
starts = [node for node, degree in g.in_degree if degree == 0]
mergers = [node for node, degree in g.in_degree if degree > 1]
target = [node for node, degree in g.out_degree if degree == 0][0]
paths = {s: nx.shortest_path(g, source=s, target=target) for s in starts}
reduced_paths = []
for start, path in paths.items():
    path1 = path
    merge = len(path)-1
    for _, _, path2, _ in reduced_paths:
        merged_path = [i for i in path1 if i in path2]
        merge = min(merge, path1.index(merged_path[0]))
    reduced_paths.append((start, path[merge], path1[:merge+1], nx.path_weight(g, path1[merge:], weight='weight')))
reduced_paths
xy = np.array([i for j in lst.values() for i in j])
xy = np.append(xy, np.linspace((0, 0), (np.max(xy[:, 0]), 0), int(np.max(xy[:, 0]) / .1)), axis=0)
xy = np.append(xy, np.linspace((0, 0), (0, np.max(xy[:, 1])), int(np.max(xy[:, 1]) / .1)), axis=0)
xy = np.unique(xy, axis=0)
plt.scatter(xy[:, 0], xy[:, 1], s=1)

def merge(interp, speeds, plot=True, scale=200):
    lines, lines_pre = get_lines(0, paths[starts[0]], [speeds[i] for i in paths[starts[0]]], g, interp=interp,
                                 scale=scale)
    lines = np.hstack((lines, np.zeros(lines.shape[0])[:, np.newaxis]))
    lines2, lines2_pre = get_lines(0, paths[starts[1]], [speeds[i] for i in paths[starts[1]]], g, interp=interp,
                                   scale=scale)
    merge_distance = nx.shortest_path_length(g, reduced_paths[1][1], target, weight='weight')
    lines2_pre = [line + np.array([0, -merge_distance]) for line in lines2_pre]
    lines2_pre = [line[line[:, 1] >= 0] for line in lines2_pre]
    lines2_pre = [line for line in lines2_pre if len(line) > 0]
    lines2_pre = [np.hstack((line, np.zeros(line.shape[0])[:, np.newaxis]))[:, [0, 2, 1]] for line in lines2_pre]
    lines_pre2 = np.concatenate(lines_pre, axis=0)
    lines_pre2 = lines_pre2[lines_pre2[:, 1] > merge_distance]
    lines_starts = np.unique(lines_pre2[lines_pre2[:, 0] == 0, 1])
    if isinstance(interp, dict):
        interp_min = max(0.05, min(scale/max(interp.values()), 1))
        interp_min = scale/max(interp.values())
    else:
        interp_min = interp
    shift_indices = [(lambda x: x[len(x)//2] if len(x) > 0 else None)
                     (np.argwhere(np.abs(line[:, 0] - merge_distance) < interp_min/2).flatten()) for line in lines_pre]
    shift_d1s = [lines_pre[i][j][1] for i, j in enumerate(shift_indices) if j is not None]
    attached_lines = [lines2_pre[i] + np.array([shift_d1s[i] - np.max(lines2_pre[i][:, 2]), merge_distance, 0])
                      for i in range(min(len(shift_d1s), len(lines2_pre)))]
    if len(attached_lines)>0:
        attached_lines_starts = [line[0, :] for line in attached_lines]

        def num(v1, v2):
            return int(np.sqrt(np.sum(v1**2+v2**2))/interp_min)

        attached_lines += [np.linspace(ats1, ats2, num(ats1, ats2)) for ats1, ats2 in zip(attached_lines_starts[:-1], attached_lines_starts[1:])]
        attached_lines = np.concatenate(attached_lines)
        links = [np.linspace(np.array([0,ls,0]), ats, num(np.array([0,ls,0]), ats)) for ls, ats in zip(lines_starts[::-1], attached_lines_starts)]
        link = np.concatenate(links)

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.set_aspect('equal')
            ax.scatter(lines[:, 0], lines[:, 1], lines[:, 2], s=1)
            ax.scatter(attached_lines[:,0], attached_lines[:, 1], attached_lines[:, 2], color='r', s=1)
            ax.scatter(link[:, 0], link[:, 1], link[:, 2], color='m', s=1)

        return np.concatenate([lines, attached_lines, link])
    else:
        return lines


one_speeds = {node: 1 for node in nodes}
slow3 = one_speeds.copy()
slow3[nodes[4]] = 1/5
obj = merge(.05, one_speeds)


def slow(loc, speed, interp=.1, plot=True):
    speeds = 9*[1]
    speeds[loc] = speed
    return merge(interp, speeds, plot=plot)


# obj = slow(2, 1/4, plot=False)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(obj[:, 0], obj[:, 1], obj[:, 2], s=1)

# locs = n
# minspeeds = len(locs) * [1 / 5]
# for loc, minspeed in zip(locs, minspeeds):
#     create_and_save_anim(loc, minspeed, n, name='samen_na_2')

data = dt.NDW_handler('tda_los/filmpjes/terugslag_merge_10sep.csv')
speeds = [{node: r['speed_'+node]/100 for node in nodes[1:]}|{nodes[0]: 1} for _, r in data.dfs[0].iterrows()]
intensities = [(d:={node: r['intensity_'+node] for node in nodes[1:]}) | {nodes[0]: max(d.values())} for _, r in data.dfs[0].iterrows()]
shiftxs = np.linspace(1, 1/5, 200)
clouds = [merge(.2, speed, plot=False) for speed in speeds]
clouds = [merge(intensity, speed, plot=False) for intensity, speed in zip(intensities, speeds)]
dgms = fast_dgms_from_pointclouds(clouds, max_dimension=2, n_cpus=10, sparse=.2)
# dgms2 = fast_dgms_from_pointclouds(clouds3, max_dimension=3, n_cpus=2, sparse=.2)
# dim 3 & 4 met sparse .1 & .2 leveren niks op :(
# d3gms = fast_alpha_dgms_from_pointclouds(clouds3, n_cpus=2)
anim = animodule.ScatDiagAnim3D(clouds, dgms)
# anim2 = ScatDiagAnim(clouds, dgms2, s=1)
# anim3 = DiagAnim(dgms2)
# animtest = animodule.ScatDiagAnim3D(clouds3, dgms2, s=1)
# animtest2 = animodule.ScatterAnim3D(clouds3)

d2_features = [[x for d, x in dgm if d == 1] for dgm in dgms]
d2_lifetimes = [[death - birth for birth, death in x] for x in d2_features]
plt.figure()
start, end = 840, 900
[plt.plot(pd.to_datetime(data.dfs[0].index[start:end]), data.dfs[0]['speed_'+node].to_numpy()[start:end], label=f'{node}')
 for node in list(paths.values())[1][:-1]]
plt.legend()
plt.figure()
plt.plot(pd.to_datetime(data.dfs[0].index), data.dfs[0]['speed_'+nodes[1]].to_numpy())
plt.plot(pd.to_datetime(data.dfs[0].index), [5 * (sum([y for y in x])) for x in d2_lifetimes], 'm')
