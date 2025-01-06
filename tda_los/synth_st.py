import matplotlib.pyplot as plt
import matplotlib
import networkx as nx
import numpy as np
from gtda.homology import VietorisRipsPersistence
import plotly.io as pio
import gudhi as gd
from tqdm import tqdm
import tda_los.animodule as animodule
from spacetime.local_st_gen import Spacetime
import importlib
importlib.reload(animodule)
from tda_los.animodule import ScatDiagAnim, dgms_from_pointclouds, fast_dgms_from_pointclouds, fast_alpha_dgms_from_pointclouds, DiagAnim

pio.renderers.default = 'browser'
matplotlib.use('Qt5Agg')

ystarts = np.arange(1, 8)

tdalsta = np.array([(0.0, 6.5), (0.11374407582938388, 6.3), (0.22748815165876776, 6.1), (0.34123222748815163, 5.9),
                    (0.4549763033175355, 5.7), (0.5687203791469194, 5.5), (0.6824644549763033, 5.3),
                    (0.7962085308056872, 5.1), (0.909952606635071, 4.9), (1.0236966824644549, 4.7),
                    (1.1374407582938388, 4.5), (1.2511848341232228, 4.3), (1.3649289099526065, 4.1),
                    (1.4786729857819905, 3.9), (1.4786729857819905, 3.9), (1.6325191396281442, 3.6333333333333333),
                    (1.7863652934742982, 3.3666666666666663), (1.940211447320452, 3.0999999999999996),
                    (1.940211447320452, 3.0999999999999996), (2.0480167023436358, 2.9062499999999996),
                    (2.1558219573668196, 2.7124999999999995), (2.263627212390004, 2.51875),
                    (2.3714324674131877, 2.3249999999999997), (2.4792377224363715, 2.1312499999999996),
                    (2.5870429774595554, 1.9374999999999996), (2.694848232482739, 1.7437499999999997),
                    (2.802653487505923, 1.5499999999999996), (2.9104587425291073, 1.3562499999999995),
                    (3.018263997552291, 1.1624999999999996), (3.126069252575475, 0.9687499999999996),
                    (3.233874507598659, 0.7749999999999995), (3.341679762621843, 0.5812499999999994),
                    (3.449485017645027, 0.38749999999999973), (3.5572902726682107, 0.19374999999999964),
                    (3.6650955276913946, -4.440892098500626e-16), (0.0, 0.0), (0.21559385457008204, 0.0),
                    (0.4311877091401641, 0.0), (0.6467815637102461, 0.0), (0.8623754182803282, 0.0),
                    (1.0779692728504102, 0.0), (1.2935631274204922, 0.0), (1.5091569819905741, 0.0),
                    (1.7247508365606563, 0.0), (1.9403446911307383, 0.0), (2.1559385457008204, 0.0),
                    (2.3715324002709024, 0.0), (2.5871262548409844, 0.0), (2.8027201094110663, 0.0),
                    (3.0183139639811483, 0.0), (3.23390781855123, 0.0), (3.4495016731213126, 0.0),
                    (3.6650955276913946, 0.0), (0.0, 0.0), (0.0, 0.38235294117647056), (0.0, 0.7647058823529411),
                    (0.0, 1.147058823529412), (0.0, 1.5294117647058822), (0.0, 1.911764705882353),
                    (0.0, 2.294117647058824), (0.0, 2.676470588235294), (0.0, 3.0588235294117645),
                    (0.0, 3.4411764705882355), (0.0, 3.823529411764706), (0.0, 4.205882352941177),
                    (0.0, 4.588235294117648), (0.0, 4.970588235294118), (0.0, 5.352941176470588),
                    (0.0, 5.735294117647059), (0.0, 6.117647058823529), (0.0, 6.5), (0.0, 0.0),
                    (0.21559385457008204, 0.0), (0.4311877091401641, 0.0), (0.6467815637102461, 0.0),
                    (0.8623754182803282, 0.0), (1.0779692728504102, 0.0), (1.2935631274204922, 0.0),
                    (1.5091569819905741, 0.0), (1.7247508365606563, 0.0), (1.9403446911307383, 0.0),
                    (2.1559385457008204, 0.0), (2.3715324002709024, 0.0), (2.5871262548409844, 0.0),
                    (2.8027201094110663, 0.0), (3.0183139639811483, 0.0), (3.23390781855123, 0.0),
                    (3.4495016731213126, 0.0), (3.6650955276913946, 0.0), (0.0, 0.0), (0.0, 0.38235294117647056),
                    (0.0, 0.7647058823529411), (0.0, 1.147058823529412), (0.0, 1.5294117647058822),
                    (0.0, 1.911764705882353), (0.0, 2.294117647058824), (0.0, 2.676470588235294),
                    (0.0, 3.0588235294117645), (0.0, 3.4411764705882355), (0.0, 3.823529411764706),
                    (0.0, 4.205882352941177), (0.0, 4.588235294117648), (0.0, 4.970588235294118),
                    (0.0, 5.352941176470588), (0.0, 5.735294117647059), (0.0, 6.117647058823529), (0.0, 6.5),
                    (0.0, 3.1), (0.10780525502318392, 2.90625), (0.21561051004636783, 2.7125),
                    (0.3234157650695517, 2.51875), (0.43122102009273566, 2.325), (0.5390262751159196, 2.13125),
                    (0.6468315301391034, 1.9375), (0.7546367851622874, 1.7437500000000001), (0.8624420401854713, 1.55),
                    (0.9702472952086553, 1.35625), (1.0780525502318392, 1.1625), (1.185857805255023, 0.96875),
                    (1.2936630602782069, 0.7749999999999999), (1.401468315301391, 0.5812499999999998),
                    (1.5092735703245748, 0.3875000000000002), (1.6170788253477588, 0.1937500000000001),
                    (1.7248840803709427, 0.0), (0.0, 0.0), (0.21559385457008204, 0.0), (0.4311877091401641, 0.0),
                    (0.6467815637102461, 0.0), (0.8623754182803282, 0.0), (1.0779692728504102, 0.0),
                    (1.2935631274204922, 0.0), (1.5091569819905741, 0.0), (1.7247508365606563, 0.0),
                    (1.9403446911307383, 0.0), (2.1559385457008204, 0.0), (2.3715324002709024, 0.0),
                    (2.5871262548409844, 0.0), (2.8027201094110663, 0.0), (3.0183139639811483, 0.0),
                    (3.23390781855123, 0.0), (3.4495016731213126, 0.0), (3.6650955276913946, 0.0), (0.0, 0.0),
                    (0.0, 0.38235294117647056), (0.0, 0.7647058823529411), (0.0, 1.147058823529412),
                    (0.0, 1.5294117647058822), (0.0, 1.911764705882353), (0.0, 2.294117647058824),
                    (0.0, 2.676470588235294), (0.0, 3.0588235294117645), (0.0, 3.4411764705882355),
                    (0.0, 3.823529411764706), (0.0, 4.205882352941177), (0.0, 4.588235294117648),
                    (0.0, 4.970588235294118), (0.0, 5.352941176470588), (0.0, 5.735294117647059),
                    (0.0, 6.117647058823529), (0.0, 6.5), (0.0, 3.9000000000000004),
                    (0.15384615384615385, 3.6333333333333337), (0.3076923076923077, 3.366666666666667),
                    (0.46153846153846156, 3.1000000000000005), (0.46153846153846156, 3.1000000000000005),
                    (0.5693437165616455, 2.9062500000000004), (0.6771489715848293, 2.7125000000000004),
                    (0.7849542266080133, 2.5187500000000007), (0.8927594816311972, 2.3250000000000006),
                    (1.0005647366543813, 2.1312500000000005), (1.1083699916775651, 1.9375000000000004),
                    (1.216175246700749, 1.7437500000000006), (1.3239805017239328, 1.5500000000000005),
                    (1.4317857567471168, 1.3562500000000004), (1.5395910117703009, 1.1625000000000005),
                    (1.6473962667934847, 0.9687500000000004), (1.7552015218166686, 0.7750000000000004),
                    (1.8630067768398524, 0.5812500000000003), (1.9708120318630362, 0.3875000000000006),
                    (2.0786172868862205, 0.19375000000000053), (2.1864225419094043, 4.440892098500626e-16), (0.0, 0.0),
                    (0.21559385457008204, 0.0), (0.4311877091401641, 0.0), (0.6467815637102461, 0.0),
                    (0.8623754182803282, 0.0), (1.0779692728504102, 0.0), (1.2935631274204922, 0.0),
                    (1.5091569819905741, 0.0), (1.7247508365606563, 0.0), (1.9403446911307383, 0.0),
                    (2.1559385457008204, 0.0), (2.3715324002709024, 0.0), (2.5871262548409844, 0.0),
                    (2.8027201094110663, 0.0), (3.0183139639811483, 0.0), (3.23390781855123, 0.0),
                    (3.4495016731213126, 0.0), (3.6650955276913946, 0.0), (0.0, 0.0), (0.0, 0.38235294117647056),
                    (0.0, 0.7647058823529411), (0.0, 1.147058823529412), (0.0, 1.5294117647058822),
                    (0.0, 1.911764705882353), (0.0, 2.294117647058824), (0.0, 2.676470588235294),
                    (0.0, 3.0588235294117645), (0.0, 3.4411764705882355), (0.0, 3.823529411764706),
                    (0.0, 4.205882352941177), (0.0, 4.588235294117648), (0.0, 4.970588235294118),
                    (0.0, 5.352941176470588), (0.0, 5.735294117647059), (0.0, 6.117647058823529), (0.0, 6.5)])


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
            interp_dr = scale / (data_df.iloc[t_index]['intensity_' + to] + 1e-1)
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

g = nx.DiGraph()
n = list(range(9))
g.add_nodes_from(n)
g.add_weighted_edges_from([(i, i + 1, 1) for i in n[2:-1]])
g.add_weighted_edges_from([(0,1,1),(1,4,1)])
st = Spacetime(g, np.repeat(1, len(n)), np.repeat(1, len(n)), [len(n)-1], len(n), factor=1)
lst = st.local_spacetime_alt([1, 1 / 4, 1, 1, 1, 1, 1, 1, 1], len(n)-1, interp=.1, mode='prev')
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
        merge = min(path1.index(merge), path1.index(merged_path[0]))
    reduced_paths.append((start, path[merge], path1[:merge+1], nx.path_weight(g, path1[merge:], weight='weight')))
reduced_paths
xy = np.array([i for j in lst.values() for i in j])
xy = np.append(xy, np.linspace((0, 0), (np.max(xy[:, 0]), 0), int(np.max(xy[:, 0]) / .1)), axis=0)
xy = np.append(xy, np.linspace((0, 0), (0, np.max(xy[:, 1])), int(np.max(xy[:, 1]) / .1)), axis=0)
xy = np.unique(xy, axis=0)
plt.scatter(xy[:, 0], xy[:, 1], s=1)

def merge(interp, speeds, plot=True):
    lines, lines_pre = get_lines(0, paths[0], [speeds[i] for i in paths[0]], g, interp=interp)
    lines = np.hstack((lines, np.zeros(lines.shape[0])[:, np.newaxis]))
    lines2, lines2_pre = get_lines(0, paths[2], [speeds[i] for i in paths[2]], g, interp=interp)
    lines2 = np.hstack((lines2, np.zeros(lines2.shape[0])[:, np.newaxis]))[:, [0, 2, 1]]
    merge_distance = nx.shortest_path_length(g, reduced_paths[1][1], target, weight='weight')
    lines2_pre = [line + np.array([0, -merge_distance]) for line in lines2_pre]
    lines2_pre = [line[line[:, 1] >= 0] for line in lines2_pre]
    lines2_pre = [line for line in lines2_pre if len(line) > 0]
    lines2_pre = [np.hstack((line, np.zeros(line.shape[0])[:, np.newaxis]))[:, [0, 2, 1]] for line in lines2_pre]
    lines_pre2 = np.concatenate(lines_pre, axis=0)
    lines_pre2 = lines_pre2[lines_pre2[:, 1] > merge_distance]
    lines_starts = np.unique(lines_pre2[lines_pre2[:, 0] == 0, 1])
    shift_indices = [(lambda x: x[len(x)//2] if len(x) > 0 else None)
                     (np.argwhere(np.abs(line[:, 1] - merge_distance) < interp/2).flatten()) for line in lines_pre]
    shift_d1s = [lines_pre[i][j][0] for i, j in enumerate(shift_indices) if j is not None]
    attached_lines = [lines2_pre[i] + np.array([shift_d1s[i] - np.max(lines2_pre[i][:, 0]), 4, 0])
                      for i in range(len(shift_d1s))]
    attached_lines_starts = [line[0, :] for line in attached_lines]

    def num(v1, v2, interp=interp):
        return int(np.sqrt(np.sum(v1**2+v2**2))/interp)

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


obj = merge(.05, [1, 1/5, 1, 1, 1, 1, 1, 1, 1])


def slow(loc, speed, interp=.1, plot=True):
    speeds = 9*[1]
    speeds[loc] = speed
    return merge(interp, speeds, plot=plot)


obj = slow(2, 1/4, plot=False)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(obj[:, 0], obj[:, 1], obj[:, 2], s=1)

locs = n
minspeeds = len(locs) * [1 / 5]
# for loc, minspeed in zip(locs, minspeeds):
#     create_and_save_anim(loc, minspeed, n, name='samen_na_2')

shiftxs = np.linspace(1, 1/5, 50)
# clouds = [gen_lsta(locs[1], shiftx, n, interp=.2) for shiftx in shiftxs]
clouds3 = [slow(1, shiftx, interp=.2, plot=False) for shiftx in shiftxs]
oeps = [slow(3, shiftx, interp=.2, plot=False) for shiftx in shiftxs]
# dgms = fast_dgms_from_pointclouds(clouds, max_dimension=3, n_cpus=2)
dgms2 = fast_dgms_from_pointclouds(clouds3, max_dimension=3, n_cpus=2, sparse=.2)
oepsdgms = fast_dgms_from_pointclouds(oeps, max_dimension=3, n_cpus=2, sparse=.2)
# dim 3 & 4 met sparse .1 & .2 leveren niks op :(
# d3gms = fast_alpha_dgms_from_pointclouds(clouds3, n_cpus=2)
# anim = ScatDiagAnim(clouds, dgms, s=1)
# anim2 = ScatDiagAnim(clouds, dgms2, s=1)
anim3 = DiagAnim(dgms2)
animtest = animodule.ScatDiagAnim3D(clouds3, dgms2, s=1)
animtest.anim.save('tda_los/filmpjes/3d.gif',fps=15)
animtest2 = animodule.ScatDiagAnim3D(oeps, oepsdgms)
animtest2.anim.save('tda_los/filmpjes/3d_other.gif',fps=15)

d2_features = [[x for d, x in dgm if d == 1] for dgm in dgms2]
d2_lifetimes = [[death - birth for birth, death in x] for x in d2_features]
plt.plot([25 * (sum([y for y in x if y > .1])) for x in d2_lifetimes], 'm')
