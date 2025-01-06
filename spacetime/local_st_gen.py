import pickle

import pandas as pd

import snelweg.datatools2 as dt
import networkx as nx
import numpy as np
from ripser import Rips
import persim
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from spacetime.TDA_trial import specific_graph, specific_graph_old
import gudhi as gd
from gudhi.hera import wasserstein_distance as wd
from gudhi.representations.vector_methods import BettiCurve
from gudhi.representations import Landscape

matplotlib.use('Qt5Agg')
plt.rcParams.update({'text.usetex': False})


def speedplots(data, t, x="num"):
    assert x == "num" or x == "date"
    date_df = data.dfs[t]
    labelbase = "speed_RWS01_MONIBAS_0201hrr0"
    where = ["413", "444", "452", "478"]
    plt.figure()
    plt.title(dt.df_date(data.dfs, t))
    if x == "num":
        for n in where:
            plt.plot(date_df[labelbase+n+"ra"].to_numpy(), label=n)
    else:
        for n in where:
            plt.plot(date_df[labelbase+n+"ra"], label=n)
    plt.legend()


class Spacetime:
    def __init__(self, road_graph, all_speeds, max_speeds, targets, horizon, atol=1, factor=60):
        self.graph = road_graph
        self.all_speeds = all_speeds
        self.targets = targets
        self.horizon = horizon
        self.num_nodes = len(road_graph.nodes)
        if hasattr(max_speeds, "__getitem__"):
            self.max_speeds = max_speeds
        else:
            self.max_speeds = np.repeat(max_speeds, self.num_nodes)
        self.factor = factor
        best_case_time = self.graph_forward_info_speed(road_graph, self.max_speeds)
        self.best_case_subgraphs = {
            target: [source for source, time in
                     dict(nx.single_target_shortest_path_length(best_case_time, target)).items() if
                     time < horizon + atol] for target in targets}

    def local_spacetime(self, speeds, target, additional_features=None, tf=1):
        subgraph = nx.subgraph(self.graph, self.best_case_subgraphs[target])
        distance_dict = dict(nx.shortest_path_length(subgraph, target=target, weight='weight'))
        time_dicts = [dict(
            nx.shortest_path_length(self.graph_forward_info_speed(subgraph, speeds[i]), target=target, weight='weight'))
                      for i in range(len(speeds))]
        return [[(time_dict[i] + (j - len(speeds) + 1) * tf, distance_dict[i], speeds[j][i]) for i in distance_dict]
                for j, time_dict in enumerate(time_dicts)]

    def comp_lst(self, speeds, target, tf=1, interp=None):
        subgraph = nx.subgraph(self.graph, self.best_case_subgraphs[target])
        timegraph = self.graph_forward_info_speed(subgraph, speeds)
        points = []
        for node in subgraph.nodes:
            sp = nx.shortest_path(timegraph, source=node, target=target, weight='weight')
            dist = -sum([subgraph[sp[i]][sp[i + 1]]['weight'] for i in range(len(sp) - 1)])
            time = -sum([timegraph[sp[i]][sp[i + 1]]['weight'] for i in range(len(sp) - 1)])
            points.append((time, dist))
            for outbound, inbound in zip(sp[:-1], sp[1:]):
                dist += subgraph[outbound][inbound]['weight']
                time += timegraph[outbound][inbound]['weight']
                points.append((time, dist))
        if interp is not None:
            output = {0: []}
            i = 0
            for prev, cur in zip(points[:-1], points[1:]):
                if cur[0] < 1e-3:
                    i += 1
                    output[i] = []
                else:
                    dist = np.sqrt(np.sum((np.array(prev) - np.array(cur)) ** 2))
                    num = int(dist / interp)
                    fill = np.linspace(prev, cur, num)
                    output[i] += [tuple(t) for t in fill]
            return output
        else:
            return points

    def local_spacetime_alt(self, speeds, target, additional_features=None, rescaling=200, tf=1, interp=None, mode='mean'):
        subgraph = nx.subgraph(self.graph, self.best_case_subgraphs[target])
        timegraph = self.graph_forward_info_speed(subgraph, speeds, mode=mode)
        points = []
        for node in subgraph.nodes:
            sp = nx.shortest_path(timegraph, source=node, target=target, weight='weight')
            dist = sum([subgraph[sp[i]][sp[i + 1]]['weight'] for i in range(len(sp) - 1)])
            time = 0
            if additional_features is not None:
                points.append((time, dist, additional_features[node] / rescaling))
                for outbound, inbound in zip(sp[:-1], sp[1:]):
                    dist -= subgraph[outbound][inbound]['weight']
                    time += timegraph[outbound][inbound]['weight']
                    points.append((time, dist, additional_features[inbound] / rescaling))
            else:
                points.append((time, dist))
                for outbound, inbound in zip(sp[:-1], sp[1:]):
                    dist -= subgraph[outbound][inbound]['weight']
                    time += timegraph[outbound][inbound]['weight']
                    points.append((time, dist))
        if interp is not None:
            output = {0: []}
            i = 0
            for prev, cur in zip(points[:-1], points[1:]):
                if cur[0] < 1e-3:
                    i += 1
                    output[i] = []
                else:
                    dist = np.sqrt(np.sum((np.array(prev) - np.array(cur)) ** 2))
                    num = int(dist / interp)
                    fill = np.linspace(prev, cur, num)
                    output[i] += [tuple(t) for t in fill]
            return output
        else:
            return points

    def plot_local_spacetime(self, maxspeed, speeds, target, additional_features=None, tf=1):
        lsts = self.local_spacetime(speeds, target, additional_features, tf)
        x_scats = [[x[0] for x in localst] for localst in lsts]
        y_scats = [[y[1] for y in localst] for localst in lsts]
        fig, ax = plt.subplots()
        ax.axvline(x=self.horizon, color='k', alpha=.5, lw=.5)
        ax.axhline(y=maxspeed / (self.factor / self.horizon), color='k', alpha=.5, lw=.5)
        [ax.axline((-len(speeds) + 1 + i, 0), slope=maxspeed / (self.factor), linestyle='-', color='k', alpha=.5, lw=.5)
         for i in range(len(speeds))]
        ax.axline((self.horizon, 0), slope=-maxspeed / self.factor, color='k', alpha=.5, lw=.5)
        [ax.scatter(x_scats[i], y_scats[i]) for i in range(len(x_scats))]
        # ax.plot(np.linspace(0,15,2), 100/60*np.linspace(0,15,2),'k--',label="100 km/h")
        ax.set_xlabel('Time')
        ax.set_ylabel('Distance')
        ax.set_aspect('equal', 'box')

    def graph_forward_info_speed(self, subgraph, speeds, mode='mean'):
        """
        :param subgraph: (sub)graph on which to calculate
        :param speeds: measured speeds
        :return: complete (directed) graph with time distances
        """
        time_graph = subgraph.copy()
        for out, to, data in time_graph.edges(data=True):
            if mode == 'mean':
                data['weight'] = self.factor * 2 * data['weight'] / (speeds[out] + speeds[to])
            elif mode == 'prev':
                data['weight'] = self.factor * data['weight'] / speeds[out]
            else:
                raise ValueError("Mode not supported, options are 'mean' or 'prev'")
        return time_graph


if __name__ == "__main__":
    # data = dt.NDW_handler('spacetime/TDA_datasets/TDA_spacetime.csv')
    try:
        print(data)
    except Exception as e:
        data = dt.NDW_handler('snelweg/ochtend_alles/A20Gouda_alles_ochtend.csv')
    # date_index = 120
    # 120 forward: file tussen i=[53,216], dip bij i=[235, 240]
    # 98 backward: file vanaf [123, 170]
    # date_df = data.dfs[date_index]
    # assert dt.df_date(data.dfs, date_index) == '2023-05-11'
    # target = "RWS01_MONIBAS_0201hrr0413ra"
    date_index = 120
    interpolation = 1
    forward = True
    date_df = data.dfs[date_index]
    target = "RWS01_MONIBAS_0201hrr0413ra"
    labels, G = specific_graph_old(data)
    Grev = G.reverse(copy=True)
    target_index = [i for i in labels if labels[i] == target][0]
    max_speed = 100
    window_size = 2
    horizon = 10
    ward = 'Forward' if forward else 'Backward'
    if forward:
        st = Spacetime(G, [0], max_speed, [target_index], horizon)
    else:
        st = Spacetime(Grev, [0], max_speed, [target_index], horizon)
    test_speeds = [{label: data.dfs[0]['speed_' + labels[label]].to_numpy()[i] for label in labels} for i in
                   range(window_size)]
    test_speeds_jam = [{label: data.dfs[0]['speed_' + labels[label]].to_numpy()[60 + i] for label in labels} for i in
                       range(window_size)]


    def get_lst(t_index):
        test_speeds = [{label: date_df['speed_' + labels[label]].to_numpy()[i + t_index] for label in labels} for i
                       in range(window_size)]
        lsts = st.local_spacetime(test_speeds, target_index)
        return [(x[0], x[1]) for localst in lsts for x in localst]


    def get_lsta(t_index, interp=None):
        test_speeds = {label: date_df['speed_' + labels[label]].to_numpy()[t_index] for label in labels}
        # test_intensities = {label: data.dfs[0]['intensity_'+labels[label]].to_numpy()[t_index] for label in labels}
        # lsta = st.local_spacetime_alt(test_speeds, target_index, test_intensities, interp=interp)
        # return [x[0] for x in lsta], [y[1] for y in lsta], [z[2] for z in lsta]
        lsta = st.local_spacetime_alt(test_speeds, target_index, interp=interp)
        output = dict()
        for key in lsta:
            output[key] = ([x[0] for x in lsta[key]], [y[1] for y in lsta[key]])
        return output


    def tda_lsta(t_index, interp=None, intense=True, df=date_df, rescaling=200):
        test_speeds = {label: df['speed_' + labels[label]].to_numpy()[t_index] for label in labels}
        test_intensities = {label: df['intensity_'+labels[label]].to_numpy()[t_index] for label in labels}
        if intense:
            lsta = st.local_spacetime_alt(test_speeds, target_index, additional_features=test_intensities, rescaling=rescaling, interp=interp)
        else:
            lsta = st.local_spacetime_alt(test_speeds, target_index, interp=interp)
        output = []
        ymax, xmax, zmax = 0, 0, 0
        for key in lsta:
            if intense:
                output += [(x[0], x[1], x[2]) for x in lsta[key]]
            else:
                output += [(x[0], x[1]) for x in lsta[key]]
            xmax = max([xmax]+[x[0] for x in lsta[key]])
            ymax = max([ymax]+[x[1] for x in lsta[key]])
            if intense:
                zmax = max([zmax]+[x[2] for x in lsta[key]])
                output += [(x, 0, z) for x in np.arange(0, xmax, interp) for z in np.arange(0, zmax, interp)]
                output += [(0, y, z) for y in np.arange(0, ymax, interp) for z in np.arange(0, zmax, interp)]
            else:
                output += [(x[0], x[1]) for x in np.linspace((0, 0), (xmax, 0), int(xmax/interp))]
                output += [(x[0], x[1]) for x in np.linspace((0, 0), (0, ymax), int(xmax/interp))]
        return output


    def get_lstc(t_index, interp=None):
        test_speeds = {label: date_df['speed_' + labels[label]].to_numpy()[t_index] for label in labels}
        lstc = st.comp_lst(test_speeds, target_index, interp=interp)
        output = dict()
        for key in lstc:
            output[key] = ([x[0] for x in lstc[key]], [y[1] for y in lstc[key]])
        return output


    def tda_lstc(t_index, interp=None):
        test_speeds = {label: date_df['speed_' + labels[label]].to_numpy()[t_index] for label in labels}
        lstc = st.comp_lst(test_speeds, target_index, interp=interp)
        output = []
        for key in lstc:
            output += [(x[0], x[1]) for x in lstc[key]]
        return output


    def get_lsta_numpy(t_index):
        test_speeds = {label: date_df['speed_' + labels[label]].to_numpy()[t_index] for label in labels}
        test_intensities = {label: date_df['intensity_' + labels[label]].to_numpy()[t_index] for label in labels}
        lsta = st.local_spacetime_alt(test_speeds, target_index, test_intensities)
        return np.array(lsta)


    def rev(x):
        if x < 70:
            return 20
        else:
            return 1e-2


    def get_lst_rev(t_index):
        test_speeds = [{label: date_df['speed_' + labels[label]].to_numpy()[i + t_index] for label in labels} for i
                       in range(window_size)]
        lsts = st.local_spacetime(test_speeds, target_index)
        return [(x[0], x[1]) for localst in lsts for x in localst]


    # n = len(date_df)-50
    # w_dists = np.zeros(n)
    # rips = Rips(maxdim=2)
    #
    # with warnings.catch_warnings():
    #     warnings.filterwarnings('ignore', category=UserWarning)
    #     for i in tqdm(range(n)):
    #         ac1 = gd.AlphaComplex(points=tda_lstc(i, interp=.2))
    #         st1 = ac1.create_simplex_tree()
    #         st1.compute_persistence()
    #         dgm1 = st1.persistence_intervals_in_dimension(0)
    #         ac2 = gd.AlphaComplex(points=tda_lstc(i+1, interp=.2))
    #         st2 = ac2.create_simplex_tree()
    #         st2.compute_persistence()
    #         dgm2 = st2.persistence_intervals_in_dimension(0)
    #         w_dists[i] = wd(dgm1, dgm2, matching=False)
    #
    # fig, ax = plt.subplots(figsize=(12, 9))
    # ax.set_title(f"{dt.df_date(data.dfs, date_index)}")
    # ax.plot(date_df['speed_RWS01_MONIBAS_0201hrr0413ra'].to_numpy(), label="Speed")
    # ax.set_xlabel("Minute")
    # ax.set_ylabel("Speed")
    # ax2 = ax.twinx()
    # ax2.plot(np.arange(1, len(w_dists)+1), w_dists, 'r', label="W distance")
    # ax2.set_ylabel("W distance")
    # fig.legend(labelcolor="linecolor")
    # plt.savefig(f"spacetime/wassersteinplots/act_rev_in{window_size}_{dt.df_date(data.dfs, 0)}.png")

    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # # ax = fig.add_subplot()
    # plott = 144
    # ax.set_title(f"{ward} information flow to target {target_index} at t={plott}")
    # plt.grid(True)
    # ax.set_xlabel("Time")
    # ax.set_ylabel("Distance")
    # plotset = tda_lsta(plott, intense=True, interp=interpolation, df=date_df)
    # # for plot in plotset.values():
    # #     ax.scatter(plot[0], plot[1], s=.25)
    # xs = [x[0] for x in plotset]
    # ys = [y[1] for y in plotset]
    # zs = [z[2] for z in plotset]
    # ax.scatter(xs, ys, zs, s=1)
    #
    # nrows, ncols = 3, 4
    # fig, axs = plt.subplots(nrows, ncols)
    # fig.suptitle(f"{ward}, {dt.df_date(data.dfs, date_index)}, interp={interpolation}")
    # for i in range(nrows*ncols):
    #     t = 120+10*i
    #     ac = gd.AlphaComplex(points=tda_lsta(t, intense=True, df=date_df, interp=interpolation))
    #     stree = ac.create_simplex_tree(max_alpha_square=15)
    #     stree.compute_persistence()
    #     # dgm = stree.persistence_intervals_in_dimension(1)
    #     dgm = stree.persistence()
    #     ax = gd.plot_persistence_diagram(dgm, axes=axs[i//ncols, i % ncols])
    #     ax.set_title(f"Time index {t}")
    #
    # n = len(date_df)-1
    # w_dists = np.empty(n)
    # ac = gd.AlphaComplex(points=tda_lsta(0, intense=False, interp=interpolation, df=date_df))
    # stree = ac.create_simplex_tree(max_alpha_square=15)
    # stree.compute_persistence()
    # prev = stree.persistence_intervals_in_dimension(1)
    # bc = Landscape(num_landscapes=2, resolution=5)
    # result = bc.fit_transform([prev])
    #
    #
    # def get_next(i, interp, df):
    #     ac = gd.AlphaComplex(points=tda_lsta(i+1, intense=False, interp=interp, df=df))
    #     stree = ac.create_simplex_tree()
    #     stree.compute_persistence()
    #     return stree.persistence_intervals_in_dimension(1)
    # for i in tqdm(range(n)):
    #     next = get_next(i, interpolation, date_df)
    #     w_dists[i] = wd(prev, next, matching=False)
    #     prev = next
    #
    # fig, ax1 = plt.subplots()
    # ax1.plot(date_df["speed_"+target].to_numpy(), label='Speed 413')
    # ax1.plot(date_df["speed_RWS01_MONIBAS_0201hrr0363ra"].to_numpy(), label='Speed 363')
    # # ax1.plot(date_df["speed_RWS01_MONIBAS_0201hrr0413ra"].to_numpy(), label='Speed 413')
    # # ax1.plot(date_df["speed_RWS01_MONIBAS_0201hrr0444ra"].to_numpy(), label='Speed 444 ')
    # # ax1.plot(date_df["speed_RWS01_MONIBAS_0201hrr0452ra"].to_numpy(), label='Speed 452')
    # # ax1.plot(date_df["speed_RWS01_MONIBAS_0201hrr0478ra"].to_numpy(), label='Speed 478')
    # ax1.set_title(f"{ward} information, i={date_index}, {dt.df_date(data.dfs, date_index)}, interp={interpolation}")
    # ax2 = ax1.twinx()
    # ax2.plot(np.arange(1, n+1), w_dists, 'r--', label='Wasserstein distance')
    # fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    #
    # landscape_list = []
    # ls = Landscape(num_landscapes=2, resolution=5)

    # for df in [date_df]:  # tqdm(data.export(all_speeds=True, all_intensities=True, all_norm_speeds=False, all_norm_intensities=False)):
    #     temp_list = []
    #     for i in range(len(df)):
    #         ac = gd.AlphaComplex(points=tda_lsta(i, intense=False, interp=interpolation, df=df))
    #         stree = ac.create_simplex_tree(max_alpha_square=32)
    #         stree.compute_persistence()
    #         temp_list.append(ls.fit_transform([stree.persistence_intervals_in_dimension(1)]))
    #     landscape_list.append(temp_list)
    # ax1.plot(8*np.sum(np.array(landscape_list[0]),axis=2).flatten())
    # with open("snelweg/grote_gegevens/with_intensity_rs200_landscape_forward_2x5.pkl", 'wb') as f:
    #     pickle.dump(landscape_list, f)

    dfslst = []
    dates = [46, 98, 120]
    # for k, df in tqdm(enumerate(data.dfs), total=len(data.dfs)):
    for k in dates:
        df = data.dfs[k]
        dts = np.array([])
        for i in range(len(df)):
            tda = tda_lsta(i, interp=.84, intense=False, df=df)
            if i == 0:
                tarrs = np.array(tda)
            else:
                tarrs = np.vstack([tarrs, np.array(tda)])
            dts = np.append(dts, np.repeat(df.index._data[i], len(tda)))
        dfslst.append(pd.DataFrame(tarrs, columns=['time', 'distance']).assign(day=dt.df_date(data.dfs, k), datetime=dts))
        # bigdata = pd.concat(dfslst, axis=0)
        dfslst[-1].to_csv(f'{k}_tda_interp_.84_{ward}.csv')
