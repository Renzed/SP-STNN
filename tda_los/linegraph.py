import networkx as nx
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from tda_los.animodule import ScatDiagAnim, fast_dgms_from_pointclouds, fast_alpha_dgms_from_pointclouds
from snelweg.datatools2 import NDW_handler

matplotlib.use('Qt5Agg')
sns.set_theme(style='whitegrid')


def fint(x):
    if x[-2:] == 'ra':
        return int(x[2:5])
    else:
        return int(x[:3])

def dgm_in_dim(dgm, dim):
    newdgm = [i[1] for i in dgm if i[0] == dim]
    return np.array(newdgm)


if __name__ == "__main__":
    plt.close('all')
    mode = "zoeterwoude"
    weekpart = "werkweek"
    if mode == "terugslag" and weekpart == "werkweek":
        sel = ['RWS01_MONIBAS_0201hrr0433ra', 'RWS01_MONIBAS_0201hrr0439ra', 'RWS01_MONIBAS_0201hrr0444ra',
               'RWS01_MONIBAS_0201hrr0449ra', 'RWS01_MONIBAS_0201hrr0452ra', 'RWS01_MONIBAS_0201hrr0461ra']
        data = NDW_handler('tda_los/terugslag_werkweek/terugslag_detail_maand.csv', loc_list=sel, smoothing=1)
        bullshit = 'fake_0428ra'
    if mode == "terugslag" and weekpart == "weekend":
        sel = ['RWS01_MONIBAS_0201hrr0433ra', 'RWS01_MONIBAS_0201hrr0439ra', 'RWS01_MONIBAS_0201hrr0444ra',
               'RWS01_MONIBAS_0201hrr0449ra', 'RWS01_MONIBAS_0201hrr0452ra', 'RWS01_MONIBAS_0201hrr0461ra']
        data = NDW_handler('tda_los/terugslag_weekend/terugslag_detail_maand_weekend.csv', loc_list=sel, smoothing=1)
        bullshit = 'fake_0428ra'
    if mode == "rijnsweerd" and weekpart == "werkweek":
        sel = ['RWS01_MONIBAS_0271hrl0854ra', 'RWS01_MONIBAS_0271hrl0851ra', 'RWS01_MONIBAS_0271hrl0847ra',
               'RWS01_MONIBAS_0271hrl0844ra', 'RWS01_MONIBAS_0271hrl0839ra', 'RWS01_MONIBAS_0271hrl0835ra',
               'RWS01_MONIBAS_0271hrl0829ra_1', 'RWS01_MONIBAS_0271hrl0825ra']
        data = NDW_handler('tda_los/rijnsweerd_werkweek/rijnsweerd_werkweek.csv', loc_list=sel, smoothing=1)
        bullshit = 'fake_0859ra'
    if mode == "rijnsweerd" and weekpart == "weekend":
        sel = ['RWS01_MONIBAS_0271hrl0854ra', 'RWS01_MONIBAS_0271hrl0851ra', 'RWS01_MONIBAS_0271hrl0847ra',
               'RWS01_MONIBAS_0271hrl0844ra', 'RWS01_MONIBAS_0271hrl0839ra', 'RWS01_MONIBAS_0271hrl0835ra',
               'RWS01_MONIBAS_0271hrl0829ra_1', 'RWS01_MONIBAS_0271hrl0825ra']
        data = NDW_handler('tda_los/rijnsweerd_weekend/rijnsweerd_weekend.csv', loc_list=sel, smoothing=1)
        bullshit = 'fake_0859ra'
    if mode == "zoeterwoude" and weekpart == "werkweek":
        sel = ['RWS01_MONIBAS_0041hrl0391ra', 'RWS01_MONIBAS_0041hrl0388ra', 'RWS01_MONIBAS_0041hrl0384ra',
               'RWS01_MONIBAS_0041hrl0380ra', 'RWS01_MONIBAS_0041hrl0375ra', 'RWS01_MONIBAS_0041hrl0371ra',
               'RWS01_MONIBAS_0041hrl0367ra', 'RWS01_MONIBAS_0041hrl0362ra']
        data = NDW_handler('tda_los/zoeterwoude_werkweek/zoeterwoude_dorp_werkweek.csv', loc_list=sel, smoothing=1)
        bullshit = 'fake_0396ra'
    if mode == "zoeterwoude" and weekpart == "weekend":
        sel = ['RWS01_MONIBAS_0041hrl0391ra', 'RWS01_MONIBAS_0041hrl0388ra', 'RWS01_MONIBAS_0041hrl0384ra',
               'RWS01_MONIBAS_0041hrl0380ra', 'RWS01_MONIBAS_0041hrl0375ra', 'RWS01_MONIBAS_0041hrl0371ra',
               'RWS01_MONIBAS_0041hrl0367ra', 'RWS01_MONIBAS_0041hrl0362ra']
        data = NDW_handler('tda_los/zoeterwoude_weekend/zoeterwoude_dorp_weekend.csv', loc_list=sel, smoothing=1)
        bullshit = 'fake_0396ra'
    df = data.dfs[0]
    df = df.assign(speed_fake_0428ra=np.repeat(100, df.shape[0]), intensity_fake_0428ra=np.repeat(4000, df.shape[0]))
    for loc in sel:
        df['density_' + loc] = df['intensity_' + loc] / df['speed_' + loc]
    sel = [bullshit] + sel
    date = df.index[0][:10]
    df.index = pd.to_datetime(df.index)
    sdf = df[['speed_' + loc for loc in data.locations]]
    osdf = sdf[np.logical_and(sdf.index > pd.to_datetime(date + ' 06:00:00'),
                              sdf.index < pd.to_datetime(date + ' 06:30:00'))]
    odf = df[np.logical_and(df.index > pd.to_datetime(date + ' 06:00:00'),
                            df.index < pd.to_datetime(date + ' 06:30:00'))]
    wdf = df[np.logical_and(df.index > pd.to_datetime(date + ' 05:50:00'),
                            df.index < pd.to_datetime(date + ' 06:50:00'))]
    oldf = df[np.logical_and(df.index > pd.to_datetime(date + ' 06:00:00'),
                             df.index < pd.to_datetime(date + ' 10:30:00'))]
    # for i in range(len(odf.index)):
    #     fig, ax = plt.subplots()
    #     sns.lineplot(osdf, ax=ax)
    #     ax.axvline(x=odf.index[i], color='k')
    #     ax.set_title(f'Frame {i:02}')
    #     plt.savefig(f'tda_los/filmpjes/data/A20_to_fake_2024-08-27/speeds_{i:02}.png')
    #     plt.close()
    g = nx.DiGraph()
    g.add_nodes_from(sel)
    g.add_weighted_edges_from(
        [(sel[i], sel[i + 1], np.abs(fint(sel[i + 1][-7:]) - fint(sel[i][-7:])) / 10) for i in range(len(sel) - 1)])
    g = g.reverse()
    path = nx.shortest_path(g, source=sel[-1], target=sel[0])


    # def omegas(idf, loc1, loc2):
    #     return ((idf[f'intensity_RWS01_MONIBAS_0201hrr0{loc1}ra'] - idf[f'intensity_RWS01_MONIBAS_0201hrr0{loc2}ra']) /
    #             (idf[f'density_RWS01_MONIBAS_0201hrr0{loc1}ra'] - idf[f'density_RWS01_MONIBAS_0201hrr0{loc2}ra'])).map(
    #         lambda x: min(x, 0) if x>-50 else 0)
    #
    #
    # omg49 = omegas(wdf, 444, 449)
    # # omg9 = omegas(wdf, 433, 444)
    #
    # fig, axs = plt.subplots(ncols=2)
    # axs[0].plot(omg49, label='omg49')
    # axs[0].set_title("min(ω, 0) between 1.0 km and 1.5 km downstream")
    # axs[0].set_xlim(wdf.index[0], wdf.index[-1])
    # # axs[0].plot(omg9, label='omg9')
    # axs[1].plot(wdf['speed_RWS01_MONIBAS_0201hrr0433ra'], label='Prediction target')
    # axs[1].plot(wdf['speed_RWS01_MONIBAS_0201hrr0439ra'], '--', label='0.5 km downstream')
    # axs[1].plot(wdf['speed_RWS01_MONIBAS_0201hrr0444ra'], 'g:', label='1.0 km downstream')
    # axs[1].plot(wdf['speed_RWS01_MONIBAS_0201hrr0449ra'], 'r', ls='-.', label='1.5 km downstream')
    # axs[1].legend()
    # axs[1].set_title('Speeds')
    # axs[1].set_xlim(wdf.index[0], wdf.index[-1])


    def get_line(t_index, interp=.05, data_df=df, scale=200, gain=1):
        d_max = np.abs((fint(sel[-1][-7:]) - fint(sel[0][-7:])) / 10)
        xy = np.array([[0, d_max]])
        for out, to in zip(path[:-1], path[1:]):
            dd = np.abs((fint(to[-7:]) - fint(out[-7:]))) / 10
            speed = (data_df.iloc[t_index]['speed_' + to] + data_df.iloc[t_index]['speed_' + to]) / 2
            speed = gain * (speed - 100) + 100
            delta = np.array([1.8 * 60 * dd / speed, -dd])
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


    def get_lines(t_index, interp=.05, data_df=df, resolution=2, scale=200, gain=1):
        maxline = get_line(t_index, interp, data_df, scale=scale, gain=gain)
        max_d = np.max(integer_floor(maxline[:, 1], integer=resolution))
        if interp == 'intensity':
            max_i = np.max(data_df.iloc[t_index][['intensity_' + loc for loc in data.locations]].to_numpy()) + 1e-1
            interp_min = scale / max_i
        else:
            interp_min = interp
        xy = np.concatenate(
            [shift_line(maxline, offset=i, resolution=resolution) for i in range(int(max_d * resolution))], axis=0)
        xy = xy[np.all(xy >= 0, axis=1), :]
        maxes = np.max(xy, axis=0)
        xy = np.append(xy, np.linspace(np.array([0, 0]), np.array([0, maxes[1]]), int(maxes[1] / interp_min)), axis=0)
        xy = np.append(xy, np.linspace(np.array([0, 0]), np.array([maxes[0], 0]), int(maxes[0] / interp_min)), axis=0)
        xy = np.unique(xy, axis=0)
        return xy

    def get_graphs(df, base_g):
        output = []
        for _, r in df.iterrows():
            ig = base_g.copy()
            for e in ig.edges:
                ig[e[0]][e[1]]['weight'] /= r['speed_'+e[1]] / 60
            output.append(ig)
        return output

    def maxmin(G):
        A = nx.floyd_warshall_numpy(G)
        return np.max(A[A<np.inf])

    # test = get_lines(0, interp='intensity', data_df=df)
    # fig, ax = plt.subplots()
    # ax.scatter(test[:, 0], test[:, 1], s=1)

    for i, temp_df in enumerate(data.dfs):
        if mode == "terugslag":
            temp_df = temp_df.assign(speed_fake_0428ra=np.repeat(100, temp_df.shape[0]), intensity_fake_0428ra=np.repeat(4000, temp_df.shape[0]))
        elif mode == "rijnsweerd":
            temp_df = temp_df.assign(speed_fake_0859ra=np.repeat(100, temp_df.shape[0]), intensity_fake_0859ra=np.repeat(4000, temp_df.shape[0]))
        elif mode == "zoeterwoude":
            temp_df = temp_df.assign(speed_fake_0396ra=np.repeat(100, temp_df.shape[0]), intensity_fake_0396ra=np.repeat(4000, temp_df.shape[0]))
        clouds = [get_lines(i, interp='intensity', data_df=temp_df, scale=200) for i in range(temp_df.shape[0])]
        temp_date = temp_df.index[0][:10]
        temp_df.index = pd.to_datetime(temp_df.index)
        # graphs = get_graphs(df, g)
        dgms = fast_dgms_from_pointclouds(clouds, n_cpus=5, max_dimension=3, sparse=.2)
        print(f'Running ({i}) for df at date {temp_date} with shape {temp_df.shape}')
        with open(f'tda_los/{mode}_{weekpart}/dgms/i'+str(i)+'dgms_ochtendspits_sparse_'+temp_date+'.pkl', 'wb') as f:
            pickle.dump(dgms, f)
    # anim = ScatDiagAnim(clouds, dgms, s=1, interval=16)
    # plt.close()
    # anim.anim.save('tda_los/filmpjes/data/A20_to_fake_heledag_sparse_' + date + '.gif', writer='pillow', fps=60)

    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.plot(wdf['speed_RWS01_MONIBAS_0201hrr0433ra'], label='Prediction target')
    # ax.plot(wdf['speed_RWS01_MONIBAS_0201hrr0439ra'], ls='--', label='0.6 km downstream')
    # ax.plot(wdf['speed_RWS01_MONIBAS_0201hrr0444ra'], ls='-.', label='1.1 km downstream')
    # ax.plot(wdf['speed_RWS01_MONIBAS_0201hrr0449ra'], ls=':', label='1.6 km downstream')
    # ax.legend()
    # ax.set_xlabel('Time (dd hh:mm)')
    # ax.set_ylabel('Speed (km/h)')
    # ax.set_xlim(wdf.index[0], wdf.index[-1])
    # plt.tight_layout()
    #
    #
    # with open('tda_los/objects/idgms_heledag_2024-08-27.pkl', 'rb') as f:
    #     dgms2 = pickle.load(f)
    # # d1_features = [[x for d, x in dgm if d == 1] for dgm in dgms]
    # # d1_lifetimes = [[death - birth for birth, death in x] for x in d1_features]
    # d2_features = [[x for d, x in dgm if d == 1] for dgm in dgms2]
    # d2_lifetimes = [[death - birth for birth, death in x] for x in d2_features]
    # alpha = 0.05
    # # nd1 = [sum(np.array(x) > alpha) for x in d1_lifetimes]
    # nd2 = [sum(np.array(x) > alpha) for x in d2_lifetimes]
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax2 = ax.twinx()
    # ax.plot(df['speed_' + sel[1]], '-')
    # index = df.index
    # # ax.plot(index, [20 * sum(x) for x in d2_lifetimes], 'y')
    # ax.set_xlim(min(index), max(index))
    # ax.set_ylim(0, 150)
    # ax2.set_ylim(-0.2, 5.5)
    # ax.set_xlabel('Time (dd hh:mm)')
    # ax.set_ylabel('Speed (km/h)')
    # ax2.plot(index, [maxmin(x) for x in graphs], 'y--', label='Graph diameter')
    # ax2.plot(index, [(sum([y for y in x if y > alpha])) for x in d2_lifetimes], 'r:', label='Travel lifetime')
    # ax2.set_ylabel('Metric value')
    # ax2.legend()
    #
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax2 = ax.twinx()
    # ax.plot(wdf['speed_' + sel[1]], '-')
    # w = np.logical_and(df.index > pd.to_datetime(date + ' 05:50:00'),
    #                    df.index < pd.to_datetime(date + ' 06:50:00'))
    # index = wdf.index
    # # ax.plot(index, [20 * sum(x) for x in d2_lifetimes], 'y')
    # ax.set_xlim(min(index), max(index))
    # ax.set_ylim(0, 150)
    # ax2.set_ylim(-0.2, 5.5)
    # ax.set_xlabel('Time (dd hh:mm)')
    # ax.set_ylabel('Speed (km/h)')
    # ax.axvline(x=index[25], color='k')
    # ax2.plot(index, np.array([maxmin(x) for x in graphs])[w], 'y--', label='Graph diameter')
    # ax2.plot(index, np.array([(sum([y for y in x if y > alpha])) for x in d2_lifetimes])[w], 'r:', label='Travel lifetime')
    # ax2.set_ylabel('Metric value')
    # ax2.legend()
    #
    # plt.figure()
    # plt.plot([20 * (sum([y for y in x])) for x in d2_lifetimes])
    # plt.plot([25 * (sum([y for y in x if y > alpha])) for x in d2_lifetimes])
    #
