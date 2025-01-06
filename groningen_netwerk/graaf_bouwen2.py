import pandas as pd
from tqdm import tqdm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import pickle
import snelweg.datatools2 as dt

matplotlib.use('Qt5Agg')


def link_from_dir(row, datacol):
    if datacol == 'max':
        data = 60*row['length']/row['max']
    elif datacol == 'length':
        data = row['length']
    else:
        raise ValueError("datacol not valid")
    if row['direction'] == 2:
        return row['pointnrb'], row['pointnra'], data
    elif row['direction'] == 1:
        return row['pointnra'], row['pointnrb'], data
    else:
        raise ValueError("direction not valid")


positions = pd.read_csv('groningen_netwerk/point_pos.csv')
ndw = pd.read_csv('groningen_netwerk/NDW_points.csv')
ndw = ndw.rename({'omnitrans_link_number': 'linknr', 'omnitrans_direction': 'direction'}, axis=1)
ndw.loc[ndw['linknr']==466705,'direction'] = 1
# ndw_links = [r['linknr'] for _, r in ndw.iterrows() if r['direction']==1]+[-r['linknr'] for _,r in ndw.iterrows() if r['direction']==2]


def get_mvvs_data(ndw_links, debug=False):
    links = pd.read_csv('groningen_netwerk/link_points.csv')
    pos = {int(row['pointnr']): (row['x'] - 229472, row['y'] - 577530) for _, row in positions.iterrows()}
    linkpos = {r['linknr']: (np.array(pos[r['pointnra']])+np.array(pos[r['pointnrb']]))/2 for _, r in links.iterrows()}
    linkpos = linkpos | {-key: val for key, val in linkpos.items()}
    links['linknr'] = links.apply(lambda x: int((1-2*(x['direction']-1))*x['linknr']), axis=1)
    to = links.apply(lambda x: int(x['pointnrb']) if x['direction'] == 1 else int(x['pointnra']), axis=1)
    out = links.apply(lambda x: int(x['pointnrb']) if x['direction'] == 2 else int(x['pointnra']), axis=1)
    links = links.assign(to=to, out=out)
    link_list = links['linknr'].unique()
    edges = dict()
    for _, link in tqdm(links.iterrows()):
        outflows = links[links['out'] == link['to']]
        for _, outflow in outflows.iterrows():
            if np.abs(outflow['linknr']) == np.abs(link['linknr']):
                continue
            edges[(link['linknr'], outflow['linknr'])] = ((link['length']+outflow['length'])/2, (link['max']+outflow['max'])/2)
    length_list = [(key[0], key[1], val[0]) for key, val in edges.items()]
    maxspeed_list = [(key[0], key[1], val[1]) for key, val in edges.items()]
    tt_list = [(key[0], key[1], 60*val[0]/val[1]) for key, val in edges.items()]
    im = plt.imread('STNN_master/groningen.png')
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.imshow(im, extent=(0, 1115 * 1000 / 108 / 0.992, 0, 964 * 1000 / 108 / 0.992))  # 1 pixel ~ 1 meter
    g = nx.DiGraph()
    g.add_nodes_from(link_list)
    tg = g.copy()
    g.add_weighted_edges_from(length_list)
    tg.add_weighted_edges_from(tt_list)

    nx.draw(g, linkpos, with_labels=False, node_size=2, ax=ax)
    nx.draw_networkx_nodes(g, linkpos, nodelist=ndw_links, node_color='r', node_size=10, ax=ax)
    adjmat = np.zeros((len(ndw_links), len(ndw_links)))
    d_mat = np.zeros_like(adjmat)
    f_mat = np.zeros((len(ndw_links), len(ndw_links)))
    ndw_g = nx.DiGraph()

    for i in tqdm(range(len(ndw_links))):
        if ndw_links[i] not in ndw_g.nodes:
            ndw_g.add_node(ndw_links[i])
        for j in range(len(ndw_links)):
            if ndw_links[j] not in ndw_g.nodes:
                ndw_g.add_node(ndw_links[j])
            try:
                path = nx.shortest_path(tg, ndw_links[i], ndw_links[j], weight='weight')
                adjmat[i, j] = nx.path_weight(tg, path, weight='weight')
                d_mat[i, j] = nx.path_weight(g, path, weight='weight')
                if i != j:
                    f_mat[i, j] = (adjmat[i, j] / (60 * d_mat[i, j] /
                                                   ((links[links['linknr'] == ndw_links[i]]['max'].iloc[0] +
                                                     links[links['linknr'] == ndw_links[j]]['max'].iloc[0])/2)))
                ndw_g.add_edge(ndw_links[i], ndw_links[j], weight=d_mat[i, j])
            except nx.exception.NetworkXNoPath:
                adjmat[i, j] = np.inf
                d_mat[i, j] = np.inf
    pmat = np.array([[np.sqrt(np.sum((np.array(linkpos[p0])-np.array(linkpos[p1]))**2))/1000 for p0 in ndw_links] for p1 in ndw_links])
    if debug:
        return g, linkpos, ndw_g, adjmat, d_mat, f_mat, pmat, ax
    else:
        return ndw_g, adjmat, d_mat, f_mat, pmat


if __name__ == "__main__":
    import os

    ndw = pd.read_csv('groningen_netwerk/NDW_points.csv')
    ndw_links = ([r['omnitrans_link_number'] for _, r in ndw.iterrows() if r['omnitrans_direction'] == 1] +
                 [-r['omnitrans_link_number'] for _, r in ndw.iterrows() if r['omnitrans_direction'] == 2])
    ndw_links = [str(x) for x in ndw_links]
    handle = dt.MVVS_handler('STNN_master/data/groningen_mvvs_data_links.csv',
                             loc_list=[str(x) for x in ndw_links], min_count_fraction=.98)
    ndw_locs = [int(x) for x in handle.locations]

    rev = False

    if rev:
        ndw_graph, adj_mat, dist_mat, frac_mat, p_mat = get_mvvs_data(ndw_locs)
        ndw_g = ndw_graph.reverse()
        labels = list(ndw_g.nodes)
        f_mat = frac_mat.T
        d_mat = dist_mat.T
        adj_mat = 60*d_mat/20
        rev_pre = "rev_"
    else:
        ndw_graph, adj_mat, d_mat, f_mat, p_mat = get_mvvs_data(ndw_locs)
        labels = ndw_locs
        rev_pre = ""

    g, linkpos, ndw_graph, adj_mat, dist_mat, frac_mat, p_mat, ax = get_mvvs_data(ndw_locs, debug=True)
    tnodes = [10]
    nx.draw_networkx_nodes(g, linkpos, nodelist=[int(ndw_locs[i]) for i in tnodes], node_color='m', node_size=100, ax=ax)
    nnodes = [30,42,49,32]
    nx.draw_networkx_nodes(g, linkpos, nodelist=[int(ndw_locs[i]) for i in nnodes], node_color='g', node_size=100, ax=ax)
    fnodes5 = [22,36,37,45,19]
    fnodes10 = [55,18,40]
    citynodes = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 42, 44, 45, 48, 49, 50, 51, 53]
    mvvs_test_points = [3, 37, 10, 45, 21, 29]
    nx.draw_networkx_nodes(g, linkpos, nodelist=[int(ndw_locs[i]) for i in mvvs_test_points], node_color='m', node_size=100, ax=ax)

    def speed_graph(row):
        ndw_tg = ndw_graph.copy()
        for edge in ndw_tg.edges:
            ndw_tg[edge[0]][edge[1]]['weight'] = 60 * (f_mat[ndw_locs.index(edge[0]), ndw_locs.index(edge[1])] *
                                                       d_mat[ndw_locs.index(edge[0]), ndw_locs.index(edge[1])] /
                                                       ((row['speed_' + str(edge[0])] + row['speed_' + str(edge[1])]) / 2))
        # n1, n2 = ndw_locs[0], ndw_locs[125]  # links 280 en -464290
        # for i in range(len(ndw_links)):
        #     for j in range(len(ndw_links)):
        #         try:
        #             s_adjmat[i, j] = nx.shortest_path_length(ndw_tg, ndw_links[i], ndw_links[j], weight='weight')
        #         except nx.exception.NetworkXNoPath:
        #             s_adjmat[i, j] = np.inf
        s_adjmat = nx.floyd_warshall_numpy(ndw_tg, weight='weight') / 10
        return s_adjmat

    if not rev:
        xmats_list = []
        for df in (pbar := tqdm([df for df in handle.dfs if len(df)>34])):
            pbar.set_description("Generating adjmats")
            xmats = np.stack([speed_graph(r) for _, r in df.iterrows()])
            xmats_list.append(xmats)


    def loc_to_arr(df, loc, i=None):
        # if i is not None:
        #     print(i)
        speeds = df['speed_' + loc].to_numpy()
        intensities = df['intensity_' + loc].to_numpy()
        # densities = intensities/(speeds+1)  # v/m = v/h / km/h
        return np.stack([intensities, speeds]).T


    datasets = [np.stack([loc_to_arr(df, str(loc), i) for loc in ndw_locs]) for i, df in tqdm(enumerate(handle.dfs)) if len(df)>34]
    # with open(f'STNN_master/data/{rev_pre}groningen_mvvs/labels.pkl', 'wb') as f:
    #     pickle.dump(labels, f)

    # assert len(xmats_list)==len(datasets)

    for i in range(len(datasets)):
        os.mkdir(f'STNN_master/data/fix/{rev_pre}groningen_flow/part{i}')
        with open(f'STNN_master/data/fix/{rev_pre}groningen_flow/part{i}/node_values.npy', 'wb') as f:
            np.save(f, datasets[i])
        with open(f'STNN_master/data/fix/{rev_pre}groningen_flow/part{i}/adj_mat.npy', 'wb') as f:
            if rev:
                np.save(f, adj_mat)
            else:
                np.save(f, xmats_list[i])
        with open(f'STNN_master/data/fix/{rev_pre}groningen_flow/part{i}/p_mat.npy', 'wb') as f:
            np.save(f, p_mat)
