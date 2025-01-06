import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import pickle

matplotlib.use('Qt5Agg')

rng = np.random.default_rng(0)

r_max = 5
n_nodes = 20

# points = {0: (-9, 1), 1: (-5, 1), 2: (-5, 0), 3: (0, 0), 4: (-1, 1), 5: (-5, -4), 6: (10, 10)}
# physical_distances = {key: np.sqrt(val[0] ** 2 + val[1] ** 2) for key, val in points.items()}
# links = [(0, 1, 4), (1, 2, 1), (2, 3, 5), (4, 1, 4), (5, 2, 4), (6, 3, np.sqrt(200))]
# target = 3
# points = {0: (3390, 2400), 1: (2200, 1820), 2: (450, 2280), 3: (4200, 2660), 4: (4030, 1670), 5: (4420, 750),
#           6: (5080, 150), 7: (3760, 2550), 8: (3310, 2590), 9: (4770, 2910), 10: (5730, 3580), 11: (7730, 4250),
#           12: (6770, 2570), 13: (8570, 2900), 14: (10050, 2270), 15: (7440, 5000), 16: (7220, 5550), 17: (6730, 6250),
#           18: (6050, 6480), 19: (5260, 7560), 20: (4720, 7840), 21: (4370, 6830), 22: (4220, 6540), 23: (3330, 6620),
#           24: (1700, 6180), 25: (2000, 5470), 26: (2610, 3750),  # hieronder gele wegen
#           }
# pre_links_los = [(2, 1, 130), (1, 0, 130), (6, 5, 130), (5, 4, 100), (4, 3, 100), (14, 13, 130), (13, 12, 70),
#                  (12, 10, 70), (13, 11, 70)]
# pre_link_ring = [(0, 7), (7, 3), (3, 9), (9, 10), (10, 11), (11, 15), (15, 16), (16, 17), (17, 18), (18, 19), (19, 20),
#                  (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 8), (8, 0)]

g_points = {0: (450, 2280), 1: (2400, 1800), 2: (4100, 2450), 3: (4100, 1550), 4: (5700, -670),
            5: (6080, -1620), 6: (3950, 3230), 7: (4040, 3490), 8: (3850, 3620), 9: (4790, 2970), 10: (4760, 3620),
            11: (4680, 4350), 12: (5140, 3840), 13: (5200, 4060), 14: (5510, 4120), 15: (5620, 3390), 16: (5650, 3590),
            17: (6040, 3710), 18: (6650, 4520), 19: (6720, 3850), 20: (7330, 4140), 21: (7530, 4660), 22: (7730, 4190),
            23: (8230, 3370), 24: (8200, 5830), 25: (6480, 6220), 26: (5510, 7060), 27: (5320, 9240),
            28: (2590, 8500), 29: (1960, 6130), 30: (1800, 5770), 31: (60, 6680)}
g_labels = {0: 'RWS01_MONIBAS_0071hrr1932ra', 1: 'RWS01_MONIBAS_0071hrr1952ra', 2: 'GEO0B_R_RWSTI362970',
            3: 'RWS01_MONIBAS_0281hrr1989ra', 4: 'GEO0B_R_RWSTI362996', 5: 'RWS01_MONIBAS_0281hrr1954ra', 6: 'PGR08_G172_Emmaviaduct_N_HTN2227',
            7: 'PGR08_Stationsweg_O_HTN2250', 8: 'PGR08_Emmasingel_O_V1_HTN2249', 9: 'PGR08_Hereweg_N_HTN2228', 10: 'PGR08_Zuiderpark_O_HTN2251',
            11: 'PGR08_Schuitendiep_Z_V1_HTN2240', 12: 'PGR08_G460_Griffeweg_O_HTN2252', 13: 'PGR08_Europaweg_ZO_HTN2254', 14: 'PGR08_Sontweg_ZW_HTN2262',
            15: 'GEO0K_K_RWSTI605', 16: 'PGR08_G2756_Europaweg_NW_HTN2232', 17: 'RWS01_MONIBAS_0070vwc1994ra',
            18: 'PGR08_SintPetersburgweg_ZW_HTN2261', 19: 'RWS01_MONIBAS_0071hrl2001ra', 20: 'RWS01_MONIBAS_0071hrl2008ra', 21: 'GEO0K_K_0_RWSTI363056',
            22: 'RWS01_MONIBAS_0461hrr0037ra', 23: 'RWS01_MONIBAS_0071hrl2043ra', 24: 'PGR08_22803_N360_hmp_4.4_Li_HTN2220',
            25: 'PGR08_315_N46_hmp_6.3_LI_HTN2216', 26: 'PGR08_N46_hmp_7.5_Li_HTN2214', 27: 'PGR08_316_N46_hmp_12.8_Li_HTN2212', 28: 'PGR08_223_N361_hmp_4.2_Li_HTN2211',
            29: 'PGR08_229_N370_hmp_53.9_Re_HTN2205', 30: 'PGR08_227_N370_hmp_0.3_Re_HTN2203', 31: 'PGR08_03409_N355_hmp_51.8_Re_HTN2204'}
g_labels2 = {0: 'RWS01_MONIBAS_0071hrr1932ra', 1: 'RWS01_MONIBAS_0071hrr1952ra',
            4: 'GEO0B_R_RWSTI362996', 5: 'RWS01_MONIBAS_0281hrr1954ra', 6: 'PGR08_G172_Emmaviaduct_N_HTN2227',
            7: 'PGR08_Stationsweg_O_HTN2250', 8: 'PGR08_Emmasingel_O_V1_HTN2249', 9: 'PGR08_Hereweg_N_HTN2228', 10: 'PGR08_Zuiderpark_O_HTN2251',
            11: 'PGR08_Schuitendiep_Z_V1_HTN2240', 12: 'PGR08_G460_Griffeweg_O_HTN2252', 13: 'PGR08_Europaweg_ZO_HTN2254', 14: 'PGR08_Sontweg_ZW_HTN2262',
            16: 'PGR08_G2756_Europaweg_NW_HTN2232', 17: 'RWS01_MONIBAS_0070vwc1994ra',
            18: 'PGR08_SintPetersburgweg_ZW_HTN2261', 20: 'RWS01_MONIBAS_0071hrl2008ra', 21: 'GEO0K_K_0_RWSTI363056',
            22: 'RWS01_MONIBAS_0461hrr0037ra', 23: 'RWS01_MONIBAS_0071hrl2043ra', 24: 'PGR08_22803_N360_hmp_4.4_Li_HTN2220',
            25: 'PGR08_315_N46_hmp_6.3_LI_HTN2216', 26: 'PGR08_N46_hmp_7.5_Li_HTN2214', 27: 'PGR08_316_N46_hmp_12.8_Li_HTN2212', 28: 'PGR08_223_N361_hmp_4.2_Li_HTN2211',
            29: 'PGR08_229_N370_hmp_53.9_Re_HTN2205', 30: 'PGR08_227_N370_hmp_0.3_Re_HTN2203', 31: 'PGR08_03409_N355_hmp_51.8_Re_HTN2204'}
g_links = [(0,1,130), (1,2,70), (5,4,100), (4,2,100), (2,9,70), (9,15,70), (15,16, 70), (2,6,50),
           (6,7,50), (8,7,50), (7,10,50), (9,11,50), (10,12,50), (12,11,50), (13,12,50), (14,13,50), (16,13,50),
           (17,16,70), (19,17,70), (20,19,70), (22,20,70), (23,22,100), (20,18,50), (21,22,70), (21,18,50),
           (24,21,70), (25,21,70), (26,25,70), (27,26,70), (28,26,70), (28,29,70), (31,29, 100), (31,30,100),
           (30,8, 70), (29, 11, 45), (18, 14, 50), (17, 14, 50), (29, 30, 70)]
target = 13
g_physical_distances = {key: np.sqrt((val[0]-g_points[target][0]) ** 2 + (val[1]-g_points[target][1]) ** 2) for key, val in g_points.items()}
# sqrt(2) = 1.4142
# g_points = np.array([i for i in g_points.values()])
target = 13
fix_list = {(28, 29): 1.41, (28,26): 1.5, (24,21): 1.41, (29,11): 1.5, (30, 8): 1.2, (4, 2): 1.15}
# rescale = np.sqrt(((2450-5200)**2+(3160-3535)**2)/((450-8210)**2+(2280-3430)**2))
# shift = np.array([2450, 3160])-g_points[0,:]*rescale
# g_points = g_points*rescale+shift
# physical_distances = {n: np.sqrt((v[0]-points[target][0])**2+(v[1]-points[target][1])**2)/1000 for n, v in points.items()}
# pre_link_ring = [(i[0], i[1], 70) for i in pre_link_ring]
# pre_links = pre_links_los + pre_link_ring


def node_time_dist(points, node1, node2, speed, factors=None):
    if factors is None:
        factors = dict()
    if (node1, node2) in factors:
        factor = factors[(node1, node2)]
    else:
        factor = 1
    return factor * 60 * np.sqrt((points[node1][0] - points[node2][0]) ** 2 + (points[node1][1] - points[node2][1]) ** 2) / 1000 / speed


# times = [(l[0], l[1], node_time_dist(l[0], l[1], l[2])) for l in pre_links]
if __name__ == "__main__":
    g_times = [(l[0], l[1], node_time_dist(g_points, l[0], l[1], l[2], fix_list)) for l in g_links]

    goal_candidates = 5

    G = nx.Graph()
    G.add_nodes_from(list(g_points.keys()))
    G.add_weighted_edges_from(g_times)
    im = plt.imread('STNN_master/groningen.png')
    # im = plt.imread('data/groningen/Geselecteerde locaties.png')
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_aspect('equal')
    im = ax.imshow(im, extent=(0, 1115 * 1000 / 108, 0, 964 * 1000 / 108))  # 1 pixel ~ 1 meter
    # im = ax.imshow(im, extent=(0, 844 * 10, 0, 739 * 10))
    nx.draw(G, pos=g_points, with_labels=True, node_size=70, ax=ax, node_color='#aaa')
    dist_max = nx.floyd_warshall_numpy(G)
    # ndw_train_named = list(g_labels2.values())[10:14]
    # ndw_train_numbered = list(g_labels2.keys())[10:14]
    band = np.argwhere(np.abs(dist_max[:, target] - r_max) <= 1/2).flatten()
    sorted_candidates = band[np.argsort([g_physical_distances[i] for i in band])[::-1]]
    selected_candidates = sorted_candidates[:goal_candidates]
    closest_candidates = np.argsort(dist_max[:, target])[1:goal_candidates+1]
    nx.draw_networkx_nodes(G, g_points, nodelist=selected_candidates, node_color='r')
    nx.draw_networkx_nodes(G, g_points, nodelist=closest_candidates, node_color='b')
    nx.draw_networkx_nodes(G, g_points, nodelist=[target], node_color='g')
    A = nx.floyd_warshall_numpy(G)/10
    #remove: node2 GEO0B_R_RWSTI362970, node15 GEO0K_K_RWSTI605, node3 RWS01_MONIBAS_0281hrr1989ra
    A = np.delete(A, [2,3,15,19], axis=0)
    A = np.delete(A, [2,3,15,19], axis=1)
    pmat = np.array([[np.sqrt((val[0]-g_points[t][0]) ** 2 + (val[1]-g_points[t][1]) ** 2) for val in g_points.values()] for t in g_points])
    pmat = np.delete(pmat, [2,3,15,19], axis=0)
    pmat = np.delete(pmat, [2,3,15,19], axis=1)
    # with open('data/groningen/labels.pkl', 'wb') as f:
    #     pickle.dump(g_labels2, f)
    # with open('data/groningen/adjmat.npy', 'wb') as f:
    #     np.save(f, A)
    # with open('data/groningen/physmat.npy', 'wb') as f:
    #     np.save(f, pmat)
    # print(sorted_candidates)
