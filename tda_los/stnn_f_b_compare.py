import gudhi as gd
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

target = 'RWS01_MONIBAS_0201hrr0413ra'

if __name__ == "__main__":
    with open("STNN_master/data/A20Gouda_alles_ochtend_stnnfull_2.pkl", 'rb') as f:
        data = pickle.load(f)
    conc_data = pd.concat(data.dfs, axis=0)
    true_f = conc_data[['check', 'stnn']]
    true_b = conc_data[['check', 'stnn_rev']]
    input_f = conc_data[['speed_'+target, 'intensity_'+target, 'stnn']]
    input_b = conc_data[['speed_'+target, 'intensity_'+target, 'stnn_rev']]

    fig, axs = plt.subplots(3, 2)
    n_subsample = 100
    s_true_f = true_f.sample(n_subsample)
    axs[0,0].scatter(true_f['check'], true_f['stnn'], alpha=.1)
    axs[0,0].axline((0,0), slope=1, color='k', linestyle=(1,(5,5)))
    axs[0,0].set_xlabel('True')
    axs[0,0].set_ylabel('Pred')
    axs[0,0].set_title('Forward')
    s_true_b = true_b.sample(n_subsample)
    axs[0,1].scatter(true_b['check'], true_b['stnn_rev'], alpha=.1)
    axs[0,1].axline((0,0), slope=1, color='k', linestyle=(1,(5,5)))
    axs[0,1].set_xlabel('True')
    axs[0,1].set_ylabel('Pred')
    axs[0,1].set_title('Backward')
    axs[1,0].scatter(s_true_f['check'], s_true_f['stnn'], alpha=.1)
    axs[1,0].axline((0,0), slope=1, color='k', linestyle=(1,(5,5)))
    axs[1,0].set_xlabel('True')
    axs[1,0].set_ylabel('Pred')
    axs[1,1].scatter(s_true_b['check'], s_true_b['stnn_rev'], alpha=.1)
    axs[1,1].axline((0,0), slope=1, color='k', linestyle=(1,(5,5)))
    axs[1,1].set_xlabel('True')
    axs[1,1].set_ylabel('Pred')

    ac = gd.RipsComplex(points=np.array(s_true_f))
    st = ac.create_simplex_tree(max_dimension=2)
    dgm = st.persistence()
    gd.plot_persistence_diagram(dgm, axes=axs[2,0])
    if n_subsample > 1000:
        del ac
        del st
        del dgm

    ac = gd.RipsComplex(points=np.array(s_true_b))
    st = ac.create_simplex_tree(max_dimension=2)
    dgm = st.persistence()
    gd.plot_persistence_diagram(dgm, axes=axs[2,1])
    if n_subsample > 1000:
        del ac
        del st
        del dgm

    fig2, axs2 = plt.subplots(1, 2)
    s_input_f = input_f.sample(n_subsample)
    s_input_b = input_b.sample(n_subsample)
    ac = gd.RipsComplex(points=np.array(s_input_f))
    st = ac.create_simplex_tree(max_dimension=3)
    dgm = st.persistence()
    gd.plot_persistence_diagram(dgm, axes=axs2[0])
    if n_subsample > 1000:
        del ac
        del st
        del dgm
    ac = gd.RipsComplex(points=np.array(s_input_b))
    st = ac.create_simplex_tree(max_dimension=3)
    dgm = st.persistence()
    gd.plot_persistence_diagram(dgm, axes=axs2[1])
    if n_subsample > 1000:
        del ac
        del st
        del dgm
