import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import gudhi as gd
matplotlib.use('Qt5Agg')
sns.set_theme(style='whitegrid')


if __name__ == "__main__":

    plt.close('all')

    N = 20
    r = 0.2
    crs = [0.06, 0.09, 0.18, 0.51, 1.01]
    # vrs = [0.05, 0.2, 0.35, 1.01, 2.01]

    t = np.linspace(0, (1 - 1 / N) * 2 * np.pi, N)
    xy1 = np.stack([np.cos(t), np.sin(t)])
    xy2 = .5*xy1+np.expand_dims(np.array([1.5, 0]), axis=1)
    xy = np.concatenate([xy1, xy2], axis=1)

    def circplot(dat, rad, limrat=1.2, alpha=.2):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(dat[0, :], dat[1, :], 'ko')
        circs = [plt.Circle(dat[:, i], rad, alpha=alpha, edgecolor='k') for i in range(dat.shape[1])]
        maxplot = limrat * np.max([np.min(dat), np.max(dat)])
        ax.set_xlim(-maxplot, maxplot)
        ax.set_ylim(-maxplot, maxplot)
        ax.set_aspect('equal')
        for circ in circs:
            ax.add_patch(circ)
        return ax

    # [circplot(xy, r) for r in crs]

    rc = gd.RipsComplex(points=xy.T)
    st = rc.create_simplex_tree(max_dimension=2)
    dgm = st.persistence()
    ax1 = gd.plot_persistence_barcode(dgm)
    # ax1.xaxis.set_visible(False)
    # ax1.axvline(x=0.156, color='k')
    ax1.set_xticks([0.156, 0.313, 0.891, 1.783])
    ax1.set_xticklabels([0.08, 0.16, 0.5, 1.0])
    # ax1.axvline(x=0.313, color='k')
    # ax1.axvline(x=0.891, color='k')
    # ax1.axvline(x=1.783, color='k')
    ax1.set_title('')
    ax2 = gd.plot_persistence_diagram(dgm)
    ax2.xaxis.set_visible(True)
    ax2.yaxis.set_visible(True)
    # ax2.axvline(x=0.156, color='k', alpha=0.5)
    ax2.set_xticks([0, 0.156, 0.313])
    ax2.set_xticklabels([0, 0.08, 0.16])
    ax2.set_yticks([0, 0.891, 1.783, 1.9625])
    ax2.set_yticklabels([0, 0.5, 1.0, '+∞'])
    # ax2.axvline(x=0.313, color='k', alpha=0.5)
    # ax2.axhline(y=0.891, color='k', alpha=0.5)
    # ax2.axhline(y=1.783, color='k', alpha=0.5)
    ax2.set_title('')
