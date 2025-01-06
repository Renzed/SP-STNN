import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import gudhi as gd
from tqdm import tqdm
import numpy as np
import tqdm_pathos as tp

# matplotlib.use('Qt5Agg')


class ScatterAnim:
    def __init__(self, data, interval=20, **plot_args):
        self.data = data
        mins = np.min(np.array([np.min(d, axis=0) for d in data]), axis=0) * 1.1
        maxs = np.max(np.array([np.max(d, axis=0) for d in data]), axis=0) * 1.1
        self.xlim = (mins[0], maxs[0])
        self.ylim = (mins[1], maxs[1])
        self.fig, self.ax = plt.subplots()
        self.scat = self.ax.scatter(data[0][:, 0], data[0][:, 1], **plot_args)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.n_frames = len(data)
        self.anim = FuncAnimation(self.fig, self.animate, frames=range(self.n_frames), interval=interval, blit=True)
        plt.show()

    def animate(self, i):
        self.scat.set_offsets(self.data[i])
        return self.scat,

class ScatterAnim3D:
    def __init__(self, data, interval=20, **plot_args):
        self.data = data
        mins = np.min(np.array([np.min(d, axis=0) for d in data]), axis=0) * 1.1
        maxs = np.max(np.array([np.max(d, axis=0) for d in data]), axis=0) * 1.1
        self.xlim = (mins[0], maxs[0])
        self.ylim = (mins[1], maxs[1])
        self.zlim = (mins[2], maxs[2])
        self.fig, self.ax = plt.subplots()
        self.ax.remove()
        self.ax = self.fig.add_subplot(projection='3d')
        self.scat, = self.ax.plot(data[0][:, 0], data[0][:, 1], data[0][:, 2], linestyle="", marker='.')
        # self.scat = self.ax.scatter(data[0][:, 0], data[0][:, 1], data[0][:, 2], **plot_args)
        self.ax.set_xlim(self.xlim)
        self.ax.set_ylim(self.ylim)
        self.ax.set_zlim(self.zlim)
        self.n_frames = len(data)
        self.anim = FuncAnimation(self.fig, self.animate, frames=range(self.n_frames), interval=interval, blit=True)
        plt.show()

    def animate(self, i):
        self.scat.set_data(self.data[i][:, 0], self.data[i][:, 1])
        self.scat.set_3d_properties(self.data[i][:, 2])
        return self.scat,


class DiagAnim:
    def __init__(self, data, interval=20, blit=False, **plot_args):
        self.data = data
        self.fig, self.ax = plt.subplots()
        gd.plot_persistence_diagram(data[0], axes=self.ax)
        dlims = np.max(np.array([f[1] for d in data for f in d]), axis=0) * 1.1
        self.diag_xlim = (self.ax.get_xlim()[0], dlims[0])
        self.diag_ylim = (self.ax.get_ylim()[0], dlims[0])
        self.ax.set_xlim(self.diag_xlim)
        self.ax.set_ylim(self.diag_ylim)
        self.n_frames = len(data)
        self.anim = FuncAnimation(self.fig, self.animate, frames=range(self.n_frames), interval=interval, blit=blit)
        plt.show()

    def animate(self, i):
        self.ax.clear()
        gd.plot_persistence_diagram(self.data[i], axes=self.ax)
        self.ax.set_xlim(self.diag_xlim)
        self.ax.set_ylim(self.diag_ylim)
        return self.ax,


class ScatDiagAnim:
    def __init__(self, scat_data, diag_data, interval=20, blit=False, **plot_args):
        self.scat_data = scat_data
        self.diag_data = diag_data
        assert len(scat_data) == len(diag_data)
        self.fig, self.axs = plt.subplots(1, 2, figsize=(8, 6))
        mins = np.min(np.array([np.min(d, axis=0) for d in scat_data]), axis=0) * 1.1
        maxs = np.max(np.array([np.max(d, axis=0) for d in scat_data]), axis=0) * 1.1
        self.scat_xlim = (mins[0], maxs[0])
        self.scat_ylim = (mins[1], maxs[1])
        dlims = np.max(np.array([f[1] for d in diag_data for f in d]), axis=0) * 1.1
        self.scat = self.axs[0].scatter(scat_data[0][:, 0], scat_data[0][:, 1], **plot_args)
        self.axs[0].set_aspect('equal')
        gd.plot_persistence_diagram(diag_data[0], axes=self.axs[1])
        self.diag_xlim = (self.axs[1].get_xlim()[0], dlims[0])
        self.diag_ylim = (self.axs[1].get_ylim()[0], dlims[0])
        self.axs[0].set_xlim(self.scat_xlim)
        self.axs[0].set_ylim(self.scat_ylim)
        self.axs[1].set_xlim(self.diag_xlim)
        self.axs[1].set_ylim(self.diag_ylim)
        self.n_frames = len(scat_data)
        self.anim = FuncAnimation(self.fig, self.animate, frames=range(self.n_frames), interval=interval, blit=blit)
        plt.show()

    def animate(self, i):
        self.scat.set_offsets(self.scat_data[i])
        self.axs[0].set_title(f"Frame = {i}")
        self.axs[1].clear()
        gd.plot_persistence_diagram(self.diag_data[i], axes=self.axs[1])
        self.axs[1].set_xlim(self.diag_xlim)
        self.axs[1].set_ylim(self.diag_ylim)
        return [self.scat, self.axs[1]]


class ScatDiagAnim3D:
    def __init__(self, scat_data, diag_data, interval=20, blit=False, **plot_args):
        self.scat_data = scat_data
        self.diag_data = diag_data
        assert len(scat_data) == len(diag_data)
        self.fig, self.axs = plt.subplots(1, 2, figsize=(12, 9), layout='constrained')
        self.axs[0].remove()
        self.axs[0] = self.fig.add_subplot(121, projection='3d')
        self.axs[0].set_aspect('equal')
        mins = np.min(np.array([np.min(d, axis=0) for d in scat_data]), axis=0) * 1.1
        maxs = np.max(np.array([np.max(d, axis=0) for d in scat_data]), axis=0) * 1.1
        self.scat_xlim = (mins[0], maxs[0])
        self.scat_ylim = (mins[1], maxs[1])
        self.scat_zlim = (mins[2], maxs[2])
        dlims = np.max(np.array([f[1] for d in diag_data for f in d]), axis=0) * 1.1
        self.scat, = self.axs[0].plot(scat_data[0][:, 0], scat_data[0][:, 1], scat_data[0][:, 2], linestyle="", marker=".")
        gd.plot_persistence_diagram(diag_data[0], axes=self.axs[1])
        self.diag_xlim = (self.axs[1].get_xlim()[0], dlims[0])
        self.diag_ylim = (self.axs[1].get_ylim()[0], dlims[0])
        self.axs[0].set_xlim(self.scat_xlim)
        self.axs[0].set_ylim(self.scat_ylim)
        self.axs[0].set_zlim(self.scat_zlim)
        self.axs[1].set_xlim(self.diag_xlim)
        self.axs[1].set_ylim(self.diag_ylim)
        self.n_frames = len(scat_data)
        self.anim = FuncAnimation(self.fig, self.animate, frames=range(self.n_frames), interval=interval, blit=blit)
        plt.show()

    def animate(self, i):
        self.scat.set_data(self.scat_data[i][:, 0], self.scat_data[i][:, 1])
        self.scat.set_3d_properties(self.scat_data[i][:, 2])
        self.axs[0].set_title(f"Frame = {i}")
        self.axs[1].clear()
        gd.plot_persistence_diagram(self.diag_data[i], axes=self.axs[1])
        self.axs[1].set_xlim(self.diag_xlim)
        self.axs[1].set_ylim(self.diag_ylim)
        return [self.scat, self.axs[1]]


def dgm_from_pointcloud(cloud, max_dimension=2):
    rc = gd.RipsComplex(points=cloud)
    st = rc.create_simplex_tree(max_dimension=max_dimension)
    return st.persistence()


def dgms_from_pointclouds(clouds, max_dimension=2):
    return [dgm_from_pointcloud(cloud, max_dimension) for cloud in tqdm(clouds)]


def ugly_sub(cloud, max_dimension, sparse):
    import gudhi as gd
    return gd.RipsComplex(points=cloud, sparse=sparse).create_simplex_tree(max_dimension=max_dimension).persistence()


def ugly_alpha_sub(cloud):
    import gudhi as gd
    return gd.AlphaComplex(points=cloud).create_simplex_tree(max_alpha_square=.3).persistence()


def fast_dgms_from_pointclouds(clouds, max_dimension=2, sparse=None, n_cpus=6):
    print(f'Calculating dgms using {n_cpus} cpu cores')
    return tp.map(ugly_sub, clouds, max_dimension, sparse=sparse, n_cpus=n_cpus)

def fast_alpha_dgms_from_pointclouds(clouds, n_cpus=6):
    print(f'Calculating dgms using {n_cpus} cpu cores')
    return tp.map(ugly_alpha_sub, clouds, n_cpus=n_cpus)


if __name__ == '__main__':
    import numpy as np
    import time

    x0 = np.linspace(0, 2 * np.pi, 144)
    y0 = np.sin(x0)
    z0 = np.cos(x0)
    circle = np.stack([np.cos(x0), np.sin(x0), np.cos(x0)]).T
    scatterdata = [circle[:i + 1, :] for i in range(len(x0))]
    # scat_anim = ScatterAnim(scatterdata)
    scat_anim = ScatterAnim3D(scatterdata)

    # t0 = time.time()
    # dgms = dgms_from_pointclouds(scatterdata)
    # t1 = time.time()
    # print(f"Normal: {t1-t0:.1f}s")
    #
    # t0 = time.time()
    # dgms = fast_dgms_from_pointclouds(scatterdata)
    # t1 = time.time()
    # print(f"MP: {t1-t0:.1f}s")
    # # dgm_anim = DiagAnim(dgms)
    #
    # double_anim = ScatDiagAnim(scatterdata, dgms)
