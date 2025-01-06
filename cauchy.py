import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import seaborn as sns
sns.set_theme(style='whitegrid')
from scipy.stats import cauchy, norm
palette = sns.color_palette()


def o(q1, q2, v1, v2):
    return (q1-q2)/((q1/v1)-(q2/v2))


def get_q(x, q):
    return sorted(x)[int(len(x)*(1-q)/2):int(len(x)*(1-(1-q)/2))]


if __name__ == "__main__":
    N = int(1e6)
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    rng = np.random.default_rng(1)
    close_params = [(2000, 200), (1950, 200), (100, 5), (85, 5)]
    far_params = [(2000, 200), (1000, 200), (100, 5), (12.685, 5)]
    params = far_params
    q1s = rng.normal(params[0][0], params[0][1], N)
    q2s = rng.normal(params[1][0], params[1][1], N)
    v1s = rng.normal(params[2][0], params[2][1], N)
    v2s = rng.normal(params[3][0], params[3][1], N)
    qr = .9
    os = o(q1s, q2s, v1s, v2s)
    qs = get_q(os, qr)
    p = sns.kdeplot(qs, label='KDE', ax=axs[0])
    loc, scale = cauchy.fit(qs)
    mu, sig = norm.fit(qs)
    x = np.linspace(axs[0].get_xlim()[0], axs[0].get_xlim()[1], 1000)
    cy = cauchy.pdf(x, loc=loc, scale=scale)
    ny = norm.pdf(x, loc=mu, scale=sig)
    axs[0].plot(x, cy, color=palette[1], linestyle='--', label='Cauchy fit')
    axs[0].plot(x, ny, color=palette[2], linestyle=':', label='Normal fit')
    axs[0].legend()
    axs[0].set_xlim(np.min(qs), np.max(qs))

    p2 = sns.ecdfplot(qs, label='ECDF', ax=axs[1])
    cy2 = cauchy.cdf(x, loc=loc, scale=scale)
    ny2 = norm.cdf(x, loc=loc, scale=scale)
    axs[1].plot(x, cy2, color=palette[1], linestyle='--', label='Cauchy fit')
    axs[1].plot(x, ny2, color=palette[2], linestyle=':', label='Normal fit')
    axs[1].legend()
    axs[1].set_xlim(np.min(qs), np.max(qs))

