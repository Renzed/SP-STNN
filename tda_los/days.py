import numpy as np
from gtda.diagrams import Amplitude
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import SingleTakensEmbedding
from gtda.plotting import plot_point_cloud
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import plotly.io as pio
from sklearn.decomposition import PCA
import pandas as pd
import snelweg.datatools2 as dt
import datetime
from tqdm import tqdm

matplotlib.use('Qt5Agg')
sns.set_theme(style="whitegrid")
pio.renderers.default = 'browser'

x = np.linspace(0, 10, 1000)
y = np.sin(5 * x) + np.cos(3 * x)

try:
    print(big)
except Exception as e:
    data = dt.NDW_handler('spacetime/TDA_datasets/fuzzy.csv')
    intensity = pd.concat(data.dfs)['intensity_RWS01_MONIBAS_0201hrr0413ra']
    speed = pd.concat(data.dfs)['speed_RWS01_MONIBAS_0201hrr0413ra']
    del(data)
    density = intensity/speed
    big = intensity
    big.index = big.index.map(lambda x: pd.to_datetime(x))


def map_index(index):
    day_dif = datetime.timedelta(days=1)
    week_diff = datetime.timedelta(days=7)
    t0 = index[0]
    return index.map(lambda x: x - 6 * day_dif * int((x - t0) / week_diff))


def get_weekday_lists(weekday):
    y_wd = big[big.index.map(lambda x: x.weekday()) == weekday].copy()
    # y_wd = big.copy()
    y_wd.index = map_index(y_wd.index)
    raw_wd = y_wd.to_numpy()
    cuts = [0]
    for i in tqdm(range(len(y_wd.index) - 1)):
        if y_wd.index[i + 1] - y_wd.index[i] > datetime.timedelta(minutes=5.1):
            cuts.append(i + 1)
    cuts.append(-1)
    rwds = []
    for i in range(len(cuts) - 1):
        rwds.append(raw_wd[cuts[i]:cuts[i + 1]])
    return rwds

days = {i:None for i in range(7)}
for day in days:
    l_wd = get_weekday_lists(day)
    i_wd = np.argmax(np.array([len(x) for x in l_wd]))

    df = pd.DataFrame({'x': x, 'y': y})
    sns.lineplot(df, x='x', y='y')

    max_M = 12
    max_tau = 288
    stride = 10
    embedder = SingleTakensEmbedding(parameters_type='search', time_delay=max_tau, dimension=max_M, stride=stride)

    embedder.fit(l_wd[i_wd])
    l_wd = [x for x in l_wd if len(x)>((embedder.dimension_-1)*embedder.time_delay_)]
    y_embedded = [embedder.transform(y) for y in l_wd]
    # pca = PCA(n_components=3)
    # print(embedder.dimension_, embedder.time_delay_)
    # y_pca = pca.fit_transform(y_embedded[0])
    # y_emb_plot = plot_point_cloud(y_embedded[0])
    # y_emb_plot.show()

    persistence = VietorisRipsPersistence(homology_dimensions=[1], n_jobs=6)
    dgms = persistence.fit_transform(y_embedded)
    # persistence.plot(dgms, sample=0)
    amp = Amplitude(n_jobs=6)
    amps = amp.fit_transform(dgms)
    lens = np.array([len(x) for x in l_wd])
    weights = lens/np.sum(lens)
    days[day] = np.average(amps, axis=0, weights=weights)[0]

plt.bar(*zip(*days.items()))
