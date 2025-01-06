import tensorflow as tf
import pickle
import numpy as np
import snelweg.datatools2 as dt
import pandas as pd
from tqdm import tqdm
import os
import importlib
importlib.reload(dt)

if __name__ == "__main__":
    scores = {'terugslag': [], 'rijnsweerd': [], 'zoeterwoude': []}
    weekend = True
    N_repeats = 100
    neurons = 1
    for loc in tqdm(scores):
        if loc == 'terugslag':
            sel = ['RWS01_MONIBAS_0201hrr0433ra', 'RWS01_MONIBAS_0201hrr0439ra', 'RWS01_MONIBAS_0201hrr0444ra',
                   'RWS01_MONIBAS_0201hrr0449ra', 'RWS01_MONIBAS_0201hrr0452ra', 'RWS01_MONIBAS_0201hrr0461ra']
            fp = 'terugslag_detail_maand'
        elif loc == 'rijnsweerd':
            sel = ['RWS01_MONIBAS_0271hrl0854ra', 'RWS01_MONIBAS_0271hrl0851ra', 'RWS01_MONIBAS_0271hrl0847ra',
                   'RWS01_MONIBAS_0271hrl0844ra', 'RWS01_MONIBAS_0271hrl0839ra', 'RWS01_MONIBAS_0271hrl0835ra',
                   'RWS01_MONIBAS_0271hrl0829ra_1', 'RWS01_MONIBAS_0271hrl0825ra']
            fp = 'rijnsweerd'
        elif loc == 'zoeterwoude':
            sel = ['RWS01_MONIBAS_0041hrl0391ra', 'RWS01_MONIBAS_0041hrl0388ra', 'RWS01_MONIBAS_0041hrl0384ra',
                   'RWS01_MONIBAS_0041hrl0380ra', 'RWS01_MONIBAS_0041hrl0375ra', 'RWS01_MONIBAS_0041hrl0371ra',
                   'RWS01_MONIBAS_0041hrl0367ra', 'RWS01_MONIBAS_0041hrl0362ra']
            fp = 'zoeterwoude_dorp'
        fp = f'/{fp}{'_weekend' if weekend else '_werkweek'}.csv'
        base_path = f'tda_los/{loc}{'_weekend' if weekend else '_werkweek'}'

        if 'combined_data.csv' in os.listdir(base_path) and 'og.pkl' in os.listdir(base_path):
            print('Using pre-existing data file')
            bigdf = pd.read_csv(base_path + '/combined_data.csv')
            with open(base_path + '/og.pkl', 'rb') as f:
                og = pickle.load(f)
        else:
            og = dt.NDW_handler(base_path + fp, loc_list=sel, smoothing=1)
            nn_data = og.export(min_length=0)
            for file in os.listdir(base_path + '/dgms'):
                with open(base_path + '/dgms/' + file, 'rb') as f:
                    data = pickle.load(f)
                    features = [[x for d, x in dgm if d == 1] for dgm in data]
                    lifetimes = [[death - birth for birth, death in x] for x in features]
                    summarization = np.array([sum(y) for y in lifetimes]) / 6  # for normalization
                    i = int("".join([n for n in file[1:3] if n.isnumeric()]))
                    print(file)
                    print(i)
                    nn_data[i] = nn_data[i].assign(summarization=summarization)
            bigdf = pd.concat(nn_data)
            with open(base_path + '/og.pkl', 'wb') as f:
                pickle.dump(og, f)
            bigdf.to_csv(base_path + '/combined_data.csv')
        conv_width = 5
        label_width = 1
        horizon = 10
        cp1 = .7
        cp2 = .85
        label = 'norm_speed_' + sel[0]
        if console_mode == 'summary':
            bigdf = bigdf[['summarization', label]]
        elif console_mode == 'self':
            bigdf = bigdf[[label, 'norm_intensity_' + sel[0]]]
        elif console_mode == 'both':
            bigdf = bigdf.drop(['start_meetperiode'], axis=1)
        elif console_mode == 'raw':
            bigdf = bigdf.drop(['summarization','start_meetperiode'], axis=1)
        window = dt.TimeseriesWindowGenerator(conv_width, label_width, horizon,
                                              [bigdf.iloc[:int(cp1*len(bigdf))]],
                                              [bigdf.iloc[int(cp1*len(bigdf)):int(cp2*len(bigdf))]],
                                              bigdf.iloc[int(cp2*len(bigdf)):],
                                              label_columns=[label])

        for i in tqdm(range(N_repeats)):
            gru = tf.keras.Sequential([
                tf.keras.layers.GRU(neurons, return_sequences=False),
                tf.keras.layers.Dense(neurons, activation='relu'),
                tf.keras.layers.Dense(units=1)
            ])

            histgru = dt.compile_and_fit(gru, window, max_epochs=50, patience=10, verbose=0)
            # outputs = np.array([x[0] for x in gru.predict(window.test)]) * og.sd_speed + og.mean_speed
            # truths = np.array([y.numpy().flatten()[0] for x in window.test for y in x[1]]) * og.sd_speed + og.mean_speed
            mse = gru.evaluate(window.test)[0]*(og.sd_speed**2)
            scores[loc].append(mse)
    with open(f'tda_los/objects/weekend_res{neurons}_{console_mode}.pkl', 'wb') as f:
        pickle.dump(scores, f)
        # print(np.var(bigdf[label].to_numpy()[int(cp2*len(bigdf)):])*(og.sd_speed**2))
