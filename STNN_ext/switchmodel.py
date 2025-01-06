import pickle
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import snelweg.datatools2 as dt
import ydf
import matplotlib

matplotlib.use('Qt5Agg')
sns.set_theme(style="whitegrid")

input_cols = ['speed_RWS01_MONIBAS_0201hrr0413ra', 'speed_RWS01_MONIBAS_0201hrr0444ra',
              'speed_RWS01_MONIBAS_0201hrr0452ra', 'speed_RWS01_MONIBAS_0201hrr0478ra',
              'intensity_RWS01_MONIBAS_0201hrr0413ra', 'intensity_RWS01_MONIBAS_0201hrr0444ra',
              'intensity_RWS01_MONIBAS_0201hrr0452ra', 'intensity_RWS01_MONIBAS_0201hrr0478ra',
              'speed_GEO0K_K_RWSTI357132', 'speed_RWS01_MONIBAS_0131hrr0150ra',
              'speed_RWS01_MONIBAS_0131hrr0162ra', 'speed_RWS01_MONIBAS_0131hrr0171ra',
              'speed_RWS01_MONIBAS_0150vwg0627ra', 'speed_RWS01_MONIBAS_0160vwy0235ra',
              'speed_RWS01_MONIBAS_0161hrl0235ra', 'speed_RWS01_MONIBAS_0161hrl0260ra',
              'speed_RWS01_MONIBAS_0201hrr0235ra', 'speed_RWS01_MONIBAS_0201hrr0257ra',
              'speed_RWS01_MONIBAS_0201hrr0272ra', 'speed_RWS01_MONIBAS_0201hrr0295ra',
              'speed_RWS01_MONIBAS_0201hrr0363ra', 'intensity_GEO0K_K_RWSTI357132',
              'intensity_RWS01_MONIBAS_0131hrr0150ra', 'intensity_RWS01_MONIBAS_0131hrr0162ra',
              'intensity_RWS01_MONIBAS_0131hrr0171ra', 'intensity_RWS01_MONIBAS_0150vwg0627ra',
              'intensity_RWS01_MONIBAS_0160vwy0235ra', 'intensity_RWS01_MONIBAS_0161hrl0235ra',
              'intensity_RWS01_MONIBAS_0161hrl0260ra', 'intensity_RWS01_MONIBAS_0201hrr0235ra',
              'intensity_RWS01_MONIBAS_0201hrr0257ra', 'intensity_RWS01_MONIBAS_0201hrr0272ra',
              'intensity_RWS01_MONIBAS_0201hrr0295ra', 'intensity_RWS01_MONIBAS_0201hrr0363ra',
              'norm_timestamp']
input_cols_weekdays = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday']


def bigdataplot(data, plot_date, cols=None):
    # plot_date = '2023-10-13'  # yyyy-mm-dd
    if cols is None:
        cols = ['fuse']
    col_names = {'switch': 'Switch', 'stnn': 'STNN forward', 'stnn_rev': 'STNN backward',
                 'speed_RWS01_MONIBAS_0201hrr0413ra': 'Naive', 'fuse': 'Fuse'}
    date_df = data[data.index.map(lambda x: str(x)[:10]) == plot_date]
    fig = plt.figure(figsize=(8, 6))
    x = pd.to_datetime(date_df.index)
    plt.plot(x, date_df['check'].to_numpy(), label='True value')
    for col in cols:
        plt.plot(x, date_df[col].to_numpy(), label=col_names[col])
    plt.xlabel("Time (day hh:mm)")
    plt.ylabel("Speed (km/h)")
    # plt.title(plot_date)
    plt.ylim(0, 120)
    plt.xlim(x[0], x[-1])
    plt.legend()


if __name__ == "__main__":
    target = 'speed_RWS01_MONIBAS_0201hrr0413ra'
    with open('STNN_master/data/A20Gouda_stnnfull_L2.pkl', 'rb') as f:
        data = pickle.load(f)
    # original = dt.NDW_handler('snelweg/ochtend_alles/A20Gouda_alles_ochtend.csv')
    for i in range(len(data.dfs)):
        if 'check' in data.dfs[i].columns:
            assert np.all(np.abs(data.dfs[i][target].to_numpy()[10:] - data.dfs[i]['check'].to_numpy()[:-10]) < 1e-4)
    big_data = pd.concat([df for df in data.dfs if {'stnn', 'stnn_rev', 'check'} <= set(df.columns)], axis=0)
    big_data['time'] = big_data.index.map(lambda x: str(x)[11:])
    h = big_data.groupby(['time', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']).mean()
    big_data['hm'] = (big_data[['time', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']]
                      .apply(lambda x: h.loc[x['time'], x['monday'], x['tuesday'],
                             x['wednesday'], x['thursday'], x['friday']]['check'], axis=1))
    print(len(big_data))
    # hist_mean = big_data.groupby(big_data.norm_timestamp.round(9))
    # hist_mean.mean()
    excl_hm = True
    excl_naive = True
    forward_err = np.abs(big_data['stnn'] - big_data['check'])**1
    backward_err = np.abs(big_data['stnn_rev'] - big_data['check'])**1
    naive_err = np.abs(big_data[target] - big_data['check'])**1 + (1e6)*excl_naive
    hm_err = np.abs(big_data['hm'] - big_data['check'])**1 + (1e6)*excl_hm
    best_model = np.argmin(np.vstack([forward_err, backward_err, naive_err, hm_err]), axis=0)
    pre_combo_naming = {0: 'stnn', 1: 'stnn_rev', 2: target, 3: 'hm'}
    readable_naming = {0: 'STNN forward', 1: 'STNN backward', 2: 'Naive', 3: 'Historic mean', 4: 'Switch', 5: 'Fuse'}
    big_data = big_data.assign(best_model=best_model)
    decision_cols = input_cols + input_cols_weekdays + ['best_model']# + ['stnn', 'stnn_rev']
    decision_data = big_data[decision_cols]

    mae_list, mse_list, rf_acc_list = [], [], []
    train_mae_list, train_mse_list = [], []
    for i in tqdm(range(1)):

        shuffled_data = big_data#.sample(frac=1)
        shuffled_decision_data = shuffled_data[decision_cols]
        init = 0
        split_q = .85

        train_decision_data = shuffled_decision_data[int(init * len(shuffled_decision_data)):int(split_q * len(shuffled_decision_data))]
        test_decision_data = shuffled_decision_data[int(split_q * len(shuffled_decision_data)):]
        templates = ydf.RandomForestLearner.hyperparameter_templates()
        train_data = shuffled_data[:int(split_q * len(shuffled_decision_data))]
        test_data = shuffled_data[int(split_q * len(shuffled_decision_data)):]
        decision_model = ydf.RandomForestLearner(label="best_model", **templates['benchmark_rank1v1']).train(
            train_decision_data, verbose=0)


        def vecdot(a, b):
            return np.array([np.dot(a[j], b[j]) for j in range(len(a))])

        fuse_pred_vecs = decision_model.predict(decision_data)
        if len(fuse_pred_vecs.shape)==1:
            print("hoi")
            fuse_pred_vecs = np.stack([1-fuse_pred_vecs, fuse_pred_vecs]).T
            fuse_pred_vecs = np.array([fuse_pred_vecs[:, 0], fuse_pred_vecs[:, 1], np.repeat(0, fuse_pred_vecs.shape[0]), np.repeat(0, fuse_pred_vecs.shape[0])]).T
        else:
            lcs = np.array([int(x) for x in decision_model.label_classes()])
            sort = np.argsort(lcs)
            # assert np.all(lcs[sort] == np.arange(4))
            fuse_pred_vecs = fuse_pred_vecs[:, sort]
        switch_pred_indices = np.argmax(fuse_pred_vecs, axis=1)
        switch_pred = np.array([big_data.iloc[i][pre_combo_naming[switch_pred_indices[i]]] for i in range(len(big_data))])
        fuse_pred = vecdot(fuse_pred_vecs, big_data[pre_combo_naming.values()].to_numpy())
        big_data['switch'] = switch_pred
        big_data['fuse'] = fuse_pred
        big_data['switch_model_index'] = switch_pred_indices
        test_data = test_data.join(big_data[['switch', 'fuse', 'switch_model_index']])
        train_data = train_data.join(big_data[['switch', 'fuse', 'switch_model_index']])


        def mae(forw, backw, naive, historic, switch, fuse, y):
            return np.mean(
                np.array([np.abs(forw - y), np.abs(backw - y), np.abs(naive - y), np.abs(historic - y), np.abs(switch - y), np.abs(fuse - y)]),
                axis=1)


        def mse(forw, backw, naive, historic, switch, fuse, y):
            return np.mean(
                np.array([(forw - y) ** 2, (backw - y) ** 2, (naive - y) ** 2, (historic - y) ** 2, (switch - y) ** 2, (fuse - y) ** 2]),
                axis=1)


        test_mae = mae(test_data['stnn'], test_data['stnn_rev'], test_data[target], test_data['hm'], test_data['switch'], test_data['fuse'], test_data['check'])
        test_mse = mse(test_data['stnn'], test_data['stnn_rev'], test_data[target], test_data['hm'], test_data['switch'], test_data['fuse'], test_data['check'])
        train_mae = mae(train_data['stnn'], train_data['stnn_rev'], train_data[target], train_data['hm'], train_data['switch'], train_data['fuse'], train_data['check'])
        train_mse = mse(train_data['stnn'], train_data['stnn_rev'], train_data[target], train_data['hm'], train_data['switch'], train_data['fuse'], train_data['check'])
        test_res = "\n".join(
            [f"{readable_naming[i]}: {test_mae[i]:.1f} MAE, {test_mse[i]:.1f} MSE" for i in range(len(test_mse))])
        print(test_res)
        mae_list.append(test_mae)
        mse_list.append(test_mse)
        train_mae_list.append(train_mae)
        train_mse_list.append(train_mse)

    final_mae = np.mean(np.array(mae_list), axis=0)
    final_mse = np.mean(np.array(mse_list), axis=0)
    final_mae_sd = np.sqrt(np.var(np.array(mae_list), axis=0))
    final_mse_sd = np.sqrt(np.var(np.array(mse_list), axis=0))

    final_train_mae = np.mean(np.array(train_mae_list), axis=0)
    final_train_mse = np.mean(np.array(train_mse_list), axis=0)
    final_train_mae_sd = np.sqrt(np.var(np.array(train_mae_list), axis=0))
    final_train_mse_sd = np.sqrt(np.var(np.array(train_mse_list), axis=0))
    test_res = "\n".join(
            [f"{readable_naming[i]}: {final_mae[i]:.1f}+-{final_mae_sd[i]:.1f} MAE, {final_mse[i]:.1f}+-{final_mse_sd[i]:.0f} MSE" for i in range(len(final_mse))])
    train_res = "\n".join(
            [f"{readable_naming[i]}: {final_train_mae[i]:.1f}+-{final_train_mae_sd[i]:.1f} MAE, {final_train_mse[i]:.1f}+-{final_train_mse_sd[i]:.0f} MSE" for i in range(len(final_mse))])
    print('Training')
    print(train_res)
    print('Testing')
    print(test_res)
    print(np.mean(np.min(np.vstack([np.abs(train_data['stnn_rev']-train_data['check']),
                                    np.abs(train_data['stnn']-train_data['check']),
                                    np.abs(train_data['hm']-train_data['check']),
                                    np.abs(train_data[target]-train_data['check'])]), axis=0)))
    print(np.mean(np.min(np.vstack([np.abs(train_data['stnn_rev']-train_data['check'])**2,
                                    np.abs(train_data['stnn']-train_data['check'])**2,
                                    np.abs(train_data['hm']-train_data['check'])**2,
                                    np.abs(train_data[target]-train_data['check'])**2]), axis=0)))
    print(np.mean(np.min(np.vstack(
        [np.abs(test_data['stnn_rev'] - test_data['check']), np.abs(test_data['stnn'] - test_data['check']),
         np.abs(test_data['hm'] - test_data['check']), np.abs(test_data[target] - test_data['check'])]), axis=0)))
    print(np.mean(np.min(np.vstack(
        [np.abs(test_data['stnn_rev'] - test_data['check']) ** 2, np.abs(test_data['stnn'] - test_data['check']) ** 2,
         np.abs(test_data['hm'] - test_data['check']) ** 2, np.abs(test_data[target] - test_data['check']) ** 2]),
                         axis=0)))
    maybe_dates = ['2024-'+x for x in ['04-09', '04-15', '04-17', '04-22', '04-23', '05-13', '05-16', '05-22']]
    for date in maybe_dates[0:1]:
        print(date)
        bigdataplot(big_data, date)
        plt.show(block=True)
