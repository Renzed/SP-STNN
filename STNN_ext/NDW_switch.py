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


def rf(train, test, cols, model_cols):
    train_data = train[cols+['best_model']]
    test_data = test[cols+['best_model']+model_cols]
    templates = ydf.RandomForestLearner.hyperparameter_templates()
    decision_model = ydf.RandomForestLearner(label='best_model', **templates['benchmark_rank1v1']).train(
        train_data, verbose=0)

    def vecdot(a, b):
        return np.array([np.dot(a[j], b[j]) for j in range(len(a))])

    fuse_pred_vecs = decision_model.predict(test_data.drop(['best_model']+model_cols, axis=1))
    lcs = decision_model.label_classes()
    print(lcs)
    switch_pred_indices = np.argmax(fuse_pred_vecs, axis=1)
    switch_pred = np.array([test_data.iloc[i][lcs[switch_pred_indices[i]]] for i in range(len(test_data))])
    fuse_pred = vecdot(fuse_pred_vecs, test_data[lcs].to_numpy())
    return switch_pred, fuse_pred, test['check']


def input_df(t, mod, name_base, sl=False):
    if mod == '_dyn':
        name_add = '_dyn'
        file_add = '_phys'
    elif mod == '_phys':
        name_add = ''
        file_add = '_phys'
    elif mod == '':
        name_add = file_add = ''
    else:
        raise ValueError('Mode not supported')
    if 'rev' in name_base:
        ra = 'rev_'
    else:
        ra = ''
    if sl:
        torch_data = np.concatenate([np.load(f'STNN_master/data/fix/{name_base}{name_add}/part{i}/combined_all_samples_25_5_per5_10_vtar{t}{file_add}_selfless.npy') for i in range(16)])
    else:
        torch_data = np.concatenate([np.load(f'STNN_master/data/fix/{name_base}{name_add}/part{i}/combined_all_samples_25_5_10_vtar{t}{file_add}.npy') for i in range(16)])
    dec_in = np.transpose(np.concatenate(np.transpose(torch_data[:, :, -1, :], [2, 1, 0]), axis=0))
    return pd.DataFrame(dec_in, columns=[f'{ra}{pqt}_{i}' for i in range(num_nearby)]+
                                        [f'{ra}{'intensity' if pqt == 'speed' else 'speed'}_{i}' for i in range(num_nearby)]+
                                        [f'{ra}dist_{i}' for i in range(num_nearby)])


def rename_dict(mode, count, n, rev=False, oos=False):
    if oos:
        b = f'out_{n}{mode}'
    else:
        b = f'in_{n}{mode}_{count}'
    if rev:
        return {b: 'pred_rev', f'check_{n}': 'check', f'hm_{n}': 'hm', f'naive_{n}': 'naive'}
    else:
        return {b: 'pred', f'check_{n}': 'check', f'hm_{n}': 'hm', f'naive_{n}': 'naive'}


def best_gen(df, cols, tar):
    amdf = pd.DataFrame(np.array([np.abs(df[c]-df[tar]) for c in cols]).T, columns=cols)
    am = np.argmin(amdf[cols], axis=1)
    # assert len(np.unique(am)) == len(cols)
    return np.array([cols[i] for i in am])


def get_scores(rfres, ordf, cols):
    amdf = pd.DataFrame(np.array([np.abs(ordf[c] - ordf['check']) for c in cols]).T, columns=cols)
    fuse_errs = np.abs(rfres[1]-rfres[2])
    switch_errs = np.abs(rfres[0]-rfres[2])
    return amdf.assign(fuse=fuse_errs.to_numpy(), switch=switch_errs.to_numpy())

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
    mode = '_dyn'
    revmode = ''
    pqt = 'intensity'
    target_node = 10
    num_nearby = 10
    hor = 5
    cp1 = 0
    cp2 = .7
    selfless = True
    slfac = -10
    in_width = 25
    filebase = 'groningen_flow' if pqt == 'intensity' else 'groningen'
    hm_cols = ['time', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']
    dec_cols = ([f'speed_{i}' for i in range(num_nearby)] +
                [f'intensity_{i}' for i in range(num_nearby)] +
                [f'dist_{i}' for i in range(num_nearby)] +
                [f'rev_speed_{i}' for i in range(num_nearby)] +
                [f'rev_intensity_{i}' for i in range(num_nearby)] +
                [f'rev_dist_{i}' for i in range(num_nearby)] + hm_cols)
    model_contenders = ['pred', 'pred_rev', 'naive']

    templates = ydf.RandomForestLearner.hyperparameter_templates()

    indiv_in_sample_res = []
    in_sample_res = []
    out_of_sample_res = []

    # data = dt.NDW_handler('STNN_master/data/groningen_near_far.csv')
    with open('STNN_master/data/fix/rev_groningen/labels.pkl', 'rb') as f:
        labs = list(pickle.load(f).values())

    with open(f'STNN_master/data/{'ts' if not selfless else 'sl'}{'_flow' if pqt == 'intensity' else ''}_in_sample_ndw.pkl', 'rb') as f:
        insample = pd.concat(pickle.load(f).dfs)
        insample['time'] = insample.index.map(lambda x: str(x)[11:])
        h = insample.iloc[:int(cp2*len(insample))].groupby(['time', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']).mean()
        for i in [10, 11, 12, 13]:
            insample[f'hm_{i}'] = (insample[['time', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']]
                                   .apply(lambda x: h.loc[x['time'], x['monday'], x['tuesday'],
                                          x['wednesday'], x['thursday'], x['friday']][f'check_{i}'], axis=1))
            insample[f'naive_{i}'] = insample[f'speed_{labs[i]}'] * slfac
    with open(f'STNN_master/data/{'ts' if not selfless else 'sl'}{'_flow' if pqt == 'intensity' else ''}_out_of_sample_ndw.pkl', 'rb') as f:
        oosample = pd.concat(pickle.load(f).dfs)
        oosample['time'] = oosample.index.map(lambda x: str(x)[11:])
        h = oosample.iloc[:int(cp2*len(oosample))].groupby(['time', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']).mean()
        for i in [10, 11, 12, 13]:
            oosample[f'hm_{i}'] = (oosample[['time', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']]
                                   .apply(lambda x: h.loc[x['time'], x['monday'], x['tuesday'],
                                          x['wednesday'], x['thursday'], x['friday']][f'check_{i}'], axis=1))
            oosample[f'naive_{i}'] = oosample[f'speed_{labs[i]}'] * slfac
    with open(f'STNN_master/data/{'ts' if not selfless else 'sl'}{'_flow' if pqt == 'intensity' else ''}_in_sample_ndw_rev.pkl', 'rb') as f:
        insamplerev = pd.concat(pickle.load(f).dfs)
        insamplerev['time'] = insamplerev.index.map(lambda x: str(x)[11:])
        h = insamplerev.iloc[:int(cp2*len(insamplerev))].groupby(['time', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']).mean()
        for i in [10, 11, 12, 13]:
            insamplerev[f'hm_{i}'] = (insamplerev[['time', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']]
                                      .apply(lambda x: h.loc[x['time'], x['monday'], x['tuesday'],
                                             x['wednesday'], x['thursday'], x['friday']][f'check_{i}'], axis=1))
            insamplerev[f'naive_{i}'] = insamplerev[f'speed_{labs[i]}'] * slfac
    with open(f'STNN_master/data/{'ts' if not selfless else 'sl'}{'_flow' if pqt == 'intensity' else ''}_out_of_sample_ndw_rev.pkl', 'rb') as f:
        oosamplerev = pd.concat(pickle.load(f).dfs)
        oosamplerev[f'time'] = oosamplerev.index.map(lambda x: str(x)[11:])
        h = oosamplerev.iloc[:int(cp2*len(oosamplerev))].groupby(['time', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']).mean()
        for i in [10, 11, 12, 13]:
            oosamplerev[f'hm_{i}'] = (oosamplerev[['time', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday']]
                                      .apply(lambda x: h.loc[x['time'], x['monday'], x['tuesday'],
                                             x['wednesday'], x['thursday'], x['friday']][f'check_{i}'], axis=1))
            oosamplerev[f'naive_{i}'] = oosamplerev[f'speed_{labs[i]}'] * slfac
    # original = dt.NDW_handler('snelweg/ochtend_alles/A20Gouda_alles_ochtend.csv')

    for basenode in [10, 11, 12, 13]:
        if basenode == 13:
            grouping = [10, 11, 12]
        elif basenode == 12:
            grouping = [11, 10, 13]
        elif basenode == 11:
            grouping = [12, 13, 10]
        elif basenode == 10:
            grouping = [13, 12, 11]

        indfs = [pd.concat([insample[[f'in_{n}{mode}_{i}', f'check_{n}', f'hm_{n}', f'naive_{n}']+hm_cols].rename(columns=rename_dict(mode, i, n)), input_df(n, mode, filebase, sl=selfless).set_index(insample.index)], axis=1) for i, n in enumerate(grouping)]
        indfsr = [pd.concat([insamplerev[[f'in_{n}{revmode}_{i}', f'check_{n}', f'hm_{n}', f'naive_{n}']+hm_cols].rename(columns=rename_dict(revmode, i, n, rev=True)), input_df(n, revmode, 'rev_'+filebase, sl=selfless).set_index(insamplerev.index)], axis=1) for i, n in enumerate(grouping)]
        odff = pd.concat([oosample[[f'out_{basenode}{mode}', f'check_{basenode}', f'hm_{basenode}', f'naive_{basenode}']+hm_cols].rename(columns=rename_dict(mode, 0, basenode, oos=True)), input_df(basenode, mode, filebase, sl=selfless).set_index(oosample.index)], axis=1)
        odfr = pd.concat([oosamplerev[[f'out_{basenode}{revmode}', f'check_{basenode}', f'hm_{basenode}', f'naive_{basenode}']+hm_cols].rename(columns=rename_dict(revmode, 0, basenode, rev=True, oos=True)), input_df(basenode, revmode, 'rev_'+filebase, sl=selfless).set_index(oosample.index)], axis=1)

        idfl = [indfs[i].merge(indfsr[i]) for i in range(len(indfs))]
        odf = odff.merge(odfr)

        idf_train = pd.concat([df.iloc[int(cp1*len(df)):int(cp2*len(df))].reset_index(drop=True) for df in idfl], axis=0, ignore_index=True)
        idf_test = pd.concat([df.iloc[int(cp2*len(df)):].reset_index(drop=True) for df in idfl], axis=0, ignore_index=True)
        odf_train = odf.iloc[int(cp1*len(odf)):int(cp2*len(odf))]
        odf_test = odf.iloc[int(cp2*len(odf)):]

        idf_train['best_model'] = best_gen(idf_train, model_contenders, 'check')
        idf_test['best_model'] = best_gen(idf_test, model_contenders, 'check')
        odf_train['best_model'] = best_gen(odf_train, model_contenders, 'check')
        odf_test['best_model'] = best_gen(odf_test, model_contenders, 'check')

        indiv_train = [df.iloc[int(cp1*len(df)):int(cp2*len(df))].reset_index(drop=True) for df in idfl]
        indiv_test = [df.iloc[int(cp2*len(df)):].reset_index(drop=True) for df in idfl]
        for i in range(len(indiv_train)):
            indiv_train[i]['best_model'] = best_gen(indiv_train[i], model_contenders, 'check')
            indiv_test[i]['best_model'] = best_gen(indiv_test[i], model_contenders, 'check')
            ires = rf(indiv_train[i], indiv_test[i], dec_cols, model_contenders)
            indiv_in_sample_res.append(get_scores(ires, indiv_test[i], model_contenders))

        idf_test_res = rf(idf_train, idf_test, dec_cols, model_contenders)
        odf_test_res = rf(odf_train, odf_test, dec_cols, model_contenders)

        in_sample_res.append(get_scores(idf_test_res, idf_test, model_contenders))
        out_of_sample_res.append(get_scores(odf_test_res, odf_test, model_contenders))

    print(len(indiv_in_sample_res))
    indiv_in_sample_res = pd.concat(indiv_in_sample_res, axis=0)
    in_sample_res = pd.concat(in_sample_res, axis=0)
    out_of_sample_res = pd.concat(out_of_sample_res, axis=0)
    #
    # maybe_dates = ['2024-'+x for x in ['04-09', '04-15', '04-17', '04-22', '04-23', '05-13', '05-16', '05-22']]
    # for date in maybe_dates[0:1]:
    #     print(date)
    #     bigdataplot(big_data, date)
    #     plt.show(block=True)
