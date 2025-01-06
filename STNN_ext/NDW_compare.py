import pickle
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Qt5Agg')
sns.set_theme(style='whitegrid')

if __name__ == '__main__':
    # with open('STNN_master/data/ts_flow_in_sample_ndw.pkl', 'rb') as f:
    #     selfless = False
    #     tar = 'Flow'
    #     ins = pickle.load(f)
    # with open('STNN_master/data/ts_flow_out_of_sample_ndw.pkl', 'rb') as f:
    #     oos = pickle.load(f)
    # with open('STNN_master/data_na_selfless/ts_in_sample_ndw.pkl', 'rb') as f:
    #     selfless = True
    #     tar = 'Speed'
    #     ins = pickle.load(f)
    # with open('STNN_master/data_na_selfless/ts_out_of_sample_ndw.pkl', 'rb') as f:
    #     oos = pickle.load(f)
    # with open('STNN_master/data_na_selfless/ts_flow_in_sample_ndw.pkl', 'rb') as f:
    #     selfless = True
    #     tar = 'Flow'
    #     ins = pickle.load(f)
    # with open('STNN_master/data_na_selfless/ts_flow_out_of_sample_ndw.pkl', 'rb') as f:
    #     oos = pickle.load(f)
    # with open('STNN_master/data/ts_in_sample_ndw.pkl', 'rb') as f:
    #     selfless = False
    #     tar = 'Speed'
    #     ins = pickle.load(f)
    # with open('STNN_master/data/ts_out_sample_ndw.pkl', 'rb') as f:
    #     oos = pickle.load(f)
    # with open('STNN_master/data/ts_in_sample_ndw_rev.pkl', 'rb') as f:
    #     selfless = False
    #     tar = 'Speed rev'
    #     ins = pickle.load(f)
    # with open('STNN_master/data/ts_out_of_sample_ndw_rev.pkl', 'rb') as f:
    #     oos = pickle.load(f)
    # with open('STNN_master/data/ts_flow_in_sample_ndw_rev.pkl', 'rb') as f:
    #     selfless = False
    #     tar = 'Flow rev'
    #     ins = pickle.load(f)
    # with open('STNN_master/data/ts_flow_out_of_sample_ndw_rev.pkl', 'rb') as f:
    #     oos = pickle.load(f)
    with open('STNN_master/data/sl_flow_in_sample_ndw_rev.pkl', 'rb') as f:
        selfless = True
        tar = 'Flow rev'
        ins = pickle.load(f)
    with open('STNN_master/data/sl_flow_out_of_sample_ndw_rev.pkl', 'rb') as f:
        oos = pickle.load(f)

    ins_df = pd.concat(ins.dfs)
    train_ins_df = ins_df.iloc[:int(.7*len(ins_df))]
    test_ins_df = ins_df.iloc[int(.7*len(ins_df)):]
    oos_df = pd.concat(oos.dfs)
    train_oos_df = oos_df.iloc[:int(.7*len(oos_df))]
    test_oos_df = oos_df.iloc[int(.7*len(oos_df)):]
    ns = [13, 12, 11, 10]
    # names = [('base', ''), ('phys', '_phys'), ('dyn', '_dyn')]
    names = [('base', ''), ('phys', '_phys')]
    # names = [('dyn','_dyn')]
    oos_dict = {name[0]: np.array([np.abs(test_oos_df['out_' + str(n) + name[1]].to_numpy() -
                                          test_oos_df['check_' + str(n)].to_numpy()) for n in ns]) for name in names}
    ins_dict = {name[0]: np.array([np.abs(test_ins_df[f'in_{n}{name[1]}_{i//4}'].to_numpy() -
                                          test_ins_df['check_' + str(n)].to_numpy()) for i, n in enumerate(3 * ns)])
                for name in names}
    oos_train = {name[0]: np.array([np.abs(train_oos_df['out_' + str(n) + name[1]].to_numpy() -
                                           train_oos_df['check_' + str(n)].to_numpy()) for n in ns]) for name in names}
    ins_train = {name[0]: np.array([np.abs(train_ins_df[f'in_{n}{name[1]}_{i//4}'].to_numpy() -
                                           train_ins_df['check_' + str(n)].to_numpy()) for i, n in enumerate(3 * ns)])
                 for name in names}
    assert len(ins_dict['phys']) == 12
    oos_res = "\n".join([f'{name} mse: {np.mean(err**2): .3g}, mae: {np.mean(err): .3g}' for name, err in oos_dict.items()])
    ins_res = "\n".join([f'{name} mse: {np.mean(err**2): .3g}, mae: {np.mean(err): .3g}' for name, err in ins_dict.items()])
    oos_res_tr = "\n".join([f'{name} mse: {np.mean(err**2): .3g}, mae: {np.mean(err): .3g}' for name, err in oos_train.items()])
    ins_res_tr = "\n".join([f'{name} mse: {np.mean(err**2): .3g}, mae: {np.mean(err): .3g}' for name, err in ins_train.items()])
    print("======================================================")
    print("Training")
    print(oos_res_tr)
    print(ins_res_tr)
    print("======================================================")
    print("Testing")
    print(oos_res)
    print(ins_res)
    # 10 11 12 13 -> ins = oos + .4

    smoothing = 1
    oos_cols = [f'out_{n}{name[1]}' for n in ns for name in names] + [f'check_{n}' for n in ns]
    ins_cols = [f'in_{n}{name[1]}_{i//4}' for i, n in enumerate(3 * ns) for name in names] + [f'check_{n}' for n in ns]
    test_oos_smooth = test_oos_df[oos_cols].rolling(smoothing).mean().iloc[smoothing-1:]
    test_ins_smooth = test_ins_df[ins_cols].rolling(smoothing).mean().iloc[smoothing-1:]

    test_tar = 13
    ins_choice = 2
    fig, axs = plt.subplots(ncols=2)
    fig.suptitle(f'{tar} {'selfless' if selfless else ''}')
    # xax = pd.to_datetime(test_oos_smooth.index)
    axs[0].plot(test_oos_smooth[f'check_{test_tar}'].to_numpy(), label=f'True {test_tar}')
    axs[0].plot(test_oos_smooth[f'out_{test_tar}_dyn'].to_numpy(), label=f'OOS {test_tar}')
    axs[0].legend()
    axs[0].set_title(f'{np.mean(np.abs(test_oos_smooth[f'check_{test_tar}'].to_numpy()-test_oos_smooth[f'out_{test_tar}_dyn'].to_numpy())**2):.1e}')
    axs[1].plot(test_ins_smooth[f'check_{test_tar}'].to_numpy(), label=f'True {test_tar}')
    axs[1].plot(test_ins_smooth[f'in_{test_tar}_dyn_{ins_choice}'].to_numpy(), label=f'INS dyn {test_tar}')
    axs[1].plot(test_ins_smooth[f'in_{test_tar}_0'].to_numpy(), label=f'INS base {test_tar}')
    axs[1].set_title(f'{np.mean(np.abs(test_ins_smooth[f'check_{test_tar}'].to_numpy()-test_ins_smooth[f'in_{test_tar}_dyn_{ins_choice}'].to_numpy())**2):.1e}')
    axs[1].legend()
