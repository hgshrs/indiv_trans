import pandas as pd
import os
import numpy as np
import matplotlib.pylab as plt
import sys

def plot_seq(fg, df, agent=0, n_steps=1, block=0):
    fg.clf()
    seqdf = df.query('agent == {} & n_steps == {} & block == {}'.format(agent, n_steps, block))
    cmap_tab = plt.get_cmap('tab10')
    ax = fg.add_subplot(1, 1, 1)
    yticks = []
    yticklabels = []
    for ss in range(n_steps):
        # ax = fg.add_subplot(n_steps, 1, ss + 1)
        _df = seqdf[seqdf['step'] == ss]
        ax.plot(_df[_df['act'] == 0]['act'] - .05 + ss, 'o', mfc=cmap_tab(0), mec=None)
        ax.plot(_df[_df['act'] == 1]['act'] - .95 + ss, 'o', mfc=cmap_tab(1), mec=None)
        ax.plot(_df[_df['rew'] > 0]['rew'] + n_steps - 1, 'r*')
        yticks.append(ss)
        yticklabels.append('Step {}'.format(ss))
    yticks.append(n_steps)
    yticklabels.append('Reward')
    ax.set_yticks(yticks, yticklabels)

def out_step_ave(df):
    set_n_steps = np.sort(df['n_steps'].unique())
    ave_rew = np.zeros(len(set_n_steps))
    for nss, n_steps in enumerate(set_n_steps):
        ave_rew[nss] = df[df['n_steps'] == n_steps]['rew'].mean() * n_steps
    return set_n_steps, ave_rew

if __name__=='__main__':
    dataset_path = 'data/bhv_mdp.csv'
    data_dir = 'bhv_mdp'
    csvpaths = [f.name for f in os.scandir(data_dir) if not f.name.startswith('.')]
    df = pd.DataFrame()
    agent_idx = 0
    trans = [
            's00a0s10', 's00a1s10', 's00a0s11', 's00a1s11',
            's10a0s20', 's10a1s20', 's10a0s21', 's10a1s21', 's11a0s20', 's11a1s20', 's11a0s21', 's11a1s21',
            's20a0s30', 's20a1s30', 's20a0s31', 's20a1s31', 's21a0s30', 's21a1s30', 's21a0s31', 's21a1s31',
            ]
    for pp, path in enumerate(csvpaths):
        if (path[-3:] == 'csv'):
            print(path)
            sessdf = pd.read_csv('{}/{}'.format(data_dir, path))
            start_idxs = sessdf[sessdf['n_steps'] > 0].index
            trial_idxs = np.sort(sessdf[sessdf['trials.thisN'] >= 0]['trials.thisN'].unique().astype(int))
            n_trials = len(trial_idxs)
            set_n_blocks = {}
            for blk_idx, start_loc in enumerate(start_idxs):
                df_ = sessdf.loc[start_loc:(start_loc + n_trials - 1)]
                n_steps = int(df_.iloc[0]['n_steps'])
                try:
                    set_n_blocks[n_steps] = set_n_blocks[n_steps] + 1
                except:
                    set_n_blocks[n_steps] = 0
                # print(start_loc, n_steps)
                for trial_idx in range(n_trials):
                    sr = df_.iloc[trial_idx]
                    blk_dict = {}
                    for ss in range(n_steps):
                        blk_dict['agent'] = [agent_idx]
                        blk_dict['id'] = [sr['participant']]
                        blk_dict['n_steps'] = [n_steps]
                        blk_dict['block'] = [set_n_blocks[n_steps]]
                        blk_dict['episode'] = [trial_idx]
                        blk_dict['step'] = [ss]
                        blk_dict['state0'] = [ss]
                        try:
                            blk_dict['state1'] = [int(sr['s{}'.format(ss)])]
                        except:
                            blk_dict['state1'] = [0]
                        for _trans in trans:
                            blk_dict[_trans] = sr[_trans]
                        blk_dict['act'] = [int(sr['act{}'.format(ss)])]
                        if ss  == (n_steps - 1):
                            blk_dict['rew'] = [sr['rew']]
                        else:
                            blk_dict['rew'] = [0]
                        rta = np.array([sr['res0.rt'], sr['res1.rt'], sr['res2.rt']])
                        blk_dict['rt'] = rta[np.logical_not(np.isnan(rta))].max() + .5
                        df = pd.concat([df, pd.DataFrame(blk_dict)], axis=0)
            agent_idx += 1
    df = df.reset_index(drop=True)
    approved_ids = df['id'].unique()
    pd.DataFrame({'id':approved_ids}).to_csv('tmp/approved_ids.csv', index=False)
    df.to_csv(dataset_path, index=False)

    df = pd.read_csv(dataset_path)
    print('Total IDs: {}'.format(len(df['id'].unique())))


    # ===============================
    # Visualize
    # ===============================
    fg = plt.figure(3)
    plot_seq(fg, df, agent=19, n_steps=3, block=0)

    plt.figure(4).clf()
    plt.bar(*out_step_ave(df))
    plt.ylim([.3, .7])

    df_seq = pd.DataFrame()
    for uu, id_ in enumerate(df['id'].unique()):
        dfu = df[df['id'] == id_]
        for n_steps in dfu['n_steps'].unique():
            dfus = dfu[dfu['n_steps'] == n_steps]
            dict_step = {
                    'id': id_,
                    'n_steps': n_steps,
                    'rew': dfus['rew'].mean() * n_steps,
                    'rt_min': dfus['rt'].min(),
                    'rt_max': dfus['rt'].max(),
                    'rt_mean': dfus['rt'].mean(),
                    'rt_std': dfus['rt'].std(),
                    'act_mean': dfus['act'].mean(),
                    }
            df_seq = pd.concat([df_seq, pd.Series(dict_step).to_frame().T], axis=0)

    plt.figure(1).clf()
    plt.figure(2).clf()
    plt.figure(3).clf()
    for ss, n_steps in enumerate(df_seq['n_steps'].unique()):
        plt.figure(1)
        plt.subplot(3, 1, ss+1)
        plt.hist(df_seq[df_seq['n_steps'] == n_steps]['rew'])
        plt.figure(2)
        plt.subplot(3, 1, ss+1)
        plt.hist(df_seq[df_seq['n_steps'] == n_steps]['rt_mean'])
        plt.figure(3)
        plt.subplot(3, 1, ss+1)
        plt.hist(df_seq[df_seq['n_steps'] == n_steps]['rt_std'], bins=100)

    rew_iqr = {}
    rew_outunder = {}
    rew_outover = {}
    for n_steps in dfu['n_steps'].unique():
        rew_iqr[n_steps] = df_seq[df_seq['n_steps'] == n_steps]['rew'].quantile(.75) - df_seq[df_seq['n_steps'] == n_steps]['rew'].quantile(.25)
        rew_outunder[n_steps] = df_seq[df_seq['n_steps'] == n_steps]['rew'].quantile(.25) - 1.5 * rew_iqr[n_steps]
        rew_outover[n_steps] = df_seq[df_seq['n_steps'] == n_steps]['rew'].quantile(.75) + 1.5 * rew_iqr[n_steps]

    reject_ids = []
    for uu, id_ in enumerate(df['id'].unique()):
        for n_steps in dfu['n_steps'].unique():
            ser = df_seq.query('id == @id_ & n_steps == @n_steps').iloc[0]
            if (ser['act_mean'] < .15) or (ser['act_mean'] > .85) or (ser['rt_mean'] < .75) or (ser['rt_std'] > 10.) or (ser['rew'] < rew_outunder[n_steps]):
                reject_ids.append(id_)

    luus = list(set(df['id'].unique()).difference(set(reject_ids)))
    print('Remaining IDs: {}'.format(len(luus)))
    bonus = []
    bonus_sum = 100
    new_df = pd.DataFrame()
    for uu, id_ in enumerate(luus):
        _new_df = df[df['id'] == id_]
        _new_df = _new_df.drop(columns='agent')
        _new_df['agent'] = [uu] * len(_new_df)
        new_df = pd.concat([new_df, _new_df], axis=0)
        bonus.append(df[df['id'] == id_]['rew'].mean())
    new_df.to_csv('data/bhv_mdp2.csv', index=False)
    bonus = np.array(bonus)
    bonus = np.floor(bonus * bonus_sum / bonus.sum() * 100, ) / 100
    pay_bonus = pd.DataFrame(data=np.array([luus, list(bonus)]).T, columns=['id', 'value'])
    pay_bonus.to_csv('tmp/pay_bonus.csv', index=False)
    print('Bonus expted sum: {}, bonus sum: {}'.format(bonus_sum, bonus.sum()))
