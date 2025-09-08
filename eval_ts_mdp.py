import sys
import importlib
import pandas as pd
import train_it_mdp as sv
importlib.reload(sv)
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pylab as plt
import n_choice_markov as mdp
importlib.reload(mdp)
import numpy as np
from torch.utils.data import DataLoader, Dataset
import joblib
import os

def load_net(net_path, suffix='.latest'):
    _, _, actnet = sv.init_nets(dim_state, n_actions, n_steps, n_steps)
    try:
        mm = torch.load(net_path + suffix, weights_only=False, map_location=torch.device(device))
        actnet.load_state_dict(mm['actnet'])
        train_loss = mm['train_loss']
        valid_loss = mm['valid_loss']
        print('Loaded {}{} at {} epoch.'.format(net_path, suffix, len(train_loss)))
    except:
        train_loss = []
        valid_loss = []
        print('Failed to load {}{}.'.format(net_path, suffix))
    return actnet, train_loss, valid_loss

def compute_percent_match(real, pred):
    if real.ndim > 2:
        real = real.reshape(-1, 2)
        pred = pred.reshape(-1, 2)
    pred = pred.numpy().astype(float)
    n_trials, n_actions = real.shape
    match = np.zeros([n_trials])
    for tt in range(n_trials):
        act = np.random.choice([0, 1], p=pred[tt] / pred[tt].sum())
        match[tt] = real[tt, act]
    return match.mean()


if __name__=='__main__':
    n_blocks = 3
    lr = 1e-3 # learning rate for NN optimizers
    # n_epochs = 25000
    n_epochs = 0
    n_gen_seqs = 100
    n_gen_episodes = 150

    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print('Device: {}'.format(device))

    df_path = './data/bhv_mdp2.csv'
    df = pd.read_csv(df_path)
    pids = list(df['id'].unique())

    res_path = 'tmp/actnet_loss_a.csv'
    try:
        dfres = pd.read_csv(res_path)
    except:
        dfres = pd.DataFrame()
    for n_steps in [2, 3]:
        for pid in pids:
            iddf = pd.read_csv('tmp/split_leave_{}.csv'.format(pid))
            net_path = 'tmp/mdp_ts_s{}_leave_{}.pth'.format(n_steps, pid)
            # net_path = 'tmp/mdp_ts_s{}_leave_{}.pth'.format(n_steps, pids[1])

            df_train, n_actions = sv.load_df(df_path, n_steps, iddf[iddf['set'] == 'train']['id'], n_blocks)
            ds_train = sv.mk_ds(df_train, parallel=False)
            dim_state = ds_train.dim_state_onehot
            x_train, y_train, seqs_train = sv.out_seqs_each_agent(df_train)
            act_real_train = y_train[:, :, :n_actions]
            df_valid, n_actions = sv.load_df(df_path, n_steps, iddf[iddf['set'] == 'valid']['id'], n_blocks)
            x_valid, y_valid, seqs_valid = sv.out_seqs_each_agent(df_valid)
            act_real_valid = y_valid[:, :, :n_actions]
            df_test, n_actions = sv.load_df(df_path, n_steps, iddf[iddf['set'] == 'test']['id'], n_blocks)
            x_test, y_test, seqs_test = sv.out_seqs_each_agent(df_test)
            act_real_test = y_test[:, :, :n_actions]
            mp = mdp.mk_default_mdp(n_steps=n_steps)

            # ===============================
            # Training task solvers
            # ===============================
            actnet, train_loss, valid_loss = load_net(net_path, suffix='.latest')
            opt_act = optim.Adam(actnet.parameters(), lr=lr)
            f_loss_fidelity = nn.BCELoss() # reconstruction loss

            actnet = actnet.to(device)
            x_train, y_train = x_train.to(device), y_train.to(device)
            act_real_train = act_real_train.to(device)
            x_valid, y_valid = x_valid.to(device), y_valid.to(device)
            act_real_valid = act_real_valid.to(device)

            rew_rate = np.zeros(n_epochs)
            rew_rate[:] = np.nan
            for ee in tqdm(range(n_epochs)):

                actnet.train()
                actnet.zero_grad()
                opt_act.zero_grad()

                act_pred, _, _ = actnet(x_train)
                batch_loss = f_loss_fidelity(act_pred, act_real_train)
                batch_loss .backward()
                opt_act.step()
                train_loss.append(batch_loss.item())

                actnet.eval()
                act_pred, _, _ = actnet(x_valid)
                valid_loss.append(f_loss_fidelity(act_pred, act_real_valid).item())

                if ee % 100 == 0:
                    if len(net_path) > 0:
                        saving_vars = {
                                'actnet':actnet.state_dict(),
                                'train_loss':train_loss,
                                'valid_loss':valid_loss,
                                }
                        torch.save(saving_vars, net_path + '.latest')

                        if train_loss[-1] <= np.min(train_loss):
                            torch.save(saving_vars, net_path + '.train')

                        if valid_loss[-1] <= np.min(valid_loss):
                            torch.save(saving_vars, net_path + '.valid')

                if ee % 2500 == 0:
                    plt.figure(1); plt.subplot(121); plt.cla()
                    plt.semilogy(train_loss)
                    plt.semilogy(valid_loss)
                    plt.legend(['train', 'valid'])

                    ag = sv.actnet2agent(actnet.eval())
                    # genseq = mdp.generate_seq(ag, mp, n_episodes=n_gen_episodes, verbose=False, ax=None)
                    genseqs = joblib.Parallel(n_jobs=-1)(joblib.delayed(mdp.generate_seq)(ag, mp, n_gen_episodes) for ii in range(n_gen_seqs))
                    genseq = pd.DataFrame()
                    for ii in range(n_gen_seqs):
                        genseq = pd.concat([genseq, genseqs[ii]], axis=0)
                    rew_rate[ee] = genseq['rew'].mean() * n_steps
                    plt.figure(1); plt.subplot(122); plt.cla()
                    plt.plot(np.arange(n_epochs)[np.logical_not(np.isnan(rew_rate))], rew_rate[np.logical_not(np.isnan(rew_rate))])
                    plt.plot(ee, df_train['rew'].mean() * n_steps, 'o')
                    plt.pause(.01)

            # ===============================
            # Evaluation with testing set
            # ===============================
            actnet, _, _ = load_net(net_path, suffix='.valid')
            actnet = actnet.to('cpu')
            actnet.eval()

            act_pred, _, _ = actnet(x_test)
            test_loss = f_loss_fidelity(act_pred, act_real_test).item()
            percent_match_test = compute_percent_match(act_real_test, act_pred.detach())

            loss_opp = (train_loss[-1] * len(seqs_train) + valid_loss[-1] * len(seqs_valid)) / (len(seqs_train) + len(seqs_valid))
            act_pred_train, _, _ = actnet(x_train)
            percent_match_train = compute_percent_match(act_real_train, act_pred_train.detach())
            act_pred_valid, _, _ = actnet(x_valid)
            percent_match_valid = compute_percent_match(act_real_valid, act_pred_valid.detach())
            percent_match_opp = (percent_match_train * len(seqs_train) + percent_match_valid * len(seqs_valid)) / (len(seqs_train) + len(seqs_valid))

            res = {'id': pid, 'n_steps': n_steps, 'nllik_org': test_loss, 'nllik_opp': loss_opp, 'match_org': percent_match_test, 'match_opp': percent_match_opp}
            dfres = pd.concat([dfres, pd.Series(res).to_frame().T], axis=0)
            dfres = dfres.drop_duplicates(subset=['id', 'n_steps'], keep='last', ignore_index=True)
            dfres.to_csv(res_path, index=False)
