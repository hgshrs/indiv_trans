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

def load_net(net_path, fmark='.latest'):
    _, _, actnet = sv.init_nets(dim_state, n_actions, n_steps, n_steps)
    print(actnet)
    try:
        mm = torch.load(net_path + fmark)
        actnet.load_state_dict(mm['actnet'])
        train_loss = mm['train_loss']
        valid_loss = mm['valid_loss']
        print('Loaded {}{} at {} epoch.'.format(net_path, fmark, len(train_loss)))
    except:
        train_loss = []
        valid_loss = []
        print('Failed to load {}{}.'.format(net_path, fmark))
    return actnet, train_loss, valid_loss


if __name__=='__main__':
    n_steps = 2
    n_blocks = 3
    lr = 1e-5 # learning rate for NN optimizers
    n_epochs = 100000
    # n_epochs = 0
    fmark = '.latest'
    n_gen_seqs = 100
    n_gen_episodes = 150

    id_tvt_path = 'tmp/split_{}_train_valid_test.csv'.format('bhv')
    iddf = pd.read_csv(id_tvt_path)
    df_path = './data/bhv_mdp2.csv'

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
    # Training an action net
    # ===============================
    net_path = 'tmp/checkpoints/mdp_actnet_{}_s{}.pth'.format('bhv', n_steps)
    actnet, train_loss, valid_loss = load_net(net_path)
    opt_act = optim.Adam(actnet.parameters(), lr=lr)
    f_loss_fidelity = nn.BCELoss() # reconstruction loss

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

        if ee % 1000 == 0:
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
    try:
        dfres_a = pd.read_csv('tmp/actnet_loss_a.csv')
    except:
        dfres_a = pd.DataFrame()
    actnet, train_loss, valid_loss = load_net(net_path, fmark='.valid')
    actnet.eval()
    act_pred, _, _ = actnet(x_test)
    f_loss_fidelity = nn.BCELoss(reduction='none') # reconstruction loss
    test_loss_trial = f_loss_fidelity(act_pred, act_real_test)
    test_loss_ag = np.zeros(y_test[:, :, 2].unique().shape[0])
    for pp, pidx in enumerate(y_test[:, :, 2].unique()):
        id_ = df_test[df_test['agent'] == pidx.item()].iloc[0]['id']
        test_loss_ag[pp] = test_loss_trial[y_test[:, :, n_actions] == pidx].mean()
        test_loss_opp = test_loss_trial[y_test[:, :, n_actions] != pidx].mean()
        res = {
                'id': id_,
                'agent': int(pidx.item()),
                'n_steps': n_steps,
                'nllik_org': test_loss_ag[pp],
                'nllik_opp': test_loss_opp.item(),
                }
        dfres_a = pd.concat([dfres_a, pd.Series(res).to_frame().T], axis=0)
    print('Loss for test: {:.3f}+-{:.3f}'.format(test_loss_ag.mean(), test_loss_ag.std()))
    dfres_a.to_csv('tmp/actnet_loss_a.csv', index=False)
