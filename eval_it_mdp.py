import sys
import importlib
import pickle
import train_it_mdp as sv
importlib.reload(sv)
import torch
import torch.nn as nn
import n_choice_markov as mdp
importlib.reload(mdp)
import numpy as np
import matplotlib.pylab as plt
from sklearn.feature_selection import mutual_info_regression
import itertools
import scipy.stats
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"
import joblib
import pandas as pd
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import os 
import show_res_mdp as srm
importlib.reload(srm)
import copy

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

def eval_org_others(target_id, seqs, actnets, zs, n_steps_src, n_steps_trg, n_actions_trg=2):
    valid_pids = list(actnets.keys())
    losses = np.zeros([len(valid_pids), 2])
    matches = np.zeros([len(valid_pids), 2])
    # print('{} participants are valid.'.format(len(valid_pids)))
    f_loss_fidelity = nn.BCELoss() # reconstruction loss

    loss_opp = .0
    match_opp = .0
    x, y = seqs[target_id]
    act_real = y[:, :, :n_actions_trg]
    cmbres = pd.DataFrame()
    for _pid in valid_pids:
        _act_pred, _, _ = actnets[_pid](x)
        act_pred = _act_pred.detach()
        loss = f_loss_fidelity(act_pred, act_real).item()
        percent_match = compute_percent_match(act_real, act_pred)
        # print(target_id, _pid, loss, percent_match)
        if _pid == target_id:
            loss_org = loss
            match_org = percent_match
        else:
            loss_opp += loss
            match_opp += percent_match
        res1 = {'n_steps_src':n_steps_src, 'n_steps_trg':n_steps_trg,
                'target_id': target_id, 'source_id': _pid,
                'nllik':loss, 'match':percent_match,
                'z1':zs[_pid][0][0].item(), 'z2':zs[_pid][0][1].item(),
                }
        cmbres = pd.concat([cmbres, pd.Series(res1).to_frame().T], axis=0)
    loss_opp /= len(valid_pids) - 1
    match_opp /= len(valid_pids) - 1
    res1 = {'id': target_id, 'n_steps_src': n_steps_src, 'n_steps_trg': n_steps_trg,
            'nllik_org': loss_org, 'nllik_opp': loss_opp,
            'match_org': match_org, 'match_opp': match_opp}
    dfres = pd.Series(res1).to_frame().T
    losses = [loss_org, loss_opp]
    matches = [match_org, match_opp]
    return dfres, losses, matches, cmbres

if __name__=='__main__':
    n_blocks = 3
    device = 'cpu'
    # device = 'cuda:0'
    suffix = '.valid'
    n_actions_trg = 2
    dim_z = 2
    testm = scipy.stats.ttest_rel

    df_path = './data/bhv_mdp2.csv'
    df = pd.read_csv(df_path)
    pids = list(df['id'].unique())
    prod_set = [
            (2, 3),
            (3, 2),
            ]

    s2o = sv.state2onehot(max_n_steps=3)
    dim_state = s2o.len
    dfres_path = 'tmp/encdec_loss_b.csv'
    cmbres_path = 'tmp/encdec_loss_b_cmbs.csv'
    try:
        dfres = pd.read_csv(dfres_path)
        cmbres = pd.read_csv(cmbres_path)
    except:
        dfres = pd.DataFrame()
        cmbres = pd.DataFrame()

    dfz = pd.DataFrame()
    for n_steps_src, n_steps_trg in prod_set:
        print('\n**********************\nSource: {}, Target: {}'.format(n_steps_src, n_steps_trg))

        seq_path = 'tmp/sqs_trans{}{}.pkl'.format(n_steps_src, n_steps_trg)
        try:
            with open(seq_path, 'rb') as f:
                x_src, y_src, seqs_src, x_trg, y_trg, seqs_trg = pickle.load(f)
            print('Loaded {}'.format(seq_path))
        except:
            df_src, _ = sv.load_df(df_path, n_steps_src, [], n_blocks)
            df_trg, _ = sv.load_df(df_path, n_steps_trg, [], n_blocks)
            sv.check_id_consistency(df_src, df_trg)
            x_src, y_src, seqs_src = sv.out_seqs_each_agent(df_src)
            x_trg, y_trg, seqs_trg = sv.out_seqs_each_agent(df_trg)
            with open(seq_path, 'wb') as f:
                pickle.dump([x_src, y_src, seqs_src, x_trg, y_trg, seqs_trg], f)

        # pids = pids[:10]
        for pp, leave_pid in enumerate(pids):
            net_path = 'tmp/mdp_it_s{}_s{}_leave_{}.pth'.format(n_steps_src, n_steps_trg, leave_pid)
            # net_path = 'tmp/mdp_it_s{}_s{}_leave_{}.pth'.format(n_steps_src, n_steps_trg, 'none') # a model with all data for testing
            enc, dec, actnet = sv.init_nets(dim_state, n_actions_trg, n_steps_src, n_steps_trg, device=device, verbose=False)
            enc, dec, training_losses, rew_rates = sv.load_net(enc, dec, net_path, suffix, device=device, verbose=True)
            if len(training_losses[0]) > 0:
                enc.eval(); dec.eval()
                ans = {}
                z_ags = {}
                for _pp, _pid in enumerate(pids):
                    x, y = seqs_src[_pid]
                    z_seqs = enc(x).detach().mean(0).view(1, dim_z)
                    z_ags[_pid] = z_seqs
                    w_ags = dec(z_seqs).detach()
                    fixed_actnet, params = sv.put_w2net(actnet, w_ags[0])
                    ans[_pid] = copy.deepcopy(fixed_actnet.eval())

                dfres_, losses, matches, cmbres_ = eval_org_others(leave_pid, seqs_trg, ans, z_ags, n_steps_src, n_steps_trg)
                dfres = pd.concat([dfres, dfres_], axis=0)
                cmbres = pd.concat([cmbres, cmbres_], axis=0)
        dfres = dfres.drop_duplicates(subset=['id', 'n_steps_src', 'n_steps_trg'], keep='last', ignore_index=True)
        cmbres = cmbres.drop_duplicates(subset=['source_id', 'target_id', 'n_steps_src', 'n_steps_trg'], keep='last', ignore_index=True)

        _df = dfres.query('n_steps_src == @n_steps_src & n_steps_trg == @n_steps_trg')
        ress = testm(_df['nllik_org'], _df['nllik_opp'], alternative='less')
        print('Fidelity loss (Org vs Opp):\t{:.3f} vs {:.3f} (stat={:.3f}, p={:.3f})'.format(_df['nllik_org'].mean(0), _df['nllik_opp'].mean(0), ress.statistic, ress.pvalue))
        ress = testm(_df['match_org'], _df['match_opp'], alternative='greater')
        print('%match (Org vs Opp):\t{:.3f} vs {:.3f} (stat={:.3f}, p={:.3f})'.format(_df['match_org'].mean(0), _df['match_opp'].mean(0), ress.statistic, ress.pvalue))

        cmbres.to_csv(cmbres_path, index=False)
        dfres.to_csv(dfres_path, index=False)


        # z-plane
        net_path = 'tmp/mdp_it_s{}_s{}_leave_none.pth'.format(n_steps_src, n_steps_trg)
        enc, dec, actnet = sv.init_nets(dim_state, n_actions_trg, n_steps_src, n_steps_trg, device=device, verbose=False)
        enc, dec, training_losses, rew_rates = sv.load_net(enc, dec, net_path, suffix, device=device, verbose=True)

        seq_path = 'tmp/sqs_trans{}{}.pkl'.format(n_steps_src, n_steps_trg)
        with open(seq_path, 'rb') as f:
            x_src, y_src, seqs_src, x_trg, y_trg, seqs_trg = pickle.load(f)

        z_ags = np.zeros([len(pids), dim_z])
        for pp, pid in enumerate(pids):
            x_src, y_src = seqs_src[pid]
            # z_seqs = enc(x_src).detach().mean(0).view(1, dim_z)
            z_seqs = enc(x_src).detach().view(3, dim_z)
            z_ags[pp] = z_seqs.mean(0)
            data = {'id':[pid] * 3, 'z1':z_seqs[:, 0], 'z2':z_seqs[:, 1], 'n_steps':n_steps_src}
            dfz = pd.concat([dfz, pd.DataFrame(data)], axis=0)
    dfz.to_csv('tmp/z_hum.csv', index=False)
