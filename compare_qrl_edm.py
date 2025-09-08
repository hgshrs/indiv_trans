import sys
import importlib
import n_choice_markov as mdp
importlib.reload(mdp)
import pandas as pd
import pickle
import train_it_mdp as sv
importlib.reload(sv)
import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm
import uuid
import os
import eval_it_mdp as eim
importlib.reload(eim)
import copy
import scipy.stats
import matplotlib.tri as tri
import torch

def bhv2z(seqs_src, enc, actnet):
    pids = list(seqs_src.keys())
    z_ags = {}
    actnets = {}
    for pp, pid in enumerate(pids):
        x, y = seqs_src[pid]
        z = enc(x).detach().mean(0).view(1, dim_z)
        z_ags[pid] = z
        w = dec(z).detach()
        fixed_actnet, params = sv.put_w2net(actnet, w[0])
        actnets[pid] = copy.deepcopy(fixed_actnet)
    return pids, z_ags, actnets

if __name__=='__main__':
    n_steps_src = 2
    n_steps_trg = 3
    dim_state = 5
    n_actions_trg = 2
    device = 'cpu'
    suffix = '.valid'
    dim_z = 2
    n_qrl_agents = 1000
    n_episodes = 50 # Length of a sequence
    testm = scipy.stats.ttest_rel

    # ===============================
    # Generate QRL-agent dataset
    # ===============================
    qrl_bhv_path = './data/qrl_mdp.csv'
    yn = input('Do you want to make QRL behaviour dataset? [y/]: ')
    if yn == 'y':
        dfp = pd.read_csv('tmp/param_qagents.csv')
        plt.figure(1).clf()
        plt.subplot(2, 2, 1)
        plt.hist(dfp['alpha'])
        plt.subplot(2, 2, 2)
        plt.hist(dfp['beta'])
        plt.subplot(2, 2, 3)
        plt.hist(dfp['gamma'])
        plt.subplot(2, 2, 4)
        plt.hist(dfp['init_q'])

        t1, t2 = .1, .9
        alpha1, alpha2 = dfp['alpha'].quantile(t1), dfp['alpha'].quantile(t2)
        beta1, beta2 = dfp['beta'].quantile(t1), dfp['beta'].quantile(t2)
        gamma1, gamma2 = dfp['gamma'].quantile(t1), dfp['gamma'].quantile(t2)
        init_q1, init_q2 = dfp['init_q'].quantile(t1), dfp['init_q'].quantile(t2)
        alpha1, alpha2 = .1, 1.
        beta1, beta2 = 1, 15
        gamma1, gamma2 = 1, 1
        init_q1, init_q2 = 0, 0
        print('alpha: {:.3f}--{:.3f}'.format(alpha1, alpha2))
        print('beta: {:.3f}--{:.3f}'.format(beta1, beta2))
        print('gamma: {:.3f}--{:.3f}'.format(gamma1, gamma2))
        print('init_q: {:.3f}--{:.3f}'.format(init_q1, init_q2))

        pmat = np.zeros([n_qrl_agents, 4])
        pmat[:, 0] = np.random.uniform(alpha1, alpha2, size=n_qrl_agents) # alpha
        pmat[:, 1] = np.random.uniform(beta1, beta2, size=n_qrl_agents) # beta
        pmat[:, 2] = np.random.uniform(gamma1, gamma2, size=n_qrl_agents) # gamma
        pmat[:, 3] = np.random.uniform(init_q1, init_q2, size=n_qrl_agents) # init_q

        pdf = pd.DataFrame(pmat, columns=['alpha', 'beta', 'gamma', 'init_q'])

        df = pd.DataFrame()
        for aa in tqdm(range(n_qrl_agents)):
            id_ = uuid.uuid4().urn[-8:] # ID
            alpha = pdf.loc[aa]['alpha']; beta = pdf.loc[aa]['beta']; gamma = pdf.loc[aa]['gamma']; init_q = pdf.loc[aa]['init_q']
            ag = mdp.q_agent(alpha=alpha, beta=beta, gamma=gamma, init_q=init_q)
            for n_steps in [2, 3]:
                mp = mdp.mk_default_mdp(n_steps=n_steps)
                seqdf = mdp.generate_seq(ag, mp, n_episodes, verbose=False, ax=None)
                seqdf['id'] = id_
                seqdf['agent'] = aa
                seqdf['agent_alpha'] = alpha
                seqdf['agent_beta'] = beta
                seqdf['agent_gamma'] = gamma
                seqdf['agent_init_q'] = init_q
                seqdf['n_steps'] = n_steps
                seqdf['block'] = 0
                # print(id_, n_steps, seqdf['rew'].sum())
                df = pd.concat([df, seqdf])
        df.to_csv(qrl_bhv_path, index=False)
    else:
        df = pd.read_csv(qrl_bhv_path)

    dfres = pd.DataFrame()
    cmbres = pd.DataFrame()
    dfz = pd.DataFrame()
    for n_steps_src, n_steps_trg in [[3, 2], [2, 3]]:
        print(n_steps_src, n_steps_trg)
        # ===============================
        # Load encoder
        # ===============================
        net_path = 'tmp/mdp_it_s{}_s{}_leave_none.pth'.format(n_steps_src, n_steps_trg) # a model with all data for testing
        enc, dec, actnet = sv.init_nets(dim_state, n_actions_trg, n_steps_src, n_steps_trg, device=device, verbose=False)
        enc, dec, training_losses, rew_rates = sv.load_net(enc, dec, net_path, suffix, device=device, verbose=True)

        # ===============================
        # Project QRL-agent data on z-plane
        # ===============================
        # Make a table for the agent parameters
        df_path = qrl_bhv_path
        df = pd.read_csv(df_path)
        seq_path = 'tmp/sqs_trans{}{}_qrl.pkl'.format(n_steps_src, n_steps_trg)
        res_path = 'tmp/encdec_loss_b_qrl.csv'
        cmbres_path = 'tmp/encdec_loss_b_qrl_cmb.csv'
        try:
            if yn == 'y':
                os.remove(seq_path)
            with open(seq_path, 'rb') as f:
                x_src, y_src, seqs_src, x_trg, y_trg, seqs_trg = pickle.load(f) # from eval_it_mdp.py
            print('{} loaded.'.format(seq_path))
        except:
            df_src, _ = sv.load_df(df_path, n_steps_src, [], 1)
            df_trg, _ = sv.load_df(df_path, n_steps_trg, [], 1)
            sv.check_id_consistency(df_src, df_trg)
            x_src, y_src, seqs_src = sv.out_seqs_each_agent(df_src)
            x_trg, y_trg, seqs_trg = sv.out_seqs_each_agent(df_trg)
            with open(seq_path, 'wb') as f:
                pickle.dump([x_src, y_src, seqs_src, x_trg, y_trg, seqs_trg], f)
            print('{} created.'.format(seq_path))

        pids, z_ags_qrl, ans_qrl = bhv2z(seqs_src, enc, actnet)
        for pp, pid in enumerate(pids):
            p = df[df['id'] == pid].iloc[0]
            data = {'id':pid, 'z1':z_ags_qrl[pid][0][0].item(), 'z2':z_ags_qrl[pid][0][1].item(),
                    'alpha':p['agent_alpha'], 'beta':p['agent_beta'], 'gamma':p['agent_gamma'], 'init_q':p['agent_init_q'],
                    'n_steps':n_steps_src}
            dfz = pd.concat([dfz, pd.Series(data).to_frame().T], axis=0)
        dfz.to_csv('tmp/z_qrl.csv', index=False)

        for pp, pid in enumerate(pids):
            _dfres, losses, matches, _cmbres = eim.eval_org_others(pid, seqs_trg, ans_qrl, z_ags_qrl, n_steps_src, n_steps_trg)
            dfres = pd.concat([dfres, _dfres])
            cmbres = pd.concat([cmbres, _cmbres])
        dfres = dfres.drop_duplicates(subset=['id', 'n_steps_src', 'n_steps_trg'], keep='last', ignore_index=True)
        cmbres = cmbres.drop_duplicates(subset=['target_id', 'source_id', 'n_steps_src', 'n_steps_trg'], keep='last', ignore_index=True)
        dfres.to_csv(res_path, index=False)
        cmbres.to_csv(cmbres_path, index=False)

        _df = dfres.query('n_steps_src == @n_steps_src & n_steps_trg == @n_steps_trg')
        ress = testm(_df['nllik_org'], _df['nllik_opp'], alternative='less')
        print('Fidelity loss (Org vs Opp):\t{:.3f} vs {:.3f} (stat={:.3f}, p={:.3f})'.format(_df['nllik_org'].mean(0), _df['nllik_opp'].mean(0), ress.statistic, ress.pvalue))
        ress = testm(_df['match_org'], _df['match_opp'], alternative='greater')
        print('%match (Org vs Opp):\t{:.3f} vs {:.3f} (stat={:.3f}, p={:.3f})'.format(_df['match_org'].mean(0), _df['match_opp'].mean(0), ress.statistic, ress.pvalue))
