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

def compute_fidelity_opp(aa_org, id_org):
    act_real_org = torch.zeros([0, n_episodes_trg * n_steps_trg, n_actions_trg])
    act_pred_org = torch.zeros([0, n_episodes_trg * n_steps_trg, n_actions_trg])
    act_real_opp = torch.zeros([0, n_episodes_trg * n_steps_trg, n_actions_trg])
    act_pred_opp = torch.zeros([0, n_episodes_trg * n_steps_trg, n_actions_trg])
    x, y = seqs_trg[id_org]
    _act_real = y[:, :, :n_actions_trg]
    for aa_opp, id_opp in enumerate(ids):
        fixed_actnet, params = sv.put_w2net(actnet, w_ags[aa_opp])
        _act_pred, _, _ = fixed_actnet(x)
        _act_pred = _act_pred.detach()
        if aa_org == aa_opp:
            act_real_org = torch.cat([act_real_org, _act_real], axis=0)
            act_pred_org = torch.cat([act_pred_org, _act_pred], axis=0)
        else:
            act_real_opp = torch.cat([act_real_opp, _act_real], axis=0)
            act_pred_opp = torch.cat([act_pred_opp, _act_pred], axis=0)
    return act_real_org, act_pred_org, act_real_opp, act_pred_opp

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of `x` and `y`

    Parameters
    ----------
    x, y : array_like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

if __name__=='__main__':
    # set_src = [1, 2, 3]
    set_src = [3]
    # set_trg = [1, 2, 3]
    set_trg = [2]
    n_blocks = 3
    device = 'cpu'
    # device = 'cuda:0'
    fmark = '.valid'
    n_actions_src = 2
    n_actions_trg = 2
    id_tvt_path = 'tmp/split_{}_train_valid_test.csv'.format('bhv')

    df_path = './data/bhv_mdp2.csv'
    iddf = pd.read_csv(id_tvt_path)
    prod_set = list(itertools.product(set_src, set_trg))
    # test_sets = ['train', 'valid', 'test']
    test_sets = ['test']
    if len(test_sets) == 1:
        ct = plt.get_cmap('gray')
    else:
        ct = plt.get_cmap('tab10')

    s2o = sv.state2onehot(max_n_steps=3)
    dim_state = s2o.len
    try:
        dfres_b = pd.read_csv('tmp/encdec_loss_b.csv')
    except:
        dfres_b = pd.DataFrame()

    plt.figure(6).clf()
    for n_steps_src, n_steps_trg in prod_set:
        print('\n**********************\nSource: {}, Target: {}'.format(n_steps_src, n_steps_trg))
        net_path = 'tmp/checkpoints/mdp_{}_s{}_s{}.pth'.format('bhv', n_steps_src, n_steps_trg)
        enc, dec, actnet = sv.init_nets(dim_state, n_actions_trg, n_steps_src, n_steps_trg, device=device, verbose=False)
        enc, dec, _ = sv.load_net(enc, dec, net_path, fmark, device=device, verbose=True)
        enc.eval(); dec.eval()

        mp_trg = mdp.mk_default_mdp(n_steps=n_steps_trg)
        dim_z = enc.output_size

        plt.figure(1).clf()
        plt.figure(3).clf()
        wti = np.zeros(0)
        btw = np.zeros(0)
        plt.figure(5).clf()
        loss_fidelity_agent = np.zeros([0, 2])
        for tt, set_tvt in enumerate(test_sets):
            print('================={}'.format(set_tvt))
            if set_tvt == 'all':
                ids = iddf['id']
            else:
                ids = iddf[iddf['set'] == set_tvt]['id']

            df_src, _ = sv.load_df(df_path, n_steps_src, ids, n_blocks)
            df_trg, _ = sv.load_df(df_path, n_steps_trg, ids, n_blocks)
            sv.check_id_consistency(df_src, df_trg)
            ids = df_src['id'].unique()
            print('IDs for train: {}, for test: {} (N = {})'.format('bhv', 'bhv', len(ids)))
            n_episodes_trg = len(df_trg['episode'].unique())
            seq_path = 'tmp/sqs_{}_{}_{}{}.pkl'.format('bhv', set_tvt, n_steps_src, n_steps_trg)
            try:
                with open(seq_path, 'rb') as f:
                    x_src, y_src, seqs_src, x_trg, y_trg, seqs_trg = pickle.load(f)
                print('Loaded {}'.format(seq_path))
            except:
                x_src, y_src, seqs_src = sv.out_seqs_each_agent(df_src)
                x_trg, y_trg, seqs_trg = sv.out_seqs_each_agent(df_trg)
                with open(seq_path, 'wb') as f:
                    pickle.dump([x_src, y_src, seqs_src, x_trg, y_trg, seqs_trg], f)

            z_seqs = torch.zeros([0, 3, dim_z])
            z_ags = torch.zeros([0, dim_z])
            for aa, id_ in enumerate(ids):
                x, y = seqs_src[id_]
                z_seqs_ = enc(x).detach()
                z_ags = torch.cat([z_ags, z_seqs_.mean(0).view(1, dim_z)], axis=0) 
                z_seqs = torch.cat([z_seqs, z_seqs_.view(1, 3, dim_z)], axis=0) 

            if dim_z > 4:
                cmbs_z_dim = list(itertools.combinations(range(4), 2))
            else:
                cmbs_z_dim = list(itertools.combinations(range(dim_z), 2))
            n_cols = int(np.ceil(np.sqrt(len(cmbs_z_dim))))
            n_rows = int(np.ceil(len(cmbs_z_dim) / n_cols))
            plt.figure(5)
            for cmcm, [a, b] in enumerate(cmbs_z_dim):
                ax = plt.subplot(n_cols, n_rows, cmcm + 1)
                for aa, id_ in enumerate(ids):
                    l = plt.plot(z_ags[aa, a], z_ags[aa, b], 'o', zorder=1000+aa)
                    confidence_ellipse(z_seqs[aa][:, a].numpy(), z_seqs[aa][:, b].numpy(),
                            ax, n_std=1, facecolor=l[0].get_color(), edgecolor='none', alpha=.1)
                ax.set_xlabel('$z_{}$'.format(a))
                ax.set_ylabel('$z_{}$'.format(b))
            plt.savefig('figs/z/{}_{}_s{}_s{}_{}.pdf'.format('bhv', 'bhv', n_steps_src, n_steps_trg, set_tvt),
                    bbox_inches='tight', transparent=True)

            w_ags = dec(z_ags).detach()
            res = joblib.Parallel(n_jobs=-1)(joblib.delayed(compute_fidelity_opp)(aa_org, id_org) for aa_org, id_org in enumerate(ids))
            f_loss_fidelity = nn.BCELoss() # reconstruction loss
            loss_fidelity_agent_ = np.zeros([len(ids), 2])
            for aa_org, id_org in enumerate(ids):
                loss_fidelity_agent_[aa_org, 0] = f_loss_fidelity(res[aa_org][1], res[aa_org][0])
                loss_fidelity_agent_[aa_org, 1] = f_loss_fidelity(res[aa_org][3], res[aa_org][2])

                res1 = {
                        'id': id_org,
                        'n_steps_src': n_steps_src,
                        'n_steps_trg': n_steps_trg,
                        'nllik_org': loss_fidelity_agent_[aa_org, 0],
                        'nllik_opp': loss_fidelity_agent_[aa_org, 1],
                        }
                dfres_b = pd.concat([dfres_b, pd.Series(res1).to_frame().T], axis=0)
            testm = scipy.stats.ttest_rel
            ress = testm(loss_fidelity_agent_[:, 0], loss_fidelity_agent_[:, 1], alternative='less')
            print('Fidelity loss (Org vs Opp):\t{:.3f} vs {:.3f} (stat={:.3f}, p={:.3f})'.format(*loss_fidelity_agent_.mean(0), ress.statistic, ress.pvalue))
            loss_fidelity_agent = np.concatenate([loss_fidelity_agent, loss_fidelity_agent_], axis=0)

            plt.figure(3); plt.subplot(122)
            shift = .2
            alpha = .2
            plt.plot(np.zeros(len(ids)) + shift, loss_fidelity_agent_[:, 0], 'o', alpha=alpha, c=ct(tt))
            plt.plot(np.ones(len(ids)) - shift, loss_fidelity_agent_[:, 1], 'o', alpha=alpha, c=ct(tt))
            plt.plot(np.concatenate([np.zeros(len(ids)) + shift, np.ones(len(ids)) - shift]).reshape(-1, len(ids)), np.concatenate([loss_fidelity_agent_[:, 0], loss_fidelity_agent_[:, 1]]).reshape(-1, len(ids)), alpha=alpha, c=ct(tt))
            plt.title('NLL: (p={:.3f})'.format(ress.pvalue))

            # =================
            # Btw/wti agent analysis for the laten representation
            wti_ = np.sqrt(((z_seqs - z_seqs.mean(1).view(len(ids), 1, dim_z).repeat(1, n_blocks, 1)) ** 2).sum(2)).mean(1)
            wti = np.concatenate([wti, wti_])
            btw_ = np.sqrt(((z_seqs - z_seqs.mean(1).mean(0).view(1, 1, dim_z).repeat(len(ids), n_blocks, 1)) ** 2).sum(2)).mean(1)
            btw = np.concatenate([btw, btw_])
            plt.figure(3); plt.subplot(121)
            shift = .2
            alpha = .2
            plt.plot(np.zeros(len(ids)) + shift, wti_, 'o', alpha=alpha, c=ct(tt))
            plt.plot(np.ones(len(ids)) - shift, btw_, 'o', alpha=alpha, c=ct(tt))
            plt.plot(np.concatenate([np.zeros(len(ids)) + shift, np.ones(len(ids)) - shift]).reshape(-1, len(ids)), np.concatenate([wti_, btw_]).reshape(-1, len(ids)), alpha=alpha, c=ct(tt))
            ress = testm(wti_, btw_, alternative='two-sided')
            print('z distance (with vs betw):\t{:.3f} vs {:.3f} (stat={:.3f}, p={:.3f})'.format(wti_.mean(), btw_.mean(), ress.statistic, ress.pvalue))

            # =================
            # Correlation between two latent variables of z
            z_all = z_seqs.view(z_seqs.size(0) * z_seqs.size(1), dim_z)
            if len(ids) > 3: # over 3 samples needs to compute MI by kNN.
                cmbs_z_dim = list(itertools.combinations(range(dim_z), 2))
                # print('Mutual information')
                set_z = set(range(dim_z))
                for cmcm, [a, b] in enumerate(cmbs_z_dim):
                    mir = mutual_info_regression(z_all[:, a].view(-1, 1), z_all[:, b])
                    # print('\tz{} vs z{}: {:.3f}'.format(a, b, mir[0]))
                    ress = scipy.stats.pearsonr(z_all[:, a], z_all[:, b])
                    print('\tz{} vs z{}: C={:3f}, p={:.3f}'.format(a, b, ress.statistic, ress.pvalue))
                    if ress.pvalue < .05:
                        try:
                            set_z.remove(b)
                        except:
                            pass
                print('Latent variables which seem independent: {}'.format(set_z))

        dfres_b = dfres_b.drop_duplicates(subset=['id', 'n_steps_src', 'n_steps_trg'],
                keep='last', ignore_index=True)
        dfres_b.to_csv('tmp/encdec_loss_b.csv', index=False)
        print('================={}'.format(test_sets))
        plt.figure(3); plt.subplot(122)
        plt.bar(0, loss_fidelity_agent[:, 0].mean(),
                yerr=loss_fidelity_agent[:, 0].std() / np.sqrt(loss_fidelity_agent[:, 0].shape[0]), ec='k', fc='w')
        plt.bar(1, loss_fidelity_agent[:, 1].mean(),
                yerr=loss_fidelity_agent[:, 1].std() / np.sqrt(loss_fidelity_agent[:, 1].shape[0]), ec='k', fc='w')
        plt.xticks([0, 1], ['Org', 'Opp'])
        plt.ylabel('Reconst loss')
        plt.savefig('figs/loss_{}_{}_s{}_s{}.pdf'.format('bhv', 'bhv', n_steps_src, n_steps_trg),
                bbox_inches='tight', transparent=True)
        ress = testm(loss_fidelity_agent[:, 0], loss_fidelity_agent[:, 1], alternative='two-sided')
        print('Fidelity loss (Org vs Opp):\t{:.3f} vs {:.3f} (stat={:.2e}, p={:.3f})'.format(
            loss_fidelity_agent[:, 0].mean(), loss_fidelity_agent[:, 1].mean(), ress.statistic, ress.pvalue))

        plt.figure(3); plt.subplot(121)
        plt.bar(0, wti.mean(), yerr=wti.std() / np.sqrt(wti.shape[0]), ec='k', fc='w')
        plt.bar(1, btw.mean(), yerr=btw.std() / np.sqrt(btw.shape[0]), ec='k', fc='w')
        plt.xticks([0, 1], ['Within', 'Between'])
        plt.ylabel('Distance')
        plt.savefig('figs/z_wti_btw_{}_{}_s{}_s{}.pdf'.format('bhv', 'bhv', n_steps_src, n_steps_trg),
                bbox_inches='tight', transparent=True)
        ress = testm(wti, btw, alternative='two-sided')
        print('z distance (with vs betw):\t{:.3f} vs {:.3f} (stat={:.2e}, p={:.3f})'.format(wti.mean(), btw.mean(), ress.statistic, ress.pvalue))
