import importlib
import argparse
import matplotlib.pylab as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import itertools
import numpy as np
import torch
import torch.nn as nn
import scipy.stats as sst
import os
import pandas as pd
import sklearn.metrics as metrics
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"

import train_it_mnist as ted
importlib.reload(ted)

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

def viz_z_partici(z_partici, fig):
    fig.clf()
    participants = list(z_partici.keys())
    dim_z = z_partici[participants[0]].shape[1]
    if dim_z == 1:
        ax = fig.add_subplot(111)
        _xmin, _xmax = torch.cat([*z_partici.values()]).min(), torch.cat([*z_partici.values()]).max()
        xmin = _xmin - .5 * (_xmax - _xmin)
        xmax = _xmax + .5 * (_xmax - _xmin)
        x = np.linspace(xmin, xmax, 500)
        for partici in participants:
            loc = z_partici[partici][:, 0].mean()
            scale = z_partici[partici][:, 0].std()
            # x = np.linspace(sst.norm.ppf(.05, loc, scale), sst.norm.ppf(.95, loc, scale), 100)
            ax.plot(x, sst.norm.pdf(x, loc, scale))
        ax.set_xlabel('$z_0$')
        ax.set_ylabel('Probability density')
    else:
        if dim_z > 4:
            cmbs_z_dim = list(itertools.combinations(range(4), 2))
        else:
            cmbs_z_dim = list(itertools.combinations(range(dim_z), 2))
        n_cols = int(np.ceil(np.sqrt(len(cmbs_z_dim))))
        n_rows = int(np.ceil(len(cmbs_z_dim) / n_cols))
        for cmcm, [a, b] in enumerate(cmbs_z_dim):
            ax = fig.add_subplot(n_cols, n_rows, cmcm + 1)
            for partici in participants:
                l = plt.plot(z_partici[partici][:, a].mean(), z_partici[partici][:, b].mean(), '.')
                if z_partici[partici].shape[0] > 1:
                    confidence_ellipse(z_partici[partici][:, a].numpy(), z_partici[partici][:, b].numpy(), ax, n_std=1, facecolor=l[0].get_color(), edgecolor='none', alpha=.1)
                    # plt.plot(z_partici[partici][:, a], z_partici[partici][:, b], '.', c=l[0].get_color())
            ax.set_xlabel('$z_{}$'.format(a))
            ax.set_ylabel('$z_{}$'.format(b))

if __name__ == '__main__':
    # os.system('rsync -avzh --delete-excluded higashi@192.168.100.100:/home/higashi/Research/20250109_rtnet/tmp/edm_dses.* ~/Research/20250109_rtnet/tmp/')

    # Script settings
    parser = argparse.ArgumentParser(description='Evaluate the encoder-decoder model')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    args = parser.parse_args()
    print('Dry run: {}'.format(args.dry_run))

    # Experiment parameters
    tvt = 'test'
    set_abbr_tasks = [
            'esea', 'daea', 'dsea',
            'eaes', 'daes', 'dses',
            'eada', 'esda', 'dsda',
            'eads', 'esds', 'dads',
            ]
    stestm = sst.ttest_rel
    preloaded_df = pd.read_csv('tmp/seqs.csv')

    for tasks in set_abbr_tasks:
        difficulty_src, sat_src, difficulty_trg, sat_trg = ted.translate_tasks(tasks)
        print('======================================')
        print('{}: ({}, {}) --> ({}, {})'.format(tasks, difficulty_src, sat_src, difficulty_trg, sat_trg))
        net_path = 'tmp/edm_{}.pt'.format(tasks)

        # Load encoder/decoder
        enc, dec, evn = ted.init_nets()
        enc, dec, _, _ = ted.load_net(enc, dec, net_path, state='.valid')
        loss_fn = nn.CrossEntropyLoss()
        enc.eval(); dec.eval()
        dim_z = enc.output_size

        # Load dataset
        dlsrc, dstrg = ted.load_data(difficulty_src, sat_src, difficulty_trg, sat_trg, sets_tvt=[tvt], dfseqs=preloaded_df, dry_run=args.dry_run, verbose=False)
        nx, participants = ted.out_nx_participant(dlsrc[tvt].dataset)
        z_partici = {}
        z_partici_clusters = {}
        loss_org = np.zeros([len(participants)])
        match_org = np.zeros([len(participants)])
        correct_org = np.zeros([len(participants)])
        threshold_conf = .0
        pred_trials_csv_path = 'tmp/pred_{}.csv'.format(tasks)
        pred_df = pd.DataFrame()
        for pp, partici in enumerate(participants):
            z = enc([nx[pp]]).detach()
            w = dec(z).detach()
            z_partici[partici] = z

            n_repeat = 10
            batch_size = 100
            zp = torch.zeros([n_repeat, z.size(1)])
            for ii in range(n_repeat):
                zp[ii] = enc([nx[pp][np.random.randint(0, nx[pp].size(0), batch_size)]]).detach()
            z_partici_clusters[partici] = zp

            # to original individual
            l, pred_labels, real_labels, true_labels, pred_rt, real_rt = ted.compute_participant_loss(evn, w, [partici], dstrg[tvt], loss_fn, threshold_conf=threshold_conf)
            loss_org[pp] = l.item()
            match_org[pp] = (np.array(pred_labels) == np.array(real_labels)).sum() / len(pred_labels)
            correct_org[pp] = (np.array(pred_labels) == np.array(true_labels)).sum() / len(pred_labels)
            # rterror_org[pp] = metrics.mean_absolute_error(real_rt, pred_rt)

            # save results for each trial
            _pred_df = pd.DataFrame({
                'subject':[partici] * len(pred_labels),
                'true_labels':true_labels,
                'real_labels':real_labels,
                'pred_labels':pred_labels,
                })
            pred_df = pd.concat([pred_df, _pred_df])
        pred_df.to_csv('tmp/pred_{}.csv'.format(tasks), index=False)

        n_weights = w.size(1)
        loss_opp = np.zeros([len(participants)])
        match_opp = np.zeros([len(participants)])
        correct_opp = np.zeros([len(participants)])
        for pp, partici in enumerate(participants):
            _participants = participants.copy()
            _participants.remove(partici)
            w = torch.zeros([len(_participants), n_weights])
            for pp2, partici2 in enumerate(_participants):
                w[pp2] = dec(z_partici[partici2])
            l, pred_labels_opp, real_labels_opp, true_labels_opp, pred_rt_opp, real_rt_opp = ted.compute_participant_loss(evn, w, [partici] * len(_participants), dstrg[tvt], loss_fn, threshold_conf=threshold_conf)
            loss_opp[pp] = l.item()
            match_opp[pp] = (np.array(pred_labels_opp) == np.array(real_labels_opp)).sum() / len(pred_labels_opp)
            correct_opp[pp] = (np.array(pred_labels_opp) == np.array(true_labels_opp)).sum() / len(pred_labels_opp)

        viz_z_partici(z_partici_clusters, fig=plt.figure(5))
        plt.savefig('figs/z/{}.pdf'.format(tasks), bbox_inches='tight', transparent=True)

        z_mean_p = {}
        z_all1 = torch.zeros([0, dim_z])
        for pp, partici in enumerate(participants):
            z_mean_p[partici] = z_partici_clusters[partici].mean(0)
            z_all1 = torch.cat([z_all1, z_partici_clusters[partici]], axis=0)
        z_mean = torch.stack([*z_mean_p.values()]).mean(0)
        wti = np.zeros([len(participants)])
        btw = np.zeros([len(participants)])
        for pp, partici in enumerate(participants):
            d = z_partici_clusters[partici] - z_mean_p[partici]
            wti[pp] = np.sqrt((d ** 2).sum(1)).mean(0)
            d = z_partici_clusters[partici] - z_mean
            btw[pp] = np.sqrt((d ** 2).sum(1)).mean(0)
            # d = z_mean_p[partici] - z_mean
            # btw[pp] = np.sqrt((d ** 2).sum(0))

        # z_all1 = z_all.view(z_all.size(0) * z_all.size(1), z_all.size(2))
        cmbs_z_dim = list(itertools.combinations(range(dim_z), 2))
        for cmcm, [a, b] in enumerate(cmbs_z_dim):
            ress = sst.pearsonr(z_all1[:, a], z_all1[:, b])
            print('\tz_{} vs z_{}: C={:.3f}, p={:.3f}'.format(a, b, ress.statistic, ress.pvalue))

        ress = stestm(loss_org, loss_opp, alternative='two-sided')
        print('[Loss (CEL)]\t Org {:.3f} vs Opp {:.3f} (stat={:.3f}, p={:.3f})'.format(loss_org.mean(), loss_opp.mean(), ress.statistic, ress.pvalue))
        ress = stestm(match_org, match_opp, alternative='two-sided')
        print('[%match]\t Org {:.2%} vs Opp {:.2%} (stat={:.3f}, p={:.3f})'.format(match_org.mean(), match_opp.mean(), ress.statistic, ress.pvalue))
        ress = stestm(correct_org, correct_opp, alternative='two-sided')
        print('[%correct]\t Org {:.2%} vs Opp {:.2%} (stat={:.3f}, p={:.3f})'.format(correct_org.mean(), correct_opp.mean(), ress.statistic, ress.pvalue))
        # ress = sst.ttest_rel(rterror_org, rterror_opp, alternative='two-sided')
        # print('[rt_error]\t Org {:.3f} vs Opp {:.3f} (stat={:.3f}, p={:.3f})'.format(rterror_org.mean(), rterror_opp.mean(), ress.statistic, ress.pvalue))
        ress = stestm(wti, btw, alternative='two-sided')
        print('[wti vs btw]\t wti {:.3} vs bet {:.3} (stat={:.3f}, p={:.3f})'.format(wti.mean(), btw.mean(), ress.statistic, ress.pvalue))
        plt.figure(3).clf()
        ax1 = plt.subplot(131)
        ax2 = plt.subplot(132)
        ax3 = plt.subplot(133)
        for pp, partici in enumerate(participants):
            ax1.plot([.1, .9], [loss_org[pp], loss_opp[pp]], 'k-o', alpha=.2)
            ax2.plot([.1, .9], [match_org[pp], match_opp[pp]], 'k-o', alpha=.2)
            ax3.plot([.1, .9], [correct_org[pp], correct_opp[pp]], 'k-o', alpha=.2)
        ax1.bar(0, loss_org.mean(), yerr=loss_org.std()/np.sqrt(len(participants)), ec='k', fc='w')
        ax1.bar(1, loss_opp.mean(), yerr=loss_opp.std()/np.sqrt(len(participants)), ec='k', fc='w')
        ax1.set_xticks([0, 1], ['Original', 'Others'])
        ax1.set_title('Loss')
        ax2.bar(0, match_org.mean(), yerr=match_org.std()/np.sqrt(len(participants)), ec='k', fc='w')
        ax2.bar(1, match_opp.mean(), yerr=match_opp.std()/np.sqrt(len(participants)), ec='k', fc='w')
        ax2.set_xticks([0, 1], ['Original', 'Others'])
        ax2.set_title('\\%match')
        ax3.bar(0, correct_org.mean(), yerr=correct_org.std()/np.sqrt(len(participants)), ec='k', fc='w')
        ax3.bar(1, correct_opp.mean(), yerr=correct_opp.std()/np.sqrt(len(participants)), ec='k', fc='w')
        ax3.set_xticks([0, 1], ['Original', 'Others'])
        ax3.set_title('\\%correct')

        pred_csv_path = 'tmp/pred_ed.csv'
        try:
            pred_df = pd.read_csv(pred_csv_path)
        except:
            pred_df = pd.DataFrame()
        _pred_df = pd.DataFrame({
            'difficulty_src':[difficulty_src] * len(participants),
            'sat_src':[sat_src] * len(participants),
            'difficulty_trg':[difficulty_trg] * len(participants),
            'sat_trg':[sat_trg] * len(participants),
            'subject':participants,
            'loss':loss_org,
            'loss_opp':loss_opp,
            'match':match_org,
            'match_opp':match_opp,
            'correct':correct_org,
            'correct_opp':correct_opp,
            })
        pred_df = pd.concat([pred_df, _pred_df])
        pred_df = pred_df.drop_duplicates(subset=['difficulty_src', 'sat_src', 'difficulty_trg', 'sat_trg', 'subject'], keep='last', ignore_index=True)
        # pred_df.to_csv(pred_csv_path, index=False)

        plt.pause(.1)
