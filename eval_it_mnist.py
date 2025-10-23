import sys
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

if __name__ == '__main__':
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
    pids = list(np.sort(preloaded_df['subject'].unique()))
    dim_z = 2
    loss_fn = nn.CrossEntropyLoss()

    res_path = 'tmp/mnist_opp2.csv'
    try:
        resdf = pd.read_csv(res_path)
    except:
        resdf = pd.DataFrame()

    actdf = pd.DataFrame()
    for abbr_tasks in set_abbr_tasks:
        difficulty_src, sat_src, difficulty_trg, sat_trg = ted.translate_tasks(abbr_tasks)
        print('======================================')
        print('{}: ({}, {}) --> ({}, {})'.format(abbr_tasks, difficulty_src, sat_src, difficulty_trg, sat_trg))

        # Load dataset
        iddf = pd.read_csv('tmp/mnist/split_leave_none.csv')
        iddf['set'] = 'test'
        dlsrc, dstrg = ted.load_data(difficulty_src, sat_src, difficulty_trg, sat_trg, iddf, sets_tvt=[tvt], dfseqs=preloaded_df, dry_run=args.dry_run, verbose=False)
        nx, pids = ted.out_nx_participant(dlsrc[tvt].dataset)

        # Load neural nets
        zp = {}
        wp = {}
        for pp, pid in enumerate(pids):
            net_path = 'tmp/mnist/edm_{}_leave_{}.pt'.format(abbr_tasks, pid)
            enc, dec, evn = ted.init_nets(device='cpu', verbose=False)
            enc, dec, train_loss, valid_loss, train_match, valid_match = ted.load_net(enc, dec, net_path, state='.valid', verbose=False)
            enc.eval(); dec.eval()

            wp[pid] = {}
            zp[pid] = {}
            for pp2, pid2 in enumerate(pids):
                z = enc([nx[pp2]]).detach()
                w = dec(z).detach()
                wp[pid][pid2] = w.detach()
                zp[pid][pid2] = z.detach()

        for pp, pid in enumerate(pids):
            xv, yv = dstrg[tvt].xp[pid], dstrg[tvt].yp[pid]
            z0 = zp[pid][pid][0]
            for pp2, pid2 in enumerate(pids):
                _, params = ted.put_w2net(evn, wp[pid][pid2][0])
                evid, valu, _ = torch.func.functional_call(evn, params, xv)
                ls, pred_labels, real_labels, true_labels = ted.compute_batch_loss(valu, yv, loss_fn)
                match = (np.array(pred_labels) == np.array(real_labels)).sum() / len(pred_labels)
                correct = (np.array(pred_labels) == np.array(true_labels)).sum() / len(pred_labels)
                z1 = zp[pid][pid2][0]
                _resdf = {
                    'tasks':abbr_tasks,
                    'target_id':pid, # behavour to be predicted
                    'source_id':pid2, # z for prediction
                    'nllik':ls.item(), 'match':match, 'correct':correct,
                    'z1':zp[pid][pid2][0][0].item(), 'z2':zp[pid][pid2][0][1].item(),
                    'z_dist':np.linalg.norm(z0 - z1)
                    }
                resdf = pd.concat([resdf, pd.Series(_resdf).to_frame().T], axis=0)
                if pid == pid2:
                    n_trials = len(true_labels)
                    _dic = {'tasks':[abbr_tasks] * n_trials, 'id':[pid] * n_trials, 'true':true_labels, 'real':real_labels, 'pred':pred_labels}
                    actdf = pd.concat([actdf, pd.DataFrame(_dic)], axis=0)
        resdf = resdf.drop_duplicates(subset=['tasks', 'target_id', 'source_id'], keep='last', ignore_index=True)
        resdf.to_csv(res_path, index=False)
        actdf.to_csv('tmp/mnist/pred_real_act.csv', index=False)
