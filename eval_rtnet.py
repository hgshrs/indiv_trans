import sys
import importlib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import itertools
import train_it_mnist as ted
importlib.reload(ted)
import train_ts_mnist as ttm
importlib.reload(ttm)

if __name__ == '__main__':
    device = 'cpu'
    df = pd.read_csv('tmp/seqs.csv') # created by train_decision_mod.py
    iddf = pd.read_csv('tmp/split_mnistnone.csv')
    pids = iddf['id'].unique()
    loss_fn = nn.CrossEntropyLoss() # cross entropy loss does not requre softmax before input
    threshold_a = { 'easy': 0, 'difficult': -1,
                    'accuracy focus': 0, 'speed focus': -1}

    for abbr_task in ['ea', 'es', 'da', 'ds']:
        difficulty, sat = ted.translate_tasks(abbr_task)
        print('Task: {}, {}'.format(difficulty, sat))
        ds = ted.mk_ds(df, difficulty=difficulty, sat=sat, pids=pids, device=device)

        nc = ds.x.sum(0).sum(0)
        prior = 1 / nc
        prior[0] = 0.
        prior[9] = 0.
        prior /= prior.max()

        loss = []
        match = []
        correct = []
        for pp, pid in enumerate(pids):
            xv, yv = ds.xp[pid], ds.yp[pid]
            loss_p = 0.
            true_labels = np.zeros(xv.size(0), dtype=int)
            real_labels = np.zeros(xv.size(0), dtype=int)
            pred_labels = np.zeros(xv.size(0), dtype=int)
            real_rts = np.zeros(xv.size(0), dtype=int)
            for tt in range(xv.size(0)):
                _real = yv[tt, :10]
                _real_rt = int(yv[tt, 11] + 1) + threshold_a[difficulty] + threshold_a[sat]
                _valu = xv[tt, :_real_rt].sum(0)
                _valu = _valu * prior
                _evid = F.softmax(_valu, dim=-1)

                loss_p += loss_fn(_valu, _real).item()
                true_labels[tt] = int(yv[tt, 13])
                real_labels[tt] = _real.argmax().item()
                pred_labels[tt] = _evid.argmax().item()
                real_rts[tt] = _real_rt
            loss.append(loss_p / xv.size(0))
            match.append((np.array(pred_labels) == np.array(real_labels)).sum() / len(pred_labels))
            correct.append((np.array(pred_labels) == np.array(true_labels)).sum() / len(pred_labels))
            # print(loss[-1], match[-1], correct[-1])
            # sys.exit()

        print('[{}, {}] Loss: {:.3e}, %match: {:.2%}, %correct: {:.2%}'.format(
            difficulty, sat, np.mean(loss), np.mean(match), np.mean(correct)))
        pred_csv_path = 'tmp/pred_rtn.csv'
        try:
            pred_df = pd.read_csv(pred_csv_path)
        except:
            pred_df = pd.DataFrame()
        _pred_df = pd.DataFrame({
            'task':[abbr_task] * len(pids), 'id':pids, 'nllik':loss, 'match':match, 'correct':correct})
        pred_df = pd.concat([pred_df, _pred_df])
        pred_df = pred_df.drop_duplicates(subset=['task', 'id'], keep='last', ignore_index=True)
        pred_df.to_csv(pred_csv_path, index=False)
