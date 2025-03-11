import sys
import importlib
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import itertools

import train_it_mnist as ted
importlib.reload(ted)

if __name__ == '__main__':
    device = 'cpu'
    difficulities = ['easy', 'difficult']
    sats = ['accuracy focus', 'speed focus']
    df = pd.read_csv('tmp/seqs.csv') # created by train_decision_mod.py
    iddf = pd.read_csv('tmp/split_participants_train_valid_test.csv')
    participants = iddf.query('set == "test"')['id']
    loss_fn = nn.CrossEntropyLoss()

    for difficulty, sat in itertools.product(difficulities, sats):
        ds = ted.mk_ds(df, difficulty=difficulty, sat=sat, participants=participants, device=device)
        loss = []
        match = []
        correct = []
        for pp, participant in enumerate(participants):
            xv, yv = ds.xp[participant], ds.yp[participant]
            loss_p = 0.
            true_labels = np.zeros(xv.size(0))
            real_labels = np.zeros(xv.size(0))
            pred_labels = np.zeros(xv.size(0))
            for tt in range(xv.size(0)):
                _real = yv[tt, :10]
                real_rt = int(yv[tt, 11] + 1)
                valu = xv[tt, :real_rt].sum(0)
                _pred = F.softmax(valu, dim=-1)
                loss_p += loss_fn(_pred, _real).item()
                true_labels[tt] = yv[tt, 13]
                real_labels[tt] = _real.argmax()
                pred_labels[tt] = _pred.argmax()
            loss.append(loss_p / xv.size(0))
            match.append((np.array(pred_labels) == np.array(real_labels)).sum() / len(pred_labels))
            correct.append((np.array(pred_labels) == np.array(true_labels)).sum() / len(pred_labels))

        print('[{}, {}] Loss: {:.3e}, %match: {:.2%}, %correct: {:.2%}'.format(
            difficulty, sat, np.mean(loss), np.mean(match), np.mean(correct)))
        pred_csv_path = 'tmp/pred_rtn.csv'
        try:
            pred_df = pd.read_csv(pred_csv_path)
        except:
            pred_df = pd.DataFrame()
        _pred_df = pd.DataFrame({
            'difficulty':[difficulty] * len(participants),
            'sat':[sat] * len(participants),
            'subject':participants,
            'loss':loss,
            'match':match,
            'correct':correct,
            })
        pred_df = pd.concat([pred_df, _pred_df])
        pred_df = pred_df.drop_duplicates(subset=['difficulty', 'sat', 'subject'], keep='last', ignore_index=True)
        pred_df.to_csv(pred_csv_path, index=False)
