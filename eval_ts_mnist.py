import importlib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

import train_it_mnist as ted
importlib.reload(ted)
import train_ts_mnist as tts
importlib.reload(tts)

if __name__ == '__main__':
    train_type = 'given'
    res_path = 'tmp/mnist_ts2.csv'

    loss_fn = nn.CrossEntropyLoss()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    try:
        resdf = pd.read_csv(res_path)
    except:
        resdf = pd.DataFrame()
    preloaded_df = pd.read_csv('tmp/seqs.csv') # made by make_mnist_seqs.py
    pids = list(np.sort(preloaded_df['subject'].unique()))

    set_abbr_tasks = [
            'eaea', 'esea', 'daea', 'dsea',
            'eses', 'eaes', 'daes', 'dses',
            'dada', 'eada', 'esda', 'dsda',
            'dsds', 'eads', 'esds', 'dads',
            ]
    for abbr_tasks in set_abbr_tasks:
        iddf = pd.read_csv('tmp/mnist/split_leave_none.csv')
        iddf['set'] = 'test'
        difficulty, sat = ted.translate_tasks(abbr_tasks[-2:])
        dl, _ = ted.load_data(difficulty_src=difficulty, sat_src=sat, iddf=iddf, sets_tvt=['test'], dfseqs=preloaded_df, verbose=False)
        for pp, pid in enumerate(pids):
            net_path = 'tmp/mnist/gru_{}_{}_{}.pt'.format(abbr_tasks[:2], train_type, pid)
            _, _, evn = ted.init_nets(device=device)
            evn, train_loss, valid_loss = tts.load_net(evn, net_path, state='.valid')
            evn.eval()

            xv, yv = dl['test'].dataset.xp[pid], dl['test'].dataset.yp[pid]
            evid, valu, hn = evn(xv)
            ls, pred_labels, real_labels, true_labels = ted.compute_batch_loss(valu, yv, loss_fn)
            match = (np.array(pred_labels) == np.array(real_labels)).sum() / len(pred_labels)
            correct = (np.array(pred_labels) == np.array(true_labels)).sum() / len(pred_labels)
            _resdf = {'tasks':abbr_tasks, 'id':pid, 'train_type':train_type, 'nllik':ls.item(), 'match':match, 'correct':correct}
            resdf = pd.concat([resdf, pd.Series(_resdf).to_frame().T], axis=0)
        resdf = resdf.drop_duplicates(subset=['tasks', 'id', 'train_type'], keep='last', ignore_index=True)
        resdf.to_csv(res_path, index=False)
