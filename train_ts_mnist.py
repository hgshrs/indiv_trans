import sys
import os
import importlib
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pyro
from pyro.infer.autoguide import AutoDiagonalNormal, AutoNormal
from tqdm import tqdm
import numpy as np
import pandas as pd
import joblib
import copy
import matplotlib.pylab as plt

import train_it_mnist as ted
importlib.reload(ted)

def give_evidence(det_model, guide, x):
    det_model.load_state_dict(guide(x, None))
    det_model.eval()
    return torch.exp(det_model(x).detach())

def load_net(net, path, state='.latest', device='cpu', verbose=True):
    train_loss, valid_loss = [], []
    train_match, valid_match = [], []
    train_correct, valid_correct = [], []
    try:
        mm = torch.load(path + state, weights_only=False, map_location=torch.device(device))
        net.load_state_dict(mm['model_state'])
        train_loss = mm['train_loss']
        valid_loss = mm['valid_loss']
        # train_match = mm['train_match']
        # valid_match = mm['valid_match']
        # train_correct = mm['train_correct']
        # valid_correct = mm['valid_correct']
        if verbose:
            print('Loaded {}{} at {} epoch'.format(path, state, len(train_loss)))
    except:
        if verbose:
            print('Failed to loading {}{}'.format(path, state))
    return net, train_loss, valid_loss #, train_match, valid_match, train_correct, valid_correct

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Simuate RTNet')
    parser.add_argument('--train-type', type=str, default='leave',
                        help='type of training data: leave (default), given')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--plot-interval', type=int, default=10, metavar='N',
                        help='how many epochs to wait before plotting training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cuda:0' if use_cuda else 'cpu'
    args.plot_interval = args.epochs + 1 if args.plot_interval == 0 else args.plot_interval
    # print('Dry run: {}'.format(args.dry_run))
    print('Device: {}'.format(device))
    print('Train type: {}'.format(args.train_type))


    preloaded_df = pd.read_csv('tmp/seqs.csv') # made by make_mnist_seqs.py
    pids = list(np.sort(preloaded_df['subject'].unique()))
    dim_z = 2
    loss_fn = nn.CrossEntropyLoss()

    set_abbr_task = ['ea', 'es', 'da', 'ds']
    for abbr_task in set_abbr_task:
        difficulty, sat = ted.translate_tasks(abbr_task)
        print('Task: {}, {}'.format(difficulty, sat))

        for pp, pid in enumerate(pids):
        # for pp, pid in enumerate(['none']):
            iddf_path = 'tmp/mnist/split_leave_{}.csv'.format(pid)
            if args.train_type == 'leave':
                try:
                    iddf = pd.read_csv(iddf_path)
                    print('Loaded {}'.format(iddf_path))
                except:
                    iddf = ted.leave_one_participant_id(pids, pid, sp_tv=[.9, .1])
                    iddf.to_csv(iddf_path, index=False)
                    print('Created {}'.format(iddf_path))
            elif args.train_type == 'given':
                iddf = pd.read_csv('tmp/mnist/split_leave_none.csv')
                if pid != 'none':
                    iddf['set'] = 'valid'
                    iddf.loc[iddf['id'] == pid, 'set'] = 'train'

            net_path = 'tmp/mnist/gru_{}_{}_{}.pt'.format(abbr_task, args.train_type, pid)
            _, _, evn = ted.init_nets(device=device)
            _, train_loss, valid_loss = load_net(evn, net_path, state='.latest')
            evn, _, _ = load_net(evn, net_path, state='.valid')
            opt = optim.Adam(evn.parameters(), lr=args.lr)

            ########################
            # Train
            ########################
            n_epochs = args.epochs - len(train_loss)
            if n_epochs > 0:
                dl, _ = ted.load_data(difficulty_src=difficulty, sat_src=sat, batch_size=args.batch_size,
                        iddf=iddf, sets_tvt=['train', 'valid'], device=device, dry_run=args.dry_run)
                n_batches_train = len(dl['train'])
                n_batches_valid = len(dl['valid'])

            train_match, valid_match = [], []
            train_correct, valid_correct = [], []
            for ee in tqdm(range(n_epochs)):

                # network training
                pred_labels, real_labels, true_labels = [], [], []
                batch_loss = 0.
                evn.train()
                evn.zero_grad()
                opt.zero_grad()
                for x, y in dl['train']:
                    evid, valu, hn = evn(x)
                    l, _pred_labels, _real_labels, _true_labels = ted.compute_batch_loss(valu, y, loss_fn)
                    pred_labels += _pred_labels
                    real_labels += _real_labels
                    true_labels += _true_labels
                    l.backward()
                    opt.step()
                    batch_loss += l.item()
                    if args.dry_run:
                        n_batches_train = 1
                        break
                train_loss.append(batch_loss / n_batches_train)
                train_match.append((np.array(pred_labels) == np.array(real_labels)).sum() / len(pred_labels))
                train_correct.append((np.array(pred_labels) == np.array(true_labels)).sum() / len(pred_labels))

                # compute validation loss
                pred_labels, real_labels, true_labels = [], [], []
                batch_loss = 0.
                evn.eval()
                for x, y in dl['valid']:
                    evid, valu, hn = evn(x)
                    l, _pred_labels, _real_labels, _true_labels = ted.compute_batch_loss(valu, y, loss_fn)
                    pred_labels += _pred_labels
                    real_labels += _real_labels
                    true_labels += _true_labels
                    batch_loss += l.item()
                    if args.dry_run:
                        n_batches_valid = 1
                        break
                valid_loss.append(batch_loss / n_batches_valid)
                valid_match.append((np.array(pred_labels) == np.array(real_labels)).sum() / len(pred_labels))
                valid_correct.append((np.array(pred_labels) == np.array(true_labels)).sum() / len(pred_labels))

                saving_vars = {'model_state':evn.state_dict(), 'train_loss':train_loss, 'valid_loss':valid_loss}
                torch.save(saving_vars, net_path + '.latest')
                if train_loss[-1] <= np.min(train_loss):
                    torch.save(saving_vars, net_path + '.train')
                if valid_loss[-1] <= np.min(valid_loss):
                    torch.save(saving_vars, net_path + '.valid')

                if (ee + 1) % args.plot_interval == 0:
                    plt.figure(1).clf()
                    plt.subplot(131); plt.gca()
                    plt.semilogy(train_loss)
                    plt.semilogy(valid_loss)
                    plt.title('Train loss')
                    plt.subplot(132); plt.gca()
                    plt.plot(train_match)
                    plt.plot(valid_match)
                    plt.title('%match')
                    plt.subplot(133); plt.gca()
                    plt.plot(train_correct)
                    plt.plot(valid_correct)
                    plt.title('%correct')
                    plt.pause(.01)
