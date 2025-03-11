import sys
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
import uuid
import scipy.stats

import rtnet as rtn
importlib.reload(rtn)
import analyze_bhv_mnist as ab
importlib.reload(ab)
import train_it_mnist as ted
importlib.reload(ted)

def give_evidence(det_model, guide, x):
    det_model.load_state_dict(guide(x, None))
    det_model.eval()
    return torch.exp(det_model(x).detach())

def load_net(net, path, state='.latest', device='cpu', verbose=True):
    train_loss = []
    valid_loss = []
    try:
        mm = torch.load(path + state, weights_only=False, map_location=torch.device(device))
        net.load_state_dict(mm['model_state'])
        train_loss = mm['train_loss']
        valid_loss = mm['valid_loss']
        if verbose:
            print('Loaded {}{} at {} epoch'.format(net_path, state, len(train_loss)))
    except:
        if verbose:
            print('Failed to loading {}{}'.format(net_path, state))
    return net, train_loss, valid_loss

class evidGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, dropout=0, bidirectional=False, n_lls=1):
        super(evidGRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # Network to estimate action
        self.hidden = nn.GRU(
                    input_size=self.input_size,
                    hidden_size=hidden_size,
                    num_layers=n_layers,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=bidirectional,
        )
        layers = []
        if bidirectional:
            gruout_size = copy.copy(hidden_size) * 2
        else:
            gruout_size = copy.copy(hidden_size)

        layers = []
        for ll in range(n_lls - 1):
            layers.append(nn.Linear(gruout_size, gruout_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(gruout_size, self.output_size))
        self.out_resp = nn.Sequential(*layers)

        layers = []
        n_lls_conf = 2
        for ll in range(n_lls_conf - 1):
            layers.append(nn.Linear(self.output_size, self.output_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(self.output_size, 1))
        self.out_conf = nn.Sequential(*layers)

    def forward(self, x, h0=None):
        output, hn = self.hidden(x, h0)
        values = self.out_resp(output)
        evid = F.softmax(values, dim=-1)
        conf = self.out_conf(evid)
        conf = F.sigmoid(self.out_conf(evid))
        return evid, values, conf, hn

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Simuate RTNet')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--n-generated-seqs', type=int, default=100000, metavar='N',
                        help='number of generated sequences (default: 0)')
    parser.add_argument('--fast-bnn', action='store_true', default=False,
                        help='use fast computation of BNN using preloaded weights')
    parser.add_argument('--difficulty', type=str, default='easy', metavar='easy/difficult',
                        help='difficulty for GRU sample (default: easy)')
    parser.add_argument('--sat', type=str, default='accuracy', metavar='accuracy/speed',
                        help='focus for GRU sample (default: accuracy)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--light', action='store_true', default=False,
                        help='Use a light-weight model or not')
    parser.add_argument('--plot-interval', type=int, default=0, metavar='N',
                        help='how many epochs to wait before plotting training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    args.sat += ' focus'
    device = 'cuda:0' if use_cuda else 'cpu'
    args.plot_interval = args.epochs + 1 if args.plot_interval == 0 else args.plot_interval
    print('Light weight: {}'.format(args.light))
    print('Dry run: {}'.format(args.dry_run))
    print('Device: {}'.format(device))
    print('Fast BNN: {}'.format(args.fast_bnn))


    ########################
    # Create marged sequence datasets (cnn_score and bhv)
    ########################
    seq_path = 'tmp/seqs.csv'
    try:
        dfseqs = pd.read_csv(seq_path)
        n_seqs = len(dfseqs['uid'].unique())
    except:
        dfseqs = pd.DataFrame()
        n_seqs = 0

    if n_seqs < args.n_generated_seqs:
        # Load pretrained model
        net_path, net_param_path, model, guide, _, _, _, _ = rtn.mk_pyro_model(path='tmp/bnn{}.pt', light=args.light, device=device, guide_hide=True)
        det_model = rtn.Net(use_pyro=False, light_weight=args.light).to(device)

        # Load behavioral data
        bhv = ab.load_bhv(path='rtnet/behavioral data.csv', reject=True)
        part_idxs = bhv.subject.unique()
        n_resps = len(bhv)
        n_parts = len(part_idxs)

        # Load mnist data
        test_loader = rtn.create_mnist_loaders(sets='test', test_batch_size=16)

        # Marge
        rt2frame_resol = .1 # RT resolution for conversion from sec to frame
        max_n_frames = int(np.round(bhv['resp_rt'].max() / rt2frame_resol))
        noise_lvs = {'easy': 2.1, 'difficult': 2.9}
        for ii in tqdm(range(n_seqs, args.n_generated_seqs)):

            uid = str(uuid.uuid4())[:8]
            tt = np.random.randint(n_resps)
            s = bhv.iloc[tt]
            img = test_loader.dataset[s['mnist_index'] - 1][0]
            img = img.view(1, 1, 227, 227)
            true_label = test_loader.dataset[s['mnist_index'] - 1][1]
            if s['stim'] != true_label:
                print('Warning: the true labels on MNIST and bhv are not match!')

            n_frames = int(np.round(s['resp_rt'] / rt2frame_resol))
            imgn = img + noise_lvs[s['difficulty']] * torch.rand(img.shape)
            imgn = imgn.to(device)

            if args.fast_bnn:
                n_det_models = 100
                if (ii == n_seqs) or (ii % 1000 == 0):
                    det_models = []
                    for jj in range(n_det_models):
                        _det_model = rtn.Net(use_pyro=False, light_weight=args.light).to(device)
                        _det_model.load_state_dict(guide(imgn, None))
                        _det_model.eval()
                        det_models.append(_det_model)

            df_seq = pd.DataFrame()
            for ff in range(max_n_frames):

                if args.fast_bnn:
                    evid = torch.exp(det_models[np.random.randint(n_det_models)](imgn).detach())[0]
                else:
                    evid = give_evidence(det_model, guide, imgn)[0]

                sd = s.to_frame().T
                sd['frame'] = [ff]
                sd['evidence0'] = [evid[0].item()]
                sd['evidence1'] = [evid[1].item()]
                sd['evidence2'] = [evid[2].item()]
                sd['evidence3'] = [evid[3].item()]
                sd['evidence4'] = [evid[4].item()]
                sd['evidence5'] = [evid[5].item()]
                sd['evidence6'] = [evid[6].item()]
                sd['evidence7'] = [evid[7].item()]
                sd['evidence8'] = [evid[8].item()]
                sd['evidence9'] = [evid[9].item()]
                df_seq = pd.concat([df_seq, sd], axis=0)
            df_seq['n_frames'] = [n_frames] * max_n_frames
            df_seq['uid'] = [uid] * max_n_frames
            dfseqs = pd.concat([dfseqs, df_seq], axis=0)

            if args.fast_bnn:
                if (ii == (args.n_generated_seqs - 1)) or (ii % 100 == 0):
                    dfseqs.to_csv(seq_path, index=False)


    ########################
    # Train
    ########################
    net_path = 'tmp/gru_d{}s{}.pt'.format(args.difficulty[0], args.sat[0])
    _, _, evn = ted.init_nets(device=device)
    # evn = ted.evidGRU(input_size=10, output_size=10, hidden_size=4, n_layers=1, dropout=0, n_lls=1).to(device)
    evn, train_loss, valid_loss = load_net(evn, net_path, state='.valid')
    opt = optim.Adam(evn.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    if args.epochs > 0:
        dl, dstrg = ted.load_data(difficulty_src=args.difficulty, sat_src=args.sat, batch_size=args.batch_size,
                sets_tvt=['train', 'valid'], device=device, dry_run=args.dry_run)
        n_batches_train = len(dl['train'])
        n_batches_valid = len(dl['valid'])
    match_train, match_valid = [], []
    correct_train, correct_valid = [], []
    for ee in tqdm(range(args.epochs)):
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
        match_train.append((np.array(pred_labels) == np.array(real_labels)).sum() / len(pred_labels))
        correct_train.append((np.array(pred_labels) == np.array(true_labels)).sum() / len(pred_labels))

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
        match_valid.append((np.array(pred_labels) == np.array(real_labels)).sum() / len(pred_labels))
        correct_valid.append((np.array(pred_labels) == np.array(true_labels)).sum() / len(pred_labels))

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
            plt.plot(match_train)
            plt.plot(match_valid)
            plt.title('%match-to-behav')
            plt.subplot(133); plt.gca()
            plt.plot(correct_train)
            plt.plot(correct_valid)
            plt.title('%correct')
            plt.pause(.01)


    ########################
    # Test
    ########################
    dl, dstrg = ted.load_data(difficulty_src=args.difficulty, sat_src=args.sat, batch_size=args.batch_size,
            sets_tvt=['test'], device=device, dry_run=args.dry_run)
    evn, train_loss, valid_loss = load_net(evn, net_path, state='.valid')
    evn.eval()
    participants = list(dl['test'].dataset.xp.keys())
    loss, match, correct = [], [], []
    for pp, partici in enumerate(participants):
        xv, yv = dl['test'].dataset.xp[partici], dl['test'].dataset.yp[partici]
        evid, valu, hn = evn(xv) # output: softmax prob, raw value before softmax, final hidden state
        l, pred_labels, real_labels, true_labels = ted.compute_batch_loss(valu, yv, loss_fn)
        loss.append(l.item())
        match.append((np.array(pred_labels) == np.array(real_labels)).sum() / len(pred_labels))
        correct.append((np.array(pred_labels) == np.array(true_labels)).sum() / len(pred_labels))

    print('[{}, {}] Loss: {:.3e}, %match: {:.2%}, %correct: {:.2%}'.format(
        args.difficulty, args.sat, np.mean(loss), np.mean(match), np.mean(correct)))
    pred_csv_path = 'tmp/pred_ts.csv'
    try:
        pred_df = pd.read_csv(pred_csv_path)
    except:
        pred_df = pd.DataFrame()
    _pred_df = pd.DataFrame({
        'difficulty':[args.difficulty] * len(participants),
        'sat':[args.sat] * len(participants),
        'subject':participants,
        'loss':loss,
        'match':match,
        'correct':correct,
        })
    pred_df = pd.concat([pred_df, _pred_df])
    pred_df = pred_df.drop_duplicates(subset=['difficulty', 'sat', 'subject'], keep='last', ignore_index=True)
    pred_df.to_csv(pred_csv_path, index=False)
