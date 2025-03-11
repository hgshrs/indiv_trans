import sys
import importlib
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import joblib
import copy
from tqdm import tqdm
import numpy as np
import matplotlib.pylab as plt

def init_nets(
        dim_z=2,
        enc_hidden_size=16, enc_n_layers=4, enc_n_lls=1,
        evn_hidden_size=4, evn_n_layers=1, evn_n_lls=1,
        dec_n_lls=1,
        device='cpu', verbose=False):
    enc = zGRU_batch(input_size=10+10, output_size=dim_z, hidden_size=enc_hidden_size, n_layers=enc_n_layers, n_lls=enc_n_lls).to(device)
    evn = evidGRU(input_size=10, output_size=10, hidden_size=evn_hidden_size, n_layers=evn_n_layers, dropout=0, n_lls=evn_n_lls).to(device)
    n_params = count_net_params(evn)
    dec = DL(input_size=dim_z, output_size=n_params, n_lls=dec_n_lls).to(device)

    if verbose:
        print(enc)
        print(evn)
        print('#params for task solver: {}'.format(n_params))
        print(dec)
    return enc, dec, evn

class mk_ds(Dataset):
    def __init__(self, df, difficulty, sat, participants=[], parallel=True, device='cpu'):
        self.difficulty = difficulty
        self.sat = sat
        self.device = device

        if len(participants) == 0:
            participants = df['subject'].unique()
        res = []
        if parallel:
            res = joblib.Parallel(n_jobs=-1)(joblib.delayed(df2seq)(df, partic, difficulty, sat) for partic in participants)
        else:
            for partic in participants:
                res.append(df2seq(df, partic, difficulty, sat))

        x = []
        y = []
        self.xp = {}
        self.yp = {}
        for aa, partici in enumerate(participants):
            if len(res[aa][0]) > 0:
                x += res[aa][0]
                y += res[aa][1]
                self.xp[partici] = torch.stack(res[aa][0]).to(self.device)
                self.yp[partici] = torch.stack(res[aa][1]).to(self.device)
            else:
                print('Participant {} is missing.'.format(partici))
        self.x = torch.stack(x).to(self.device)
        self.y = torch.stack(y).to(self.device)
        self.len = len(self.x)

    def __getitem__(self,idx):
        return self.x[idx], self.y[idx]
  
    def __len__(self):
        return self.len

def df2seq(df, partic, difficulty, sat):
    max_n_frames = df['n_frames'].max()
    _seqs = df.query('subject == @partic & difficulty == @difficulty & sat == @sat')
    x = []
    y = []
    z = []
    for uid in _seqs['uid'].unique():
        cur_seq = _seqs.query('uid == @uid')
        if len(cur_seq) == max_n_frames:
            n_frames = cur_seq.iloc[0]['n_frames']
            confidence = cur_seq.iloc[0]['confidence']
            # evid = torch.zeros([n_frames, 10], dtype=torch.float32)
            evid = torch.zeros([max_n_frames, 10], dtype=torch.float32)
            # for ff in range(n_frames):
            for ff in range(max_n_frames):
                evid[ff] = torch.tensor([cur_seq.iloc[ff]['evidence{}'.format(ei)] for ei in range(10)], dtype=torch.float32)
            resp = nn.functional.one_hot(torch.tensor(cur_seq.iloc[0]['response']), 10)
            true_label = cur_seq.iloc[0]['stim']
            x.append(evid)
            y.append(torch.cat([resp, torch.tensor([partic, n_frames, confidence, true_label])], axis=0).float())
            # z.append(true_label)
    return x, y

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

    def forward(self, x, h0=None):
        output, hn = self.hidden(x, h0)
        values = self.out_resp(output)
        output = F.softmax(values, dim=-1)
        return output, values, hn

class zGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, dropout=0, bidirectional=False, n_lls=1):
        super(zGRU, self).__init__()
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
        self.out_z = nn.Sequential(*layers)

    def forward(self, x, h0=None):
        if h0 == None:
            device = x.device
            if self.hidden.bidirectional:
                h0 = torch.zeros([self.hidden.num_layers * 2, x.shape[0], self.hidden.hidden_size])
            else:
                h0 = torch.zeros([self.hidden.num_layers, x.shape[0], self.hidden.hidden_size])
            h0 = h0.to(device)
        output, hn = self.hidden(x, h0)
        output = output[:, -1]
        z = self.out_z(output)
        return z

class zGRU_batch(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, n_layers, dropout=0, bidirectional=False, n_lls=1):
        super(zGRU_batch, self).__init__()
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
        self.gruout_size = gruout_size

        layers = []
        for ll in range(n_lls - 1):
            layers.append(nn.Linear(gruout_size, gruout_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(gruout_size, self.output_size))
        self.out_z = nn.Sequential(*layers)

    def forward(self, x, h0=None):
        # x is a list of length of #samples to use different number of samples for each individual
        # x[i] is a tensor: batch_size_enc (depend on sample), sequence length, input vector at each step
        z = []
        for ss in range(len(x)):
            if h0 == None:
                h0_ = init_h0(self.hidden.num_layers, x[ss].size(0), self.hidden.hidden_size, self.hidden.bidirectional, x[ss].device)
            output, hn = self.hidden(x[ss], h0_)
            z.append(self.out_z(output[:, -1].mean(0).view(1, -1)).view(-1))
        z = torch.stack(z)
        return z

def init_h0(n_layers, batch_size, hidden_size, bidirectional=False, device='cpu'):
    if bidirectional:
        h0 = torch.zeros([n_layers * 2, batch_size, hidden_size])
    else:
        h0 = torch.zeros([n_layers, batch_size, hidden_size])
    return h0.to(device)

class DL(nn.Module):
    def __init__(self, input_size, output_size, n_lls=1):
        super(DL, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        layers = []
        for ll in range(n_lls - 1):
            layers.append(nn.Linear(input_size, input_size, bias=False))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(input_size, output_size, bias=True))
        self.dense = nn.Sequential(*layers)

    def forward(self, x):
        return self.dense(x)

def count_net_params(net):
    net_params = net.state_dict()
    net_pkeys = net_params.keys()
    n_params = 0
    for pkey in net_pkeys:
        n_params += net_params[pkey].numel()
    return n_params

def put_w2net(net, w):
    net_params = net.state_dict()
    net_pkeys = net_params.keys()
    n_params = 0
    for pkey in net_pkeys:
        shape = net_params[pkey].shape
        _n = net_params[pkey].numel()
        net_params[pkey] = w[n_params:(n_params + _n)].view(shape)
        n_params += net_params[pkey].numel()
    net.load_state_dict(net_params)
    # net.eval()
    return net, net_params

def load_net(enc, dec, path, state='.latest', device='cpu', verbose=True):
    train_loss = []
    valid_loss = []
    try:
        mm = torch.load(path + state, weights_only=False, map_location=torch.device(device))
        enc.load_state_dict(mm['enc_state'])
        dec.load_state_dict(mm['dec_state'])
        train_loss = mm['train_loss']
        valid_loss = mm['valid_loss']
        if verbose:
            print('Loaded {}{} at {} epoch'.format(path, state, len(train_loss)))
    except:
        if verbose:
            print('Failed to loading {}{}'.format(path, state))
    return enc, dec, train_loss, valid_loss

def compute_participant_loss(evn, w, seq_partici_labels, ds, loss_fn, threshold_conf=.0):
    pred_labels, real_labels, true_labels = [], [], []
    real_rt, pred_rt = [], []
    l = torch.tensor(0., requires_grad=True)
    for ss, partici in enumerate(seq_partici_labels):
        _, params = put_w2net(evn, w[ss])
        # xv, yv = x[[ss]], y[[ss]] # self-training. should be chaned to self, but different responses
        xv, yv = ds.xp[partici], ds.yp[partici]
        evid, valu, _ = torch.func.functional_call(evn, params, xv)
        ls, _pred_labels, _real_labels, _true_labels = compute_batch_loss(valu, yv, loss_fn)
        pred_labels += _pred_labels
        real_labels += _real_labels
        true_labels += _true_labels
        l = l + ls

        if threshold_conf > .0:
            real_rt += yv[:, 11].to(int).tolist()
            conf = evid.sort().values[:, :, -1] - evid.sort().values[:, :, -2]
            for ii in range(yv.size(0)):
                _pred_rt = xv.size(1)
                if conf[ii].max() > threshold_conf:
                    _pred_rt = np.where(conf[ii] > threshold_conf)[0][0]
                pred_rt.append(_pred_rt)
    return l / len(seq_partici_labels), pred_labels, real_labels, true_labels, pred_rt, real_rt

def compute_batch_loss(nout, y, loss_fn):
    pred_labels, real_labels, true_labels = [], [], []
    l = torch.tensor(0., requires_grad=True)
    for tt in range(y.shape[0]):
        pred = nout[tt, int(y[tt, 11]) - 1]
        real = y[tt, :10]
        l = l + loss_fn(pred, real)
        pred_labels.append(pred.argmax().item())
        real_labels.append(real.argmax().item())
        true_labels.append(int(y[tt, 13]))
    return l/y.shape[0], pred_labels, real_labels, true_labels

def load_data(difficulty_src=None, sat_src=None, difficulty_trg=None, sat_trg=None,
        sets_tvt=['train', 'valid', 'test'], batch_size=128,
        dfseqs=[], seq_path='tmp/seqs.csv', iddf_path='tmp/split_participants_train_valid_test.csv',
        dry_run=False, verbose=True, device='cpu'):
    if len(dfseqs) == 0:
        dfseqs = pd.read_csv(seq_path) # created by train_decision_mod.py
    # sp_tvt = [7, 1, 2] # ratio for train/valid/test participants
    # iddf = split_ids(dfseqs['subject'].unique(), sp_tvt); iddf.to_csv('tmp/split_participants_train_valid_test.csv', index=False)
    iddf = pd.read_csv(iddf_path)

    if dry_run:
        n_seqs = len(iddf) * 100
        ids = dfseqs.sample(frac=1)['uid'].unique()[:n_seqs]
        dfseqs = dfseqs.query('uid in @ids')

    if verbose:
        print('\t#partic\t#src\t#trg')
    dlsrc = {}
    dstrg = {}
    for tvt in sets_tvt:
        participants_ = iddf.query('set == @tvt')['id']
        src_len, trg_len = 0, 0
        if type(difficulty_src) != type(None):
            dlsrc[tvt] = DataLoader(
                    mk_ds(dfseqs, difficulty=difficulty_src, sat=sat_src, participants=participants_, device=device),
                    shuffle=True, batch_size=batch_size)
            src_len = dlsrc[tvt].dataset.len
        if type(difficulty_trg) != type(None):
            dstrg[tvt] = mk_ds(dfseqs, difficulty=difficulty_trg, sat=sat_trg, participants=participants_, device=device)
            trg_len = dstrg[tvt].len
        if verbose:
            print('{}\t{}\t{}\t{}'.format(tvt, len(participants_), src_len, trg_len))
    return dlsrc, dstrg

def split_ids(pids, sp_tvt):
    np.random.shuffle(pids)
    pids_train = pids[:int(sp_tvt[0] / np.sum(sp_tvt) * len(pids))]
    iddf_train = pd.DataFrame(pids_train, columns=['id'])
    iddf_train['set'] = 'train'
    pids_valid = pids[int(sp_tvt[0] / np.sum(sp_tvt) * len(pids)):int(np.sum(sp_tvt[:2]) / np.sum(sp_tvt) * len(pids))]
    iddf_valid = pd.DataFrame(pids_valid, columns=['id'])
    iddf_valid['set'] = 'valid'
    pids_test = pids[int(np.sum(sp_tvt[:2]) / np.sum(sp_tvt) * len(pids)):]
    iddf_test = pd.DataFrame(pids_test, columns=['id'])
    iddf_test['set'] = 'test'
    iddf_tvt = pd.concat([iddf_train, iddf_valid, iddf_test], axis=0).reset_index()
    iddf_tvt.to_csv('tmp/split_participants_train_valid_test.csv', index=False)
    return iddf_tvt

def concat_xy_enc(x, y):
    iny = torch.zeros_like(x)
    for tt in range(y.shape[0]):
        iny[tt, int(y[tt, 11] - 1)] = y[tt, :10]
    nx = torch.cat([x, iny], axis=2)
    return nx

def out_nx_participant(ds):
    device = ds.x.device
    participants = list(ds.xp.keys())
    nx = []
    for pp, partici in enumerate(participants):
        _xp, _yp = ds.xp[partici], ds.yp[partici]
        nx += [concat_xy_enc(_xp, _yp).to(device)]
    return nx, participants

def translate_tasks(tasks):
    tt = []
    for v in tasks:
        if v == 'e':
            tt.append('easy')
        elif v == 'd':
            tt.append('difficult')
        elif v == 'a':
            tt.append('accuracy focus')
        elif v == 's':
            tt.append('speed focus')
    if len(tasks) == 2:
        difficulty, sat = tt
        return difficulty, sat
    if len(tasks) == 4:
        difficulty_src, sat_src, difficulty_trg, sat_trg = tt
        return difficulty_src, sat_src, difficulty_trg, sat_trg

if __name__ == '__main__':
    # Script settings
    parser = argparse.ArgumentParser(description='Simulate RTNet')
    parser.add_argument('--tasks', type=str, default='eaes',
                        help='Transfer tasks (default: eaes)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--save-interval', type=int, default=10, metavar='N',
                        help='how many epochs to wait before saving training status (default: 1)')
    parser.add_argument('--plot-interval', type=int, default=0, metavar='N',
                        help='how many epochs to wait before plotting training status (default: 0)')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    device = 'cuda:0' if use_cuda else 'cpu'
    args.plot_interval = args.epochs + 1 if args.plot_interval == 0 else args.plot_interval
    print('Dry run: {}'.format(args.dry_run))
    print('Device: {}'.format(device))
    difficulty_src, sat_src, difficulty_trg, sat_trg = translate_tasks(args.tasks)

    # Experiment parameters
    print('Source: {}, {}'.format(difficulty_src, sat_src))
    print('Target: {}, {}'.format(difficulty_trg, sat_trg))
    net_path = 'tmp/edm_{}.pt'.format(args.tasks)

    # Load models
    enc, dec, evn = init_nets(device=device, verbose=True)
    enc, dec, train_loss, valid_loss = load_net(enc, dec, net_path, state='.latest', device=device)
    opt_enc = optim.Adam(enc.parameters(), lr=args.lr)
    opt_dec = optim.Adam(dec.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Load dataset
    dlsrc, dstrg = load_data(difficulty_src, sat_src, difficulty_trg, sat_trg,
            sets_tvt=['train', 'valid'], batch_size=args.batch_size,
            device=device, dry_run=args.dry_run, verbose=False)

    # train encoder/decoder
    match_train = []
    match_valid = []
    for ee in tqdm(range(args.epochs)):

        # train---participant-wise evaluation
        enc.train()
        enc.zero_grad()
        opt_enc.zero_grad()
        dec.train()
        dec.zero_grad()
        opt_dec.zero_grad()
        nx, participants = out_nx_participant(dlsrc['train'].dataset)
        z = enc(nx)
        w = dec(z)
        l, pred_labels, real_labels, true_labels, _, _ = compute_participant_loss(evn, w, participants, dstrg['train'], loss_fn)
        l.backward()
        opt_enc.step()
        opt_dec.step()
        train_loss.append(l.item())
        match_train.append((np.array(pred_labels) == np.array(real_labels)).sum() / len(pred_labels))

        # valid---participant-wise evaluation
        enc.eval()
        dec.eval()
        nx, participants = out_nx_participant(dlsrc['valid'].dataset)
        z = enc(nx)
        w = dec(z)
        l, pred_labels, real_labels, true_labels, _, _ = compute_participant_loss(evn, w, participants, dstrg['valid'], loss_fn)
        valid_loss.append(l.item())
        match_valid.append((np.array(pred_labels) == np.array(real_labels)).sum() / len(pred_labels))

        if (ee + 1) % args.save_interval == 0:
            saving_vars = {
                    'enc_state':enc.state_dict(), 'dec_state':dec.state_dict(),
                    'train_loss':train_loss, 'valid_loss':valid_loss}
            torch.save(saving_vars, net_path + '.latest')
            if train_loss[-1] <= np.min(train_loss):
                torch.save(saving_vars, net_path + '.train')
            if valid_loss[-1] <= np.min(valid_loss):
                torch.save(saving_vars, net_path + '.valid')

        if (ee + 1) % args.plot_interval == 0:
            plt.figure(1).clf()
            plt.subplot(121); plt.gca()
            plt.semilogy(train_loss)
            plt.semilogy(valid_loss)
            plt.title('Train loss')
            plt.subplot(122); plt.gca()
            plt.plot(match_train)
            plt.plot(match_valid)
            plt.title('%match-to-behav')
            plt.pause(.1)
