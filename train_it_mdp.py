import sys
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pylab as plt
import itertools
import joblib
import pickle
import copy
import pandas as pd
import n_choice_markov as mdp
importlib.reload(mdp)
import os


def init_nets(dim_state, n_actions_trg, n_steps_src, n_steps_trg,
        dim_z=2,
        enc_hidden_size=32, enc_n_layers=4, enc_dropout=0, enc_bidirectional=False, enc_n_lls=4, # Parameters for the encoder
        act_hidden_size=-1, act_n_layers=1, act_dropout=0, act_bidirectional=False, act_n_lls=1, # Parameters for the actnet
        dec_n_lls=1, # Parameters for the decoder
        device='cpu', verbose=False):
    # Encoder for estimating z from behavioral sequence
    input_size = dim_state + n_actions_trg + 1 + dim_state # state (prior) + action (prior, one-hot format) + reward (prior) + state (current) 
    enc = zGRU(input_size, enc_hidden_size, enc_n_layers, dim_z, enc_dropout, enc_bidirectional, enc_n_lls).to(device)

    # Agent reproducing/generating RL behavior
    if act_hidden_size == -1:
        if n_steps_trg == 1:
            act_hidden_size = 2
        elif n_steps_trg == 2:
            act_hidden_size = 4
        elif n_steps_trg == 3:
            act_hidden_size = 8
    actnet = act_GRU(dim_state, n_actions_trg, act_hidden_size, act_n_layers, act_dropout, act_bidirectional, act_n_lls).to(device)
    n_params = count_net_params(actnet)

    # Decoder for generating GRU parameters
    dec = DL(dim_z, n_params, dec_n_lls).to(device)

    if verbose:
        print(enc)
        print(actnet)
        print('n_params: {}'.format(n_params))
        print(dec)

    return enc, dec, actnet

def split_df(df_src, df_trg, sp_tvt):
    if set(df_src['id'].unique()) != set(df_trg['id'].unique()):
        print('WARNING: The IDs does not match between the source and target domains')
    pids = df_src['id'].unique()
    np.random.shuffle(pids)
    pids_train = pids[:int(sp_tvt[0] / np.sum(sp_tvt) * len(pids))]
    pids_valid = pids[int(sp_tvt[0] / np.sum(sp_tvt) * len(pids)):int(np.sum(sp_tvt[:2]) / np.sum(sp_tvt) * len(pids))]
    pids_test = pids[int(np.sum(sp_tvt[:2]) / np.sum(sp_tvt) * len(pids)):]
    print('Split train:valid:test = {}:{}:{}'.format(len(pids_train), len(pids_valid), len(pids_test)))
    dfs_src = {}
    dfs_src['train'] = df_src.query('id in @pids_train')
    dfs_src['valid'] = df_src.query('id in @pids_valid')
    dfs_src['test'] = df_src.query('id in @pids_test')
    dfs_trg = {}
    dfs_trg['train'] = df_trg.query('id in @pids_train')
    dfs_trg['valid'] = df_trg.query('id in @pids_valid')
    dfs_trg['test'] = df_trg.query('id in @pids_test')
    return dfs_src, dfs_trg

def count_net_params(net):
    net_params = net.state_dict()
    net_pkeys = net_params.keys()
    n_params = 0
    for pkey in net_pkeys:
        n_params += net_params[pkey].numel()
    return n_params

class zGRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, output_size, dropout, bidirectional=True, n_lls=1):
        super(zGRU, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.hidden = nn.GRU(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=n_layers,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=bidirectional,
                    )
        # self.out_z = nn.Linear(hidden_size, output_size)
        layers = []
        if bidirectional:
            gruout_size = copy.copy(hidden_size) * 2
        else:
            gruout_size = copy.copy(hidden_size)

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

class act_GRU(nn.Module):
    def __init__(self, dim_state, n_actions, hidden_size, n_layers, dropout, bidirectional=False, n_lls=1):
        super(act_GRU, self).__init__()
        self.dim_state = dim_state
        self.n_actions = n_actions
        self.input_size = dim_state + n_actions + 1 + dim_state
        self.output_size = n_actions

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

        self.out_act = nn.Linear(hidden_size, self.output_size)
        layers = []
        for ll in range(n_lls - 1):
            layers.append(nn.Linear(gruout_size, gruout_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(gruout_size, self.output_size))
        self.out_act = nn.Sequential(*layers)

    def forward(self, x, h0=None):
        output, hn = self.hidden(x, h0)
        values = self.out_act(output)
        output = F.softmax(values, dim=-1)
        return output, values, hn

class DL(nn.Module):
    def __init__(self, input_size, output_size, n_lls=1):
        super(DL, self).__init__()
        self.input_size = input_size

        layers = []
        for ll in range(n_lls - 1):
            layers.append(nn.Linear(input_size, input_size, bias=False))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(input_size, output_size, bias=True))
        self.dense = nn.Sequential(*layers)

    def forward(self, x):
        return self.dense(x)

def train_net(df_src_train, df_src_valid, df_trg_train, df_trg_valid, enc, dec, actnet, losses, rew_rates, n_epochs, mp, n_gen_episodes, net_path='', batch_size=0, lr=1e-3, device='cpu', sp_intv=5, sp_intv_rew=0, parallel=True):
    train_loss = losses[0]
    valid_loss = losses[1]
    n_done_epochs = len(train_loss)
    rew_epochs = rew_rates[0]
    train_rew = rew_rates[1]
    valid_rew = rew_rates[2]
    dim_z = enc.output_size
    if sp_intv_rew == 0:
        sp_intv_rew = n_epochs

    opt_enc = optim.Adam(enc.parameters(), lr=lr)
    opt_dec = optim.Adam(dec.parameters(), lr=lr)

    agents_train = df_src_train['agent'].unique()
    agents_valid = df_src_valid['agent'].unique()
    n_actions_src = len(df_src_train['act'].unique())
    n_actions_trg = len(df_trg_valid['act'].unique())
    n_steps_trg = len(df_trg_valid['step'].unique())
    ave_rew_rate_train = np.zeros(len(agents_train))
    for aa, agent in enumerate(agents_train):
        _df = df_trg_train[df_trg_train['agent'] == agent]
        ave_rew_rate_train[aa] = _df['rew'].mean() * n_steps_trg
    ave_rew_rate_valid = np.zeros(len(agents_valid))
    for aa, agent in enumerate(agents_valid):
        _df = df_trg_valid[df_trg_valid['agent'] == agent]
        ave_rew_rate_valid[aa] = _df['rew'].mean() * n_steps_trg

    ds_src_train = mk_ds(df_src_train)
    if batch_size == 0:
        batch_size = ds_src_train.len
    dl_src_train = DataLoader(ds_src_train, shuffle=True, batch_size=batch_size)
    n_batches = len(dl_src_train)

    x_src_train, y_src_train, seqs_src_train = out_seqs_each_agent(df_src_train, device)
    x_trg_train, y_trg_train, seqs_trg_train = out_seqs_each_agent(df_trg_train, device)
    x_src_valid, y_src_valid, seqs_src_valid = out_seqs_each_agent(df_src_valid, device)
    x_trg_valid, y_trg_valid, seqs_trg_valid = out_seqs_each_agent(df_trg_valid, device)

    # Losses
    f_loss = nn.BCELoss() # reconstruction loss

    if len(train_loss) > 0:
        train_loss_min = np.min(train_loss)
        valid_loss_min = np.min(valid_loss)
    else:
        train_loss_min = np.inf
        valid_loss_min = np.inf
    for ee in tqdm(range(n_epochs)):
        running_loss = 0.
        running_valid_loss = 0.
        for x, y in dl_src_train:

            # ==========================
            # train
            # ==========================
            x = x.to(device)
            y = y.to(device)
            act_real = y[:, :, :n_actions_src]
            train_labels = y[:, 0, n_actions_src].cpu().numpy()

            enc.train()
            enc.zero_grad()
            opt_enc.zero_grad()
            dec.train()
            dec.zero_grad()
            opt_dec.zero_grad()

            z = enc(x)
            w = dec(z)
            act_real = []
            act_pred = []
            for ss in range(x.shape[0]):
                _act_real, _act_pred = valid_trg(ss, train_labels, seqs_trg_train, actnet, w, n_actions_trg)
                act_real.append(_act_real)
                act_pred.append(_act_pred)
            act_real = torch.stack(act_real)
            act_pred = torch.stack(act_pred)
            batch_loss = f_loss(act_pred, act_real)
            l = batch_loss
            l.backward()
            opt_enc.step()
            opt_dec.step()

            running_loss += l.item() / n_batches

            # ==========================
            # valid
            # ==========================
            enc.eval()
            dec.eval()
            z_src_valid = enc(x_src_valid)
            w_src_valid = dec(z_src_valid)
            valid_labels = y_src_valid[:, 0, n_actions_src].cpu().numpy()
            act_real = []
            act_pred = []
            for ss in range(x_src_valid.shape[0]):
                _act_real, _act_pred = valid_trg(ss, valid_labels, seqs_trg_valid, actnet, w_src_valid, n_actions_trg)
                act_real.append(_act_real)
                act_pred.append(_act_pred)
            act_real = torch.stack(act_real)
            act_pred = torch.stack(act_pred)
            batch_loss = f_loss(act_pred, act_real)

            l = batch_loss
            running_valid_loss += l.item() / n_batches

        train_loss.append(running_loss)
        valid_loss.append(running_valid_loss)

        enc.eval()
        dec.eval()

        if len(net_path) > 0:
            if (ee + 1) % sp_intv == 0:
                saving_vars = {
                        'enc':enc.state_dict(),
                        'dec':dec.state_dict(),
                        'train_loss':train_loss,
                        'valid_loss':valid_loss,
                        'rew_epochs':rew_epochs,
                        'train_rew':train_rew,
                        'valid_rew':valid_rew,
                        }
                torch.save(saving_vars, net_path + '.latest')

                if train_loss[-1] < train_loss_min:
                    torch.save(saving_vars, net_path + '.train')
                    train_loss_min = train_loss[-1]

                if valid_loss[-1] < valid_loss_min:
                    torch.save(saving_vars, net_path + '.valid')
                    valid_loss_min = valid_loss[-1]

        if (ee + 1) % sp_intv == 0:
            plt.figure(1); plt.subplot(121); plt.gca().cla()
            plt.semilogy(train_loss, label='train')
            plt.semilogy(valid_loss, label='valid')
            plt.legend()
            plt.title('Loss')
            plt.pause(.01)

        if (ee + 1) % sp_intv_rew == 0:
            # different-block valiation for decoder and action-net
            rew_rate_train, n_agents_train_ = out_rew_rate(seqs_trg_train, enc, dec, actnet, mp, n_gen_episodes)
            rew_rate_valid, n_agents_valid_ = out_rew_rate(seqs_trg_valid, enc, dec, actnet, mp, n_gen_episodes)
            rew_epochs.append(n_done_epochs + ee + 1)
            train_rew.append(rew_rate_train[:n_agents_train_].mean())
            valid_rew.append(rew_rate_valid[:n_agents_valid_].mean())

            # plt.figure(3).clf()
            plt.figure(1);
            plt.subplot(122); plt.gca().cla()
            plt.plot(rew_epochs, train_rew, label='train')
            plt.plot(rew_epochs, valid_rew, label='valid')
            plt.legend()
            plt.ylabel('Percentage reward')
            plt.pause(.01)

def out_rew_rate(seqs_each_agent, enc, dec, actnet, mp, n_gen_episodes, parallel=True):
    agents = list(seqs_each_agent.keys())
    n_agents = len(agents)
    if parallel:
        res = joblib.Parallel(n_jobs=-1)(joblib.delayed(generate_seq)(agent, seqs_each_agent, enc, dec, actnet, mp, n_gen_episodes) for agent in agents[:n_agents])
    else:
        res = []
        for aa, agent in enumerate(agents[:n_agents]):
            _res = generate_seq(agent, seqs_each_agent, enc, dec, actnet, mp, n_gen_episodes)
            res.append(_res)
    rew_rate = np.zeros([1, len(agents)])
    rew_rate[:] = np.nan
    for aa, agent in enumerate(agents[:n_agents]):
        rew_rate[0, aa] = res[aa]['rew'].sum() / len(res[aa]['episode'].unique())
    return rew_rate, n_agents

def generate_seq(agent_label, trg_seqs, enc, dec, actnet, mp, n_episodes=100):
    dim_z = enc.output_size
    x, y = trg_seqs[agent_label]
    z_ = enc(x).detach().mean(0).view(1, dim_z)
    w_ = dec(z_)
    fixed_actnet, params = put_w2net(actnet, w_[0])
    ag = actnet2agent(fixed_actnet)
    genseq = mdp.generate_seq(ag, mp, n_episodes=n_episodes, verbose=False, ax=None)
    return genseq

def valid_trg(ss, seq_agent_labels, trg_seqs, actnet, w, n_actions_trg):
    _, params = put_w2net(actnet, w[ss])
    id_ = agent_idx2id[seq_agent_labels[ss]]
    x, y = trg_seqs[id_]
    act_real = y[:, :, :n_actions_trg]
    act_pred, _, _ = torch.func.functional_call(actnet, params, x)
    return act_real, act_pred

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

class kl_mn_loss(nn.Module): # KL div with multivariate normal distribution
    def __init__(self):
        super(kl_mn_loss, self).__init__()

    def forward(self, a):
        mu = a.mean(0)
        S = a.T.cov()
        return .5 * (torch.dot(mu, mu) + S.trace() - a.shape[1] - torch.log(S.det()))

def show_prob(dfs):
    for key in list(dfs.keys()):
        n_agents = len(dfs[key]['agent'].unique())
        print('{}:\t{}'.format(key, n_agents))

def split_ids(pids, sp_tvt, n_agents=0):
    if n_agents > 0:
        pids = pids[:n_agents]
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
    iddf_tvt = pd.concat([iddf_train, iddf_valid, iddf_test], axis=0).reset_index(drop=True)
    return iddf_tvt

def leave_one_participant_id(pids, leave_pid, sp_tv=[.8, .2]):
    pids_ = copy.copy(pids)
    if leave_pid == 'none':
        iddf = split_ids(pids_, sp_tvt=sp_tv + [.0])
        iddf = iddf.reset_index(drop=True)
    else:
        pids_.remove(leave_pid)
        iddf = split_ids(pids_, sp_tvt=sp_tv + [.0])
        iddf_test = pd.DataFrame([leave_pid], columns=['id'])
        iddf_test['set'] = 'test'
        iddf = pd.concat([iddf, iddf_test], axis=0)
        iddf = iddf.reset_index(drop=True)
    return iddf

def load_df(df_path, n_steps, ids=[], n_blocks=3):
    df_all = pd.read_csv(df_path)
    if len(ids) == 0:
        ids = df_all['id'].unique()
    df = df_all.query('id in @ids')
    df = df[df['n_steps'] == n_steps]
    df = df[df['block'] < n_blocks]
    n_actions = len(df['act'].unique())
    return df, n_actions

def check_id_consistency(df1, df2):
    if set(df1['id'].unique()) != set(df2['id'].unique()):
        print('WARNING: The IDs does not match between the input sets')

class mk_ds(Dataset):
    def __init__(self, df, parallel=True, max_n_steps=3):
        # n_actions = len(df['act'].unique())
        s2o = state2onehot(max_n_steps=max_n_steps)
        n_actions = 2
        dim_in = s2o.len + n_actions + 1 + s2o.len # state-one-hot (prior) + action (prior, one-hot format) + reward (prior) + state-one-hot (current)
        dim_out = n_actions + 1 + 1 # action (current, one-hot format) + agent + block no.
        if parallel:
            res = joblib.Parallel(n_jobs=-1)(joblib.delayed(df2seq)(df, n_actions, s2o, dim_in, dim_out, agent) for agent in df['agent'].unique())
        else:
            res = []
            for aa, agent in enumerate(df['agent'].unique()):
                res.append(df2seq(df, n_actions, s2o, dim_in, dim_out, agent))
        x = []
        y = []
        for aa, agent in enumerate(df['agent'].unique()):
            x += res[aa][0]
            y += res[aa][1]
        self.x = torch.stack(x)
        self.y = torch.stack(y)
        self.len = len(self.x)
        self.dim_state_onehot = s2o.len

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
  
    def __len__(self):
        return self.len

def df2seq(df, n_actions, s2o, dim_in, dim_out, agent):
    cur_ag = df[df['agent'] == agent]
    x = []
    y = []
    for ss, block in enumerate(cur_ag['block'].unique()):
        cur_seq = cur_ag[cur_ag['block'] == block]
        n_steps = cur_seq.iloc[0]['n_steps']
        cur_seq.reset_index()
        n_epsteps = len(cur_seq) # n_eps x n_steps
        in_ = torch.zeros([n_epsteps, dim_in], dtype=torch.float32)
        out_ = torch.zeros([n_epsteps, dim_out], dtype=torch.float32)
        prior_state_onehot = s2o.trans_state((-1, -1))
        prior_act = [-1, -1]
        prior_rew = -1
        step_counts = 0
        for ee, ep in enumerate(cur_seq['episode'].unique()):
            cur_ep = cur_seq[cur_seq['episode'] == ep]
            for ss, step in enumerate(cur_ep['step'].unique()):
                # in_[ee * ss] = [cur_seq['state0'], cur_seq['state1']]
                step_df = cur_ep[cur_ep['step'] == step]
                state = (step_df.iloc[0]['state0'], step_df.iloc[0]['state1'])
                state_onehot = s2o.trans_state(state)
                in_[step_counts] = torch.tensor(list(prior_state_onehot) + prior_act + [prior_rew] + list(state_onehot))
                act_onehot = nn.functional.one_hot(torch.tensor([int(step_df.iloc[0]['act'])]), n_actions)
                out_[step_counts, :n_actions] = act_onehot[0]
                out_[step_counts, -2] = agent
                out_[step_counts, -1] = block
                prior_state_onehot = s2o.trans_state(state)
                prior_act = list(act_onehot[0])
                prior_rew = step_df.iloc[0]['rew']
                step_counts += 1
        x.append(in_)
        y.append(out_)
    return x, y

def out_seqs_each_agent(df, device='cpu'):
    seqs = {}
    x_all = []
    y_all = []
    for aa, id_ in enumerate(df['id'].unique()):
        _df = df[df['id'] == id_]
        _ds = mk_ds(_df, parallel=False)
        x = []
        y = []
        for nn in range(_ds.len):
            _x, _y = _ds.__getitem__(nn)
            x.append(_x)
            y.append(_y)
            x_all.append(_x)
            y_all.append(_y)
        seqs[id_] = [torch.stack(x).to(device), torch.stack(y).to(device)]
    return torch.stack(x_all).to(device), torch.stack(y_all).to(device), seqs

class state2onehot():
    def __init__(self, max_n_steps):
        max_n_steps = 3
        mp = mdp.mk_default_mdp(n_steps=max_n_steps)
        act_states = list(mp.prob_trans.keys())
        goal_states = list(mp.state_reward.keys())
        reward2state = {}
        for state in mp.state_reward.keys():
            reward2state[mp.state_reward[state]] = state
        # states = act_states + goal_states
        states = act_states
        E = np.eye(len(states), dtype=int)
        state2oh = {(-1, -1):tuple(np.zeros(len(states), dtype=int))}
        oh2state = {tuple(np.zeros(len(states), dtype=int)):(-1, -1)}
        for ss, state in enumerate(states):
            state2oh[state] = tuple(E[ss])
            oh2state[tuple(E[ss])] = state
        self.state2oh = state2oh
        self.oh2state = oh2state
        self.reward2state = reward2state
        self.len = len(states)
    
    def trans_state(self, state, rew=0, last_step=False):
        oh_state = self.state2oh[state]
        return oh_state

    def trans_onehot(self, onehot):
        return self.oh2state[onehot]

class actnet2agent():
    def __init__(self, net, max_n_steps=3):
        self.net = net.eval()
        self.net.eval()
        self.device = next(net.parameters()).device
        self.s2o = state2onehot(max_n_steps=max_n_steps)
        self.prior_state_onehot = torch.tensor(self.s2o.trans_state((-1, -1)))
        self.reset()

    def decide_action(self, state, avail_acts=[0, 1]):
        if self.prior_reward.item() == -1:
            self.hs = None
        state_onehot = torch.tensor(self.s2o.trans_state(tuple(state)))
        in_ = torch.cat([self.prior_state_onehot, self.prior_act, self.prior_reward, state_onehot], axis=0).view(1, 1, -1)
        in_ = in_.to(self.device)
        _act_pred, _, self.hs = self.net(in_, self.hs)
        _act_pred = _act_pred.cpu().detach().numpy().squeeze()
        act_prob = _act_pred / _act_pred.sum()
        act = np.random.choice(avail_acts, p=act_prob)
        return act, {0:act_prob[0], 1:act_prob[1]}

    def update(self, pre_state, act1, cur_state, r, avail_acts):
        self.prior_state = torch.tensor(pre_state, dtype=torch.float32)
        if len(avail_acts) == 0:
            last_step = True
        else:
            last_step = False
        self.prior_state_onehot = torch.tensor(self.s2o.trans_state(tuple(pre_state)))
        act_onehot = nn.functional.one_hot(torch.tensor(act1), self.net.output_size)
        self.prior_act = act_onehot
        self.prior_reward = torch.tensor([r,], dtype=torch.float32)

    def reset(self):
        self.prior_state = torch.zeros(self.net.dim_state, dtype=torch.float32)
        self.prior_act = torch.zeros(self.net.output_size, dtype=torch.float32)
        self.prior_reward = torch.tensor([-1,], dtype=torch.float32)
        self.hs = None
        self.q = {}

def load_net(enc, dec, net_path, suffix='', reset=False, device='cpu', verbose=False):
    train_loss = []
    valid_loss = []
    rew_epochs = []
    train_rew = []
    valid_rew = []
    if reset:
        if verbose:
            print('Reset the networks for {}{}'.format(net_path, suffix))
    else:
        enc_load = False
        dec_load = False
        try:
            mm = torch.load(net_path + suffix, weights_only=False, map_location=torch.device(device))
            enc.load_state_dict(mm['enc'])
            if verbose:
                enc_load = True
            dec.load_state_dict(mm['dec'])
            if verbose:
                dec_load = True
            train_loss = mm['train_loss']
            valid_loss = mm['valid_loss']
            rew_epochs = mm['rew_epochs']
            train_rew = mm['train_rew']
            valid_rew = mm['valid_rew']
            if verbose:
                print('Loaded {}{} at {} epoch'.format(net_path, suffix, len(train_loss)))
        except:
            if verbose:
                print('Failed to load: {}{}'.format(net_path, suffix))
                if enc_load:
                    print('Encoder loaded.')
                if dec_load:
                    print('Decoder loaded.')
    return enc, dec, [train_loss, valid_loss], [rew_epochs, train_rew, valid_rew]

def out_agent_idx2id(df, dict2={}):
    for id_ in df['id'].unique():
        agent_idx = df[df['id'] == id_].iloc[0]['agent']
        dict2[agent_idx] = id_
    return dict2

if __name__=='__main__':
    np.random.seed(0)
    n_steps_src = 2
    n_steps_trg = 3
    n_blocks = 3
    n_epochs = 5000
    # n_epochs = 100
    df_path = './data/bhv_mdp2.csv'
    suffix = '.valid'
    lr = 1e-5 # learning rate for NN optimizers
    n_gen_episodes = 100
    parallel = True
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print('Device: {}'.format(device))

    df = pd.read_csv(df_path)
    pids = list(df['id'].unique())
    for leave_pid in pids:
    # for leave_pid in ['none']:
        iddf_path = 'tmp/split_leave_{}.csv'.format(leave_pid)
        try:
            iddf = pd.read_csv(iddf_path)
            print('Loaded {}'.format(iddf_path))
        except:
            iddf = leave_one_participant_id(pids, leave_pid, sp_tv=[.9, .1])
            iddf.to_csv(iddf_path, index=False)
            print('Created {}'.format(iddf_path))

        net_path = 'tmp/mdp_it_s{}_s{}_leave_{}.pth'.format(n_steps_src, n_steps_trg, leave_pid)
        df_src_train, n_actions_src = load_df(df_path, n_steps_src, iddf[iddf['set'] == 'train']['id'], n_blocks)
        df_src_valid, _ = load_df(df_path, n_steps_src, iddf[iddf['set'] == 'valid']['id'], n_blocks)
        df_trg_train, n_actions_trg  = load_df(df_path, n_steps_trg, iddf[iddf['set'] == 'train']['id'], n_blocks)
        df_trg_valid, _ = load_df(df_path, n_steps_trg, iddf[iddf['set'] == 'valid']['id'], n_blocks)
        agent_idx2id = out_agent_idx2id(df_src_train)
        agent_idx2id = out_agent_idx2id(df_src_valid, agent_idx2id)
        mp_trg = mdp.mk_default_mdp(n_steps=n_steps_trg)
        s2o = state2onehot(max_n_steps=3)
        dim_state = s2o.len

        enc, dec, actnet = init_nets(dim_state, n_actions_trg, n_steps_src, n_steps_trg, device=device, verbose=False)
        enc, dec, losses, rew_rates = load_net(enc, dec, net_path, suffix, device=device, verbose=True)
        losses = [[], []]
        rew_rates = [[], [], []]
        n_done_epochs = len(losses[0])
        if n_epochs - n_done_epochs > 0:
            train_net(df_src_train, df_src_valid, df_trg_train, df_trg_valid, enc, dec, actnet, losses, rew_rates,
                    n_epochs - n_done_epochs, mp_trg, n_gen_episodes, net_path, batch_size=0, lr=lr,
                    device=device, sp_intv=100, sp_intv_rew=500, parallel=parallel)
