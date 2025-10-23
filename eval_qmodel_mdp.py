import sys
import importlib
import n_choice_markov as mdp
importlib.reload(mdp)
import train_it_mdp as sv
importlib.reload(sv)
import pandas as pd
import numpy as np
import scipy.optimize
from tqdm import tqdm
from tqdm.contrib import tenumerate
import itertools
import copy

def out_llik(dfs, alpha=.5, beta=1, gamma=0, init_q=.5, reduction='mean', negative=True):
    ag = mdp.q_agent(alpha=alpha, beta=beta, gamma=gamma, init_q=init_q)
    llik = []
    match = []

    for agent in dfs['agent'].unique():
        dfa = dfs[dfs['agent'] == agent]
        for blk in dfa['block'].unique():
            df_blk = dfa[dfa['block'] == blk]
            n_steps = df_blk.iloc[0]['n_steps']
            mp = mdp.mk_default_mdp(n_steps=n_steps)
            ag.reset()
            for stp in range(len(df_blk)):
                df_stp = df_blk.iloc[stp]
                cur_state = (df_stp['state0'], df_stp['state1'])
                avail_acts = mp.avail_acts(cur_state)
                _act, lk = ag.decide_action(cur_state, avail_acts)
                llik.append(np.log(lk[int(df_stp['act'])]))
                if df_stp['act'] == _act:
                    match.append(1)
                else:
                    match.append(0)

                if df_stp['step'] == n_steps - 1:
                    if df_stp['rew'] == 1:
                        next_state = (1, 0)
                    elif df_stp['rew'] == 0:
                        next_state = (1, 1)
                    ag.update(cur_state, df_stp['act'], next_state, df_stp['rew'], [])
                else:
                    df_next = df_blk.iloc[stp + 1]
                    next_state = (df_next['state0'], df_next['state1'])
                    avail_acts = mp.avail_acts(next_state)
                    ag.update(cur_state, df_stp['act'], next_state, df_stp['rew'], avail_acts)
    if reduction == 'none':
        out = np.array(llik)
        match = np.array(match)
    elif reduction == 'mean':
        out = np.mean(llik)
        match = np.mean(match)
    elif reduction == 'sum':
        out = np.sum(llik)
        match = np.sum(match)
    if negative:
        out *= -1
    return out, match

def optimize_fun(x, dfs):
    alpha = x[0]
    beta = x[1]
    gamma = x[2]
    init_q = x[3]
    nllik, match = out_llik(dfs, alpha, beta, gamma, init_q, reduction='mean', negative=True)
    return nllik

if __name__=='__main__':
    n_blocks = 3

    df_path = './data/bhv_mdp2.csv'
    df = pd.read_csv(df_path)
    pids = list(df['id'].unique())

    # ===============================
    # Estimate RL params for each step-task and participant
    # ===============================
    dfp = pd.DataFrame()
    for n_steps in [2, 3]:
        df, n_actions  = sv.load_df(df_path, n_steps, pids, n_blocks)

        alpha, beta, gamma, init_q = .5, 5, 1, 0
        ag = mdp.q_agent(alpha=alpha, beta=beta, gamma=gamma, init_q=init_q)
        for _id in tqdm(df['id'].unique()):
            df_ag = df[df['id'] == _id]
            res = scipy.optimize.minimize(
                    optimize_fun,
                    x0=[.5, 0, 1, .5],
                    args=(df_ag),
                    bounds=[(.001, .999), (.001, 50), (.001, .999), (.001, .999)],
                    )
            # print('{}, {:.3f}, alpha: {:.3f}, beta: {:.3f}, gamma: {:.3f}, init_q: {:.3f}'.format(_id[:4], res.fun, *res.x))
            seqdf = pd.Series()
            seqdf['id'] = _id
            seqdf['agent'] = df_ag.iloc[0]['agent']
            seqdf['alpha'] = res.x[0]
            seqdf['beta'] = res.x[1]
            seqdf['gamma'] = res.x[2]
            seqdf['init_q'] = res.x[3]
            seqdf['nllik'] = res.fun
            seqdf['n_steps'] = n_steps
            dfp = pd.concat([dfp, seqdf.to_frame().T], axis=0)
            dfp.to_csv('tmp/param_qagents.csv', index=False)
    dfp = pd.read_csv('tmp/param_qagents.csv')


    # ===============================
    # Situation SX (no transfer)
    # ===============================
    set_src = [2, 3]
    dfres_a = pd.DataFrame()
    for n_steps in set_src:
        print('[SX] {}-steps MDP.'.format(n_steps))
        df_trg, _ = sv.load_df(df_path, n_steps, pids, n_blocks)
        for pid in tqdm(pids):
            train_ids = copy.copy(pids)
            train_ids.remove(pid)
            dfp_train_ids = dfp.query('id in @train_ids & n_steps == @n_steps')
            alpha, beta, gamma, init_q = dfp_train_ids['alpha'].mean(), dfp_train_ids['beta'].mean(), dfp_train_ids['gamma'].mean(), dfp_train_ids['init_q'].mean()

            dfa = df_trg[df_trg['id'] == pid]
            nllik, match = out_llik(dfa, alpha, beta, gamma, init_q, reduction='mean', negative=True)

            dfa_ = df_trg.query('id != @pid')
            nllik_opp, match_opp = out_llik(dfa_, alpha, beta, gamma, init_q, reduction='mean', negative=True)

            res = {'id': pid, 'n_steps': n_steps, 'nllik_org': nllik, 'nllik_opp': nllik_opp, 'match_org': match, 'match_opp': match_opp}
            dfres_a = pd.concat([dfres_a, pd.Series(res).to_frame().T], axis=0)
            dfres_a.to_csv('tmp/qagent_nllik_a.csv', index=False)


    # ===============================
    # Situation SY (transfer)
    # ===============================
    set_trg = [2, 3]
    prod_set = list(itertools.product(set_src, set_trg))
    dfres_b = pd.DataFrame()
    for n_steps_src, n_steps_trg in prod_set:
        print('[SY] {}-steps --> {}-steps MDP.'.format(n_steps_src, n_steps_trg))
        df_trg, _ = sv.load_df(df_path, n_steps_trg, [], n_blocks)
        for pid in tqdm(pids):

            param = dfp.query('id == @pid & n_steps == @n_steps_src').iloc[0]
            alpha, beta, gamma, init_q = param['alpha'], param['beta'], param['gamma'], param['init_q'] 
            dfa = df_trg[df_trg['id'] == pid]
            nllik, match = out_llik(dfa, alpha, beta, gamma, init_q, reduction='mean', negative=True)

            dfa_ = df_trg.query('id != @pid')
            _ids = dfa_['id'].unique()
            nllik_opp_ = np.zeros(len(_ids))
            match_opp_ = np.zeros(len(_ids))
            for ii, id_opp in enumerate(_ids):
                param = dfp.query('id == @id_opp & n_steps == @n_steps_src').iloc[0]
                alpha, beta, gamma, init_q = param['alpha'], param['beta'], param['gamma'], param['init_q']
                nllik_opp_[ii], match_opp_[ii] = out_llik(dfa, alpha, beta, gamma, init_q, reduction='mean', negative=True)
            nllik_opp = nllik_opp_.mean()
            match_opp = match_opp_.mean()

            res = {
                    'id': pid,
                    'n_steps_src': n_steps_src,
                    'n_steps_trg': n_steps_trg,
                    'nllik_org': nllik,
                    'nllik_opp': nllik_opp,
                    'match_org': match,
                    'match_opp': match_opp,
                    }
            dfres_b = pd.concat([dfres_b, pd.Series(res).to_frame().T], axis=0)
            dfres_b.to_csv('tmp/qagent_nllik_b.csv', index=False)
