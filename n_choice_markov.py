import importlib
import sys
import copy
import numpy as np
import random
import matplotlib.pylab as plt
import pandas as pd
from tqdm import tqdm
import load_bhv as lb
importlib.reload(lb)
import copy
import uuid

class MDP():
    def __init__(self, n_states_per_step, acts, set_prob_option, state_reward, goal_steps=None, trans_change_prob=0):
        self.n_steps = len(n_states_per_step) - 1
        n_acts = len(acts)
        self.state_reward = state_reward
        self.set_prob_option = set_prob_option
        self.trans_change_prob = trans_change_prob
        self.n_states_per_step = n_states_per_step
        if goal_steps is None:
            self.goal_steps = list([self.n_steps])


        self.prob_trans = {}
        re_set_flag = True
        while re_set_flag:
            for ll in range(self.n_steps):
                for ss0 in range(n_states_per_step[ll]):
                    state = (ll, ss0)
                    self.prob_trans[state] = {}
                    for act in acts:
                        self.prob_trans[state][act] = {}
                        # prob_option = random.choice(set_prob_option).copy()
                        next_states = []
                        for ss1 in range(n_states_per_step[ll + 1]):
                            next_states.append((ll + 1, ss1))
                        self.set_trans(state, act, next_states)
            re_set_flag = not(check_expected_root(self, threshold=.6))

    def set_trans(self, state, act, next_states):
        if len(next_states) == 0:
            next_states = list(self.prob_trans[state][act].keys())
        n_next_states = len(next_states)
        _prob_option = random.choice(self.set_prob_option[n_next_states]).copy()
        for next_state in next_states:
            idx = np.random.choice(range(len(_prob_option)), p=np.ones(len(_prob_option)) / len(_prob_option))
            self.prob_trans[state][act][next_state] = _prob_option[idx]
            _prob_option.pop(idx)

    def trans(self, cur_state, act):
        avail_states = tuple(self.prob_trans[cur_state][act].keys())
        p = []
        for next_state in avail_states:
            p.append(self.prob_trans[cur_state][act][next_state])
        state_idx = np.random.choice(range(len(avail_states)), p=p)
        pre_state = cur_state
        cur_state = avail_states[state_idx]
        return cur_state

    def reward(self, state):
        try:
            r = self.state_reward[state]
        except:
            r = 0
        return r

    def is_goal(self, state):
        if state[0] in self.goal_steps:
            return True
        else:
            return False

    def env_return(self, state, act):
        next_state = self.trans(state, act)
        r = self.reward(next_state)
        goal = self.is_goal(next_state)
        return next_state, r, goal

    def avail_acts(self, state):
        try:
            # return list(self.prob_trans[state].keys())
            return list(np.sort(np.array(list(self.prob_trans[state].keys()))))
        except:
            return []

    def change_rand_trans(self, verbose=False):
        if np.random.choice([True, False], p=(self.trans_change_prob, 1 - self.trans_change_prob)):
            re_set_flag = True
            org = copy.deepcopy(self.prob_trans)
            while re_set_flag:
                self.prob_trans = copy.deepcopy(org)
                change_trans_rand_state_action(self)
                re_set_flag = not(check_expected_root(self, threshold=.6))
            if verbose:
                print('MDP has been changed.')
            return True
        else:
            return False

    def set_pre_defined_trans(self, mpseries):
        for s0 in list(self.prob_trans.keys()):
            for a in list(self.prob_trans[s0].keys()):
                for s1 in list(self.prob_trans[s0][a].keys()):
                    trans_txt = 's{}{}a{}s{}{}'.format(s0[0], s0[1], a, s1[0], s1[1])
                    self.prob_trans[s0][a][s1] = mpseries[trans_txt]

def change_trans_rand_state_action(mp):
    state = random.choice(list(mp.prob_trans.keys()))
    for act in list(mp.prob_trans[state].keys()):
        # act = random.choice(list(mp.prob_trans[state].keys()))
        not_changed = True
        pre = mp.prob_trans[state][act].copy()
        while not_changed:
            mp.set_trans(state, act, [])
            if pre == mp.prob_trans[state][act]:
                not_changed = True
            else:
                not_changed = False
    return mp

def viz_mdp(ax, mp, focusing_states=[], size=40, shift=.1):
    ax.cla()
    states = list(mp.prob_trans.keys())
    cmap_tab = plt.get_cmap('tab10')
    act_lines = {}
    for state in states:
        _shift = 0
        for aa, act in enumerate(list(mp.prob_trans[state].keys())):
            for transed_state in list(mp.prob_trans[state][act].keys()):
                p = mp.prob_trans[state][act][transed_state]
                x = np.array([state[1] + _shift, transed_state[1]])
                y = np.array([state[0], transed_state[0]])
                l = ax.plot(x, y, c=cmap_tab(aa), lw=p * 10)[0]
                act_lines[act] = l
                _shift += shift
        if state in focusing_states:
            mfc='c'
        else:
            mfc='w'
        ax.plot(state[1], state[0], 'ko', mfc=mfc, ms=size)
        ax.text(state[1], state[0], '{}'.format(state), size=.2 * size, va='center', ha='center')
    ax.legend(list(act_lines.values()), list(act_lines.keys()))

    for state in list(mp.state_reward.keys()):
        if state in focusing_states:
            mfc='c'
        else:
            mfc='w'
        ax.plot(state[1], state[0], 'ko', mfc=mfc, ms=size)
        ax.text(state[1], state[0], mp.state_reward[state], size=.5 * size, va='center', ha='center')
    ax.axis('equal')
    ax.set_ylim(-1, ax.get_ylim()[1] * 1.3)
    # ax.set_xlim(ax.get_xlim() * 1.1)

class q_agent():
    def __init__(self, alpha, beta, gamma, init_q=0):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.init_q = init_q
        self.q = {}

    def decide_action(self, state, avail_acts):
        n_avail_acts = len(avail_acts)
        upq = np.zeros(n_avail_acts)
        for cc, _act in enumerate(avail_acts):
            upq[cc] = self.out_q(state, _act)

        m = np.zeros(len(upq))
        for cc, _act in enumerate(avail_acts):
            m[cc] = np.exp(self.beta * self.out_q(state, _act))
        if m.sum() == 0.:
            m[:] = 1 / len(avail_acts)
        else:
            m /= m.sum()
        act = np.random.choice(avail_acts, p=m)
        prob = {}
        for aa, act in enumerate(avail_acts):
            prob[act] = m[aa]
        return act, prob

    def out_q(self, state, act):
        try:
            self.q[state][act] = self.q[state][act]
        except:
            try:
                self.q[state][act] = self.init_q
            except:
                self.q[state] = {}
                self.q[state][act] = self.init_q
        return self.q[state][act]

    def update(self, pre_state, act, cur_state, r, avail_acts):
        q = self.out_q(pre_state, act)

        if len(avail_acts) > 0:
            upq = np.zeros(len(avail_acts))
            for cc, _act in enumerate(avail_acts):
                upq[cc] = self.out_q(cur_state, _act)
            self.q[pre_state][act] = (1 - self.alpha) * q + self.alpha * (r + self.gamma * upq.max())
        else:
            self.q[pre_state][act] = (1 - self.alpha) * q + self.alpha * r
        return self.q

    def reset(self):
        self.q = {}

def sim(mp, ag, start_state=(0, 0), mpseries=[], verbose=False, viz_ax=None):
    cur_state = start_state
    goal = False
    avail_acts = mp.avail_acts(cur_state)
    prob_trans = np.zeros(mp.n_steps, dtype='object')
    state = np.zeros(mp.n_steps, dtype='object')
    act = np.zeros(mp.n_steps, dtype=int)
    rew = np.zeros(mp.n_steps)
    q = np.zeros(mp.n_steps, dtype='object')
    trans_change = np.zeros(mp.n_steps, dtype=int)
    if len(mpseries) > 0:
        mp.set_pre_defined_trans(mpseries)
    else:
        trans_change[0] = mp.change_rand_trans(verbose=verbose)
    for cc in range(mp.n_steps):
        prob_trans[cc] = copy.deepcopy(mp.prob_trans)
        state[cc] = cur_state
        act1, _ = ag.decide_action(cur_state, avail_acts)
        act[cc] = act1
        pre_state = cur_state
        cur_state, r, goal = mp.env_return(cur_state, act1)
        rew[cc] = r
        avail_acts = mp.avail_acts(cur_state)
        ag.update(pre_state, act1, cur_state, r, avail_acts)
        q[cc] = copy.deepcopy(ag.q)
        if viz_ax != None:
            viz_mdp(ax, mp, focusing_states=[pre_state, cur_state])
            plt.pause(.1)
    return prob_trans, state, act, rew, q, trans_change

def trans_txt2prob(txt, prob_trans):
    tl = txt.split('_')[1:]
    tl = [int(s) for s in tl]
    return prob_trans[(tl[0], tl[1])][tl[2]][(tl[3], tl[4])]

def q_txt2q(txt, q):
    tl = txt.split('_')[1:]
    tl = [int(s) for s in tl]
    return q[(tl[0], tl[1])][tl[2]]

def generate_seq(ag, mp, n_episodes=100, mpdf=[], verbose=False, ax=None):
    n_steps = mp.n_steps
    if len(mpdf) > 0:
        n_episodes = len(mpdf['episode'].unique())
    ep_trans = np.zeros([n_episodes, n_steps], dtype='object')
    ep_state = np.zeros([n_episodes, n_steps], dtype='object')
    ep_act = np.zeros([n_episodes, n_steps], dtype=int)
    ep_rew = np.zeros([n_episodes, n_steps])
    ep_q = np.zeros([n_episodes, n_steps], dtype='object')
    ep_mdp_change = np.zeros([n_episodes, n_steps], dtype=int)
    for ee in range(n_episodes):
        if len(mpdf) > 0:
            prob_trans, state, act, rew, q, trans_change = sim(mp, ag, mpseries=mpdf[mpdf['episode'] == ee].iloc[0], verbose=verbose, viz_ax=ax)
        else:
            prob_trans, state, act, rew, q, trans_change = sim(mp, ag, verbose=verbose, viz_ax=ax)
        ep_mdp_change[ee, :] = trans_change
        ep_trans[ee, :] = prob_trans
        ep_state[ee, :] = state
        ep_act[ee, :] = act
        ep_rew[ee, :] = rew
        ep_q[ee, :] = q

    n_episodes, n_steps = ep_state.shape
    data = {
            'episode': [],
            'step': [],
            'trans_change': [],
            'state0': [],
            'state1': [],
            }
    for s0 in list(ep_trans[-1, -1].keys()):
        for a in list(ep_trans[-1, -1][s0].keys()):
            for s1 in list(ep_trans[-1, -1][s0][a].keys()):
                trans_txt = 'trans_{}_{}_{}_{}_{}'.format(s0[0], s0[1], a, s1[0], s1[1])
                data[trans_txt] = []
    data['state0'] = []
    data['state1'] = []
    data['act'] = []
    data['rew'] = []
    for s0 in list(ep_q[-1, -1].keys()):
        for a in list(ep_q[-1, -1][s0].keys()):
            q_txt = 'q_{}_{}_{}'.format(s0[0], s0[1], a)
            data[q_txt] = []
    for ee in range(n_episodes):
        for ss in range(n_steps):
            data['episode'].append(ee)
            data['step'].append(ss)
            data['trans_change'].append(ep_mdp_change[ee, ss])
            for s0 in list(ep_trans[-1, -1].keys()):
                for a in list(ep_trans[-1, -1][s0].keys()):
                    for s1 in list(ep_trans[ee, -1][s0][a].keys()):
                        trans_txt = 'trans_{}_{}_{}_{}_{}'.format(s0[0], s0[1], a, s1[0], s1[1])
                        data[trans_txt].append(trans_txt2prob(trans_txt, ep_trans[ee, ss]))
            data['state0'].append(ep_state[ee, ss][0])
            data['state1'].append(ep_state[ee, ss][1])
            data['act'].append(ep_act[ee, ss])
            data['rew'].append(ep_rew[ee, ss])
            for s0 in list(ep_q[-1, -1].keys()):
                for a in list(ep_q[-1, -1][s0].keys()):
                    q_txt = 'q_{}_{}_{}'.format(s0[0], s0[1], a)
                    try:
                        data[q_txt].append(q_txt2q(q_txt, ep_q[ee, ss]))
                    except:
                        data[q_txt].append(ag.init_q)
    # for key in list(data.keys()):
        # print(key, len(data[key]))
    df = pd.DataFrame(data)
    if verbose:
        print(df)
    return df

def randout_agent_params():
    alpha = np.random.uniform(.1, 1.)
    beta = np.random.uniform(.1, 25)
    gamma = 1.
    init_q = 0.
    return alpha, beta, gamma, init_q

def mk_default_mdp(n_steps=1, trans_change_prob=1/5):
    # n_steps = len(n_states_per_step) - 1
    if n_steps == 1:
        n_states_per_step = [1, 2]
    elif n_steps == 2:
        n_states_per_step = [1, 2, 2]
    elif n_steps == 3:
        n_states_per_step = [1, 2, 2, 2]
    elif n_steps == 4:
        n_states_per_step = [1, 2, 2, 2, 2]
    state_reward = {(n_steps, 0): 1,
                    (n_steps, 1): 0 }
    acts = [0, 1]
    set_prob_option = {
            2: [[.2, .8], [.6, .4], [.8, .2], [.4, .6]],
            3: [[1/4, 1/4, 2/4], [1/3, 1/3, 1/3], [.1, .1, .8],],
            }
    trans_change_prob = trans_change_prob
    mp = MDP(n_states_per_step, acts, set_prob_option, state_reward, trans_change_prob=trans_change_prob)
    return mp

def check_expected_root(mp, threshold=.6):
    target_state = list(mp.state_reward.keys())[np.argmax(mp.state_reward.values())]
    n_steps = target_state[0]
    prob_trans = copy.deepcopy(mp.prob_trans)
    target_root = [[]] * (n_steps + 1)
    target_root_p = [0] * n_steps
    target_root[n_steps] = target_state
    new_target_state = target_state
    for sstt in range(n_steps)[::-1]:
        old_p = 0
        for ss in range(mp.n_states_per_step[sstt]):
            st = prob_trans[(sstt, ss)]
            for act in st.keys():
                try:
                    p = st[act][new_target_state]
                except:
                    p = 0.
                if p > old_p:
                    old_p = p
                    target_root[sstt] = (sstt, ss)
        new_target_state = target_root[sstt]
        target_root_p[sstt] = old_p
    if np.min(target_root_p) >= threshold:
        return True
    else:
        return False

if __name__=='__main__':
    # Make a table for the agent parameters
    table_path = 'tmp/table_agent_params.csv'
    # yn = input('Do you want update the table for the agent parameters? [y/]: ')
    # if yn == 'y':
    if False:
        n = 10000
        pt = pd.DataFrame()
        pmat = np.zeros([n, 4])
        pmat[:, 0] = np.random.uniform(.1, 1., size=n) # alpha
        pmat[:, 1] = np.random.uniform(.1, 25, size=n) # beta
        pmat[:, 2] = 1. # gamma
        pmat[:, 3] = 0. # init_q
        pdf = pd.DataFrame(pmat, columns=['alpha', 'beta', 'gamma', 'init_q'])
        pdf.to_csv(table_path, index=False)
        print('Updated {}'.format(table_path))
    else:
        print('{} was not updated.'.format(table_path))
    pdf = pd.read_csv(table_path)

    n_agents = 500 # Number of agents
    plt.figure(2).clf()
    for aa in range(n_agents):
        plt.plot(pdf.loc[aa]['alpha'], pdf.loc[aa]['beta'], 'o', ms=10, mec='k')
    plt.xlabel('Learning rate, alpha')
    plt.ylabel('Inv temperature, beta')
    # sys.exit()

    set_n_steps = [1, 2, 3]
    # set_n_steps = [3]
    n_blocks = 4 # Number of sequences for each agent
    n_episodes = 50 # Length of a sequence
    plt.figure(1).clf(); ax = plt.subplot(111)
    df = pd.DataFrame()
    for aa in tqdm(range(n_agents)):
        # alpha, beta, gamma, init_q = randout_agent_params()
        alpha = pdf.loc[aa]['alpha']; beta = pdf.loc[aa]['beta']; gamma = pdf.loc[aa]['gamma']; init_q = pdf.loc[aa]['init_q']
        id_ = uuid.uuid4().urn[-4:] # ID
        # print(alpha, beta, gamma, init_q)
        ag = q_agent(alpha=alpha, beta=beta, gamma=gamma, init_q=init_q)
        for n_steps in set_n_steps:
            for bb in range(n_blocks):
                ag.reset()
                mp = mk_default_mdp(n_steps=n_steps)
                seqdf = generate_seq(ag, mp, n_episodes, verbose=False, ax=None)
                viz_mdp(ax, mp)
                seqdf['id'] = id_
                seqdf['agent'] = aa
                seqdf['agent_alpha'] = alpha
                seqdf['agent_beta'] = beta
                seqdf['agent_gamma'] = gamma
                seqdf['agent_init_q'] = init_q
                seqdf['n_steps'] = n_steps
                seqdf['block'] = bb
                df = pd.concat([df, seqdf])
    df.to_csv('./data/art_mdp.csv', index=False)
    df = pd.read_csv('./data/art_mdp.csv')

    fg = plt.figure(3)
    lb.plot_seq(fg, df, agent=0, n_steps=3, block=1)
    plt.figure(4).clf()
    plt.bar(*lb.out_step_ave(df))
    plt.ylim([.3, .7])

