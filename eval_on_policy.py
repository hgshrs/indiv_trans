import sys
import importlib
import pandas as pd
import train_it_mdp as sv
importlib.reload(sv)
import n_choice_markov as mdp
importlib.reload(mdp)
import numpy as np
import matplotlib.pylab as plt
import pickle
import copy
from tqdm.contrib import tenumerate
import scipy.stats
import pingouin as pg
import scipy.stats

def get_percent_rewarding_act_at_final_choice(df):
    n_steps = df['step'].max() + 1
    df_ls = df[df['step'] == (n_steps - 1)]
    choice_rewarding_act = np.zeros(len(df_ls), dtype=int)
    rewarding_state = 's{}0'.format(n_steps)
    for epi in range(len(df_ls)):
        current_state = 's{}{}'.format(df_ls.iloc[epi]['state0'], df_ls.iloc[epi]['state1'])
        act = 'a{}'.format(df_ls.iloc[epi]['act'])
        rew_prob = df_ls.iloc[epi][current_state + act + rewarding_state]
        if rew_prob == .8:
            choice_rewarding_act[epi] = 1
        elif rew_prob == .2:
            choice_rewarding_act[epi] = 0
        else:
            choice_rewarding_act[epi] = -1
    if (choice_rewarding_act >= 0).sum() == 0:
        percent_rewarding_act = np.nan
    else:
        percent_rewarding_act = (choice_rewarding_act == 1).sum() / (choice_rewarding_act >= 0).sum()
    return percent_rewarding_act

if __name__=='__main__':
    n_reps = 499
    n_steps_src = 3
    n_steps_trg = 2
    dim_state = 5
    n_actions_trg = 2
    device = 'cpu'
    suffix = '.valid'
    dim_z = 2
    df_path = './data/bhv_mdp2.csv'
    df = pd.read_csv(df_path)
    res_csv = 'tmp/res_on_policy.csv'
    at_csv = 'tmp/at_on_policy.csv'


    # '''
    for n_steps_src, n_steps_trg in [[3, 2], [2, 3]]:
        df_path = './data/bhv_mdp2.csv'
        df = pd.read_csv(df_path)
        pids = list(df['id'].unique())

        net_path = 'tmp/mdp_it_s{}_s{}_leave_none.pth'.format(n_steps_src, n_steps_trg) # a model with all data for testing
        enc, dec, actnet = sv.init_nets(dim_state, n_actions_trg, n_steps_src, n_steps_trg, device=device, verbose=False)
        enc, dec, training_losses, rew_rates = sv.load_net(enc, dec, net_path, suffix, device=device, verbose=True)
        enc.eval(); dec.eval()

        mp_trg = mdp.mk_default_mdp(n_steps=n_steps_trg)


        seq_path = 'tmp/sqs_trans{}{}.pkl'.format(n_steps_src, n_steps_trg)
        try:
            with open(seq_path, 'rb') as f:
                x_src, y_src, seqs_src, x_trg, y_trg, seqs_trg = pickle.load(f)
            print('Loaded {}'.format(seq_path))
        except:
            df_src, _ = sv.load_df(df_path, n_steps_src, [])
            df_trg, _ = sv.load_df(df_path, n_steps_trg, [])
            sv.check_id_consistency(df_src, df_trg)
            x_src, y_src, seqs_src = sv.out_seqs_each_agent(df_src)
            x_trg, y_trg, seqs_trg = sv.out_seqs_each_agent(df_trg)
            with open(seq_path, 'wb') as f:
                pickle.dump([x_src, y_src, seqs_src, x_trg, y_trg, seqs_trg], f)
        ags = {}

        z_ags = np.zeros([len(pids), dim_z])
        for pp, pid in enumerate(pids):
            x_src, y_src = seqs_src[pid]
            z_seqs = enc(x_src).detach().mean(0).view(1, dim_z)
            z_ags[pp] = z_seqs
            w_ags = dec(z_seqs).detach()
            fixed_actnet, params = sv.put_w2net(actnet, w_ags[0])
            ags[pid] = sv.actnet2agent(copy.deepcopy(fixed_actnet))

        try:
            resdf = pd.read_csv(res_csv)
        except:
            resdf = pd.DataFrame()
        res = np.zeros([len(pids), 3, 3])
        for pp, pid in tenumerate(pids):
            df_trg, _ = sv.load_df(df_path, n_steps_trg, [pid])
            _pids = copy.copy(pids)
            _pids.remove(pid)
            # _pids = [pids[np.diag(np.dot((z_ags - z_ags[pp]), (z_ags - z_ags[pp]).T)).argmax()]]
            # print(pid, _pid)
            for bb, blk in enumerate(df_trg['block'].unique()):
                df_blk = df_trg[df_trg['block'] == blk]
                total_rew = 0
                mpdf = pd.DataFrame()
                for epi in range(df_blk['episode'].max() + 1):
                    td = df_blk.query('episode == @epi').iloc[:1]
                    mpdf = pd.concat([mpdf, td], axis=0)

                total_rew = int(df_blk['rew'].sum())
                percent_rewarding_act = get_percent_rewarding_act_at_final_choice(df_blk)
                _resdf = [n_steps_src, n_steps_trg, pid, bb, 'hum', total_rew, percent_rewarding_act]
                res_columns = ['n_steps_src', 'n_steps_trg', 'id', 'block', 'agent', 'total_rew', 'percent_rewarding_act']
                resdf = pd.concat([resdf, pd.DataFrame([_resdf], columns=res_columns)], axis=0)
                for rr in range(n_reps):
                    ag = ags[pid]; ag.reset()
                    genseq = mdp.generate_seq(ag, mp_trg, mpdf=mpdf, pid=pid)
                    total_rew = int(genseq['rew'].sum())
                    percent_rewarding_act = get_percent_rewarding_act_at_final_choice(genseq)
                    _resdf = [n_steps_src, n_steps_trg, pid, bb, 'org', total_rew, percent_rewarding_act]
                    resdf = pd.concat([resdf, pd.DataFrame([_resdf], columns=res_columns)], axis=0)

                    _pid = np.random.choice(_pids)
                    _ag = ags[_pid]; _ag.reset()
                    genseq2 = mdp.generate_seq(_ag, mp_trg, mpdf=mpdf, pid=_pid)
                    total_rew = int(genseq2['rew'].sum())
                    percent_rewarding_act = get_percent_rewarding_act_at_final_choice(genseq2)
                    _resdf = [n_steps_src, n_steps_trg, pid, bb, 'opp', total_rew, percent_rewarding_act]
                    resdf = pd.concat([resdf, pd.DataFrame([_resdf], columns=res_columns)], axis=0)
            resdf.to_csv(res_csv, index=False)
    # '''

    resdf = pd.read_csv(res_csv)
    pids = resdf['id'].unique()
    at = pd.DataFrame()
    at_columns = ['n_steps_src', 'n_steps_trg', 'id', 'block', 'agent', 'total_rew', 'percent_rewarding_act', 'diff_rew', 'diff_per']
    figure_n = 0
    for n_steps_src, n_steps_trg in [[3, 2], [2, 3]]:
        for pp, pid in enumerate(pids):
            for bb, blk in enumerate(resdf['block'].unique()):
                _res = resdf.query('n_steps_src == @n_steps_src & n_steps_trg == @n_steps_trg & id == @pid & block == @blk')
                hum_rew = _res[_res['agent'] == 'hum']['total_rew'].mean()
                hum_per = _res[_res['agent'] == 'hum']['percent_rewarding_act'].mean()
                dat = [n_steps_src, n_steps_trg, pid, blk, 'hum', hum_rew, hum_per, np.nan, np.nan]
                at = pd.concat([at, pd.DataFrame([dat], columns=at_columns)], axis=0)

                org_rew = _res[_res['agent'] == 'org']['total_rew'].mean()
                org_per = _res[_res['agent'] == 'org']['percent_rewarding_act'].mean()
                diff_rew = np.abs(_res[_res['agent'] == 'org']['total_rew'] - hum_rew).mean()
                diff_per = np.abs(_res[_res['agent'] == 'org']['percent_rewarding_act'] - hum_per).mean()
                dat = [n_steps_src, n_steps_trg, pid, blk, 'org', org_rew, org_per, diff_rew, diff_per]
                at = pd.concat([at, pd.DataFrame([dat], columns=at_columns)], axis=0)

                opp_rew = _res[_res['agent'] == 'opp']['total_rew'].mean()
                opp_per = _res[_res['agent'] == 'opp']['percent_rewarding_act'].mean()
                diff_rew = np.abs(_res[_res['agent'] == 'opp']['total_rew'] - hum_rew).mean()
                diff_per = np.abs(_res[_res['agent'] == 'opp']['percent_rewarding_act'] - hum_per).mean()
                dat = [n_steps_src, n_steps_trg, pid, blk, 'opp', opp_rew, opp_per, diff_rew, diff_per]
                at = pd.concat([at, pd.DataFrame([dat], columns=at_columns)], axis=0)
    at.to_csv(at_csv, index=False)

    at1 = at.query('agent != "hum"')
    aov = pg.rm_anova(data=at1, dv='diff_rew', subject='id', within=['agent', 'n_steps_trg'])
    print(aov.round(3))
    kwargs = {'padjust':'bonf', 'parametric':True, 'alternative':'two-sided'}
    post_hocs = pg.pairwise_tests(data=at1, dv='diff_rew', within=['n_steps_trg', 'agent'], subject='id', **kwargs)
    print(post_hocs.round(3))

    aov = pg.rm_anova(data=at1, dv='diff_per', subject='id', within=['agent', 'n_steps_trg'])
    print(aov.round(3))
    kwargs = {'padjust':'bonf', 'parametric':True, 'alternative':'two-sided'}
    post_hocs = pg.pairwise_tests(data=at1, dv='diff_per', within=['n_steps_trg', 'agent'], subject='id', **kwargs)
    print(post_hocs.round(3))
