import sys
import importlib
import pandas as pd
import train_it_mdp as sv
importlib.reload(sv)
import pickle
import numpy as np
import n_choice_markov as mdp
importlib.reload(mdp)

if __name__=='__main__':
    df_path = './data/bhv_mdp2.csv'
    df = pd.read_csv(df_path)
    pids = list(df['id'].unique())
    n_steps_src = 3
    n_steps_trg = 2
    n_actions_trg = 2
    dim_state = 5
    dim_z = 2
    device = 'cpu'
    suffix = '.valid'
    n_gen_episodes = 300 * 10

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

    mp = mdp.mk_default_mdp(n_steps=n_steps_trg)
    ans = {}
    for pid in pids:
        net_path = 'tmp/mdp_it_s{}_s{}_leave_{}.pth'.format(n_steps_src, n_steps_trg, pid)
        enc, dec, actnet = sv.init_nets(dim_state, n_actions_trg, n_steps_src, n_steps_trg, device=device, verbose=False)
        enc, dec, training_losses, rew_rates = sv.load_net(enc, dec, net_path, suffix, device=device, verbose=False)
        enc.eval(); dec.eval()

        if len(training_losses[0]) > 0:
            # human behaviour
            df_ = df.query('id == @pid & n_steps == @n_steps_trg')
            rew_rate_hum = df_['rew'].mean() * n_steps_trg

            # EIDT behaviour
            enc.eval(); dec.eval()

            x, y = seqs_src[pid]
            z_seqs = enc(x).detach().mean(0).view(1, dim_z)
            w_ags = dec(z_seqs).detach()
            fixed_actnet, params = sv.put_w2net(actnet, w_ags[0])
            ag = sv.actnet2agent(fixed_actnet.eval())
            genseq = mdp.generate_seq(ag, mp, n_episodes=n_gen_episodes, verbose=False, ax=None)
            rew_rate_edm = genseq['rew'].mean() * n_steps_trg
            print('[Reward rate] hum: {:.3f}, edm: {:.3f}'.format(rew_rate_hum, rew_rate_edm))
