import sys
import importlib
import pandas as pd
import scipy.stats
import matplotlib.pylab as plt
import numpy as np
import itertools
from statsmodels.stats.anova import AnovaRM
import pingouin as pg
from statsmodels.stats.multicomp import pairwise_tukeyhsd
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"
import warnings
warnings.simplefilter('ignore', FutureWarning)
print('FutureWaning is diabled')
import train_it_mdp as sv
importlib.reload(sv)
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from tqdm import tqdm

def remove_ss_cmbs(cmbs, removing_cmbs=[(2, 2), (3, 3)]):
    for removing_cmb in removing_cmbs:
        try:
            cmbs.remove(removing_cmb)
        except:
            pass
    return cmbs

def store_anova_table_a(dft, _df, model):
    dft_ = pd.DataFrame()
    dft_['id'] = _df['id']
    dft_['dom'] = dom_src
    dft_['model'] = model
    dft_['org_opp'] = 'org'
    dft_['nllik'] = _df['nllik_org']
    dft_['match'] = _df['match_org']
    dft = pd.concat([dft, dft_], axis=0)
    # dft_['org_opp'] = 'opp'
    # dft_['nllik'] = _df['nllik_opp']
    # dft = pd.concat([dft, dft_], axis=0)
    return dft

def store_anova_table_b(dft, _df, model):
    dft_ = pd.DataFrame()
    dft_['id'] = _df['id']
    dft_['dom_src'] = dom_src
    dft_['dom_trg'] = dom_trg
    dft_['model'] = model
    dft_['org_opp'] = 'org'
    dft_['nllik'] = _df['nllik_org']
    dft_['match'] = _df['match_org']
    dft = pd.concat([dft, dft_], axis=0)
    dft_['org_opp'] = 'opp'
    dft_['nllik'] = _df['nllik_opp']
    dft_['match'] = _df['match_opp']
    dft = pd.concat([dft, dft_], axis=0)
    return dft

def plot_paired(a, b, x, var='nllik', ticks=[], xlabel='Transfer direction', ylabel='Negative log-likelihood', show_indiv=True, whis=[0, 100]):
    boxplot_params = {'widths':[.8], 'whis':whis, 'patch_artist':True, 'showfliers':False}
    b1 = plt.boxplot(a[var], positions=[x], **boxplot_params)
    b2 = plt.boxplot(b[var], positions=[x+1], **boxplot_params)
    b1['boxes'][0].set_facecolor('white')
    b2['boxes'][0].set_facecolor('gray')
    if show_indiv:
        for id_ in a['id']:
            a_ = a[a['id'] == id_].iloc[0][var]
            b_ = b[b['id'] == id_].iloc[0][var]
            # plt.plot([x+.2, x+.8], [a_, b_], 'o-k', alpha=.2)
            plt.plot([x+.2, x+.8], [a_, b_], '-k', lw=.2, zorder=10)
            plt.plot([x+.2, x+.8], [a_, b_], 'ok', mfc='w', zorder=11)
    if len(ticks) == 2:
        plt.xticks([x+0, x+1], ticks)
    elif len(ticks) == 1:
        plt.xticks([x + .5], ticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.ylim(ylim)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    return b1, b2


if __name__=='__main__':
    df = pd.read_csv('./data/bhv_mdp2.csv')
    ids = list(df['id'].unique())
    fig_n = 0
    axlabels = {
            'nllik':'Negative log-likelihood',
            'match': 'Rate for behaviour matched',
            'diff_rew':'Error in total reward',
            'diff_per':'Error in high-rew selected',
            'z1':r'$z_1$',
            'z2':r'$z_2$',
            'alpha':r'Learning rate $q_\mathrm{lr}$',
            'beta':r'Inverse temperature $q_\mathrm{it}$',
            'gamma':r'Discount rate $q_\mathrm{dr}$',
            'init_q':r'Initialized value for $Q$ $q_\mathrm{init}$',
            'err_rew':'Error in total reward',
            'err_act':'Error in rate for high-rew act',
            'd_alpha':r'Abs error in learning rate $q_\mathrm{lr}$',
            'd_beta':r'Abs error in inverse temperature $q_\mathrm{lr}$',
            }
    ylims = {'nllik':[.2, 1.], 'match': [.37, .9]}
    cmap = plt.colormaps['tab10']

    # =================================
    # Evaluation in Situation SX (no transfer)
    # =================================
    # Get Situation SX
    doms_src = [2, 3]
    res_cdm = pd.read_csv('tmp/qagent_nllik_a.csv').drop_duplicates(subset=['id', 'n_steps'], keep='last', ignore_index=True)
    res_act = pd.read_csv('tmp/actnet_loss_a.csv').drop_duplicates(subset=['id', 'n_steps'], keep='last', ignore_index=True)
    dfa = pd.DataFrame()
    for dom_src in doms_src:
        _df = res_cdm.query('id in @ids & n_steps == @dom_src').sort_values(by='id')
        dfa = store_anova_table_a(dfa, _df, 'cgm')
        loss_cgm_org, loss_cgm_opp = _df['nllik_org'], _df['nllik_opp']

        _df = res_act.query('id in @ids & n_steps == @dom_src').sort_values(by='id')
        dfa = store_anova_table_a(dfa, _df, 'act')
    dfa = dfa[dfa['org_opp'] == 'org']

    # ylim = [.2, .85]
    for vv, var in enumerate(['nllik', 'match']):
        kwargs = {'dv':var, 'padjust':'bonf', 'parametric':True, 'alternative':'two-sided'}
        print('\nANOVA and multicomparison cmparing CG vs TS in {}'.format(var))
        aov = pg.rm_anova(data=dfa[dfa['org_opp'] == 'org'], dv=var, subject='id', within=['model', 'dom'])
        print(aov.round(3))
        post_hocs = pg.pairwise_tests(data=dfa[dfa['org_opp'] == 'org'], within=['dom', 'model'], subject='id', **kwargs)
        print(post_hocs.round(3))

        plt.figure(fig_n).clf(); fig_n += 1
        a, b = dfa.query('model=="cgm" & dom==2 & org_opp=="org"'), dfa.query('model=="act" & dom==2 & org_opp=="org"')
        plot_paired(a, b, 0, var, ylabel=axlabels[var])
        a, b = dfa.query('model=="cgm" & dom==3 & org_opp=="org"'), dfa.query('model=="act" & dom==3 & org_opp=="org"')
        b1, b2 = plot_paired(a, b, 2.5, var, ylabel=axlabels[var])
        plt.xticks([.5, 3.0], ['2-step', '3-step'])
        # plt.legend([b1[0], b2[0]], ['Cognitive model', 'Task solver'], loc='lower right')
        if var == 'nllik':
            lg = plt.legend([b1['boxes'][0], b2['boxes'][0]], ['Cognitive model', 'Task solver'], loc='upper right'); lg.set_zorder(20)
        plt.xlabel('Task')
        plt.ylim(ylims[var])
        plt.savefig('figs/loss_sa_{}.pdf'.format(var), bbox_inches='tight', transparent=True)

    # =================================
    # Evaluation in Situation SY (task transfer)
    # =================================
    # Get Situation SY
    doms_src = [2, 3]
    doms_trg = [2, 3]
    cmbs = list(itertools.product(doms_src, doms_trg))
    cmbs = remove_ss_cmbs(cmbs)
    res_cgm = pd.read_csv('tmp/qagent_nllik_b.csv').drop_duplicates(subset=['id', 'n_steps_src', 'n_steps_trg'], keep='last', ignore_index=True)
    res_edm = pd.read_csv('tmp/encdec_loss_b.csv').drop_duplicates(subset=['id', 'n_steps_src', 'n_steps_trg'], keep='last', ignore_index=True)
    dfb = pd.DataFrame()
    for dom_src, dom_trg in cmbs:
        _df = res_cgm.query('id in @ids & n_steps_src == @dom_src & n_steps_trg == @dom_trg').sort_values(by='id')
        dfb = store_anova_table_b(dfb, _df, 'cgm')
        _df = res_edm.query('id in @ids & n_steps_src == @dom_src & n_steps_trg == @dom_trg').sort_values(by='id')
        dfb = store_anova_table_b(dfb, _df, 'edm')

    # Evaluate Situation SY CG vs EDM
    for vv, var in enumerate(['nllik', 'match']):
        kwargs = {'dv':var, 'padjust':'bonf', 'parametric':True, 'alternative':'two-sided'}
        print('\nANOVA and multicomparison cmparing CG vs ED in {}'.format(var))
        aov = pg.rm_anova(data=dfb[dfb['org_opp'] == 'org'], dv=var, subject='id', within=['model', 'dom_src'])
        print(aov.round(3))
        post_hocs = pg.pairwise_tests(data=dfb[dfb['org_opp'] == 'org'], within=['dom_src', 'model'], subject='id', **kwargs)
        print(post_hocs.round(3))

        plt.figure(fig_n).clf(); fig_n += 1
        a, b = dfb.query('model=="cgm" & dom_src==3 & org_opp=="org"'), dfb.query('model=="edm" & dom_src==3 & org_opp=="org"')
        plot_paired(a, b, 0, var, ylabel=axlabels[var])
        a, b = dfb.query('model=="cgm" & dom_src==2 & org_opp=="org"'), dfb.query('model=="edm" & dom_src==2 & org_opp=="org"')
        b1, b2 = plot_paired(a, b, 2.5, var, ylabel=axlabels[var])
        plt.xticks([.5, 3.0], [r'3$\rightarrow$2-step', r'2$\rightarrow$3-step'])
        if var == 'nllik':
            lg = plt.legend([b1['boxes'][0], b2['boxes'][0]], ['Cognitive model', 'EIDT'], loc='upper right'); lg.set_zorder(20)
        plt.ylim(ylims[var])
        plt.savefig('figs/loss_org_model_{}.pdf'.format(var), bbox_inches='tight', transparent=True)

    # =================================
    print('\nz_dist X prediction performance')
    # =================================
    cmbres = pd.read_csv('tmp/encdec_loss_b_cmbs.csv')
    pids = cmbres['target_id'].unique()
    for n_steps_src, n_steps_trg in [[3, 2], [2, 3]]:
        for pp, pid in enumerate(pids):
            s0 = cmbres.query('n_steps_src == @n_steps_src & n_steps_trg == @n_steps_trg & target_id == @pid & source_id == @pid').iloc[0]
            z0 = np.array([s0['z1'], s0['z2']])
            for _pp, _pid in enumerate(pids):
                s1 = cmbres.query('n_steps_src == @n_steps_src & n_steps_trg == @n_steps_trg & target_id == @pid & source_id == @_pid')
                idx = s1.index[0]
                s1 = s1.iloc[0]
                z1 = np.array([s1['z1'], s1['z2']])
                d = np.linalg.norm(z0 - z1)
                cmbres.loc[idx, 'z_dist'] = d
    cmbres.to_csv('tmp/encdec_loss_b_cmbs.csv', index=False)

    for n_steps_src, n_steps_trg in [[3, 2], [2, 3]]:
        cmbres1 = cmbres.query('n_steps_src == @n_steps_src & n_steps_trg == @n_steps_trg')
        for vv, dv in enumerate(['nllik', 'match']):
            formula = '{} ~ target_id + z_dist'.format(dv)
            mdl = smf.glm(formula=formula, data=cmbres1, family=sm.families.Gamma(sm.families.links.Log())).fit()
            # print(mdl.summary())
            print('[{}] {:.3f}, pvalue={:.3f}'.format(dv, mdl.params['z_dist'], mdl.pvalues['z_dist']))

            plt.figure(fig_n).clf(); fig_n += 1
            testdf = pd.DataFrame(np.array(['abcd'] * 1000), columns=['target_id'])
            testdf['z_dist'] = np.linspace(0, cmbres1['z_dist'].max(), 1000)
            y = np.zeros([len(pids), 1000])
            for pp, pid in enumerate(pids):
                ms = plt.plot(cmbres1[cmbres1['target_id'] == pid]['z_dist'], cmbres1[cmbres1['target_id'] == pid][dv], '.', alpha=.05, mec=None)
                testdf['target_id'] = pid
                y_ = mdl.predict(testdf)
                # plt.plot(testdf['z_dist'], y_, '--', c=ms[0].get_color(), alpha=.2, zorder=100)
                y[pp] = y_
            plt.plot(testdf['z_dist'], y.mean(0), 'k', zorder=1000)
            plt.ylabel(axlabels[dv])
            plt.xlabel('Distance in individual latent rep.')
            plt.savefig('figs/org_vs_opp/{}to{}_{}.pdf'.format(n_steps_src, n_steps_trg, dv), bbox_inches='tight', transparent=True)

    # =================================
    # Traning and validation curves
    # =================================
    for dom_src, dom_trg in cmbs:
        net_path = 'tmp/mdp_it_s{}_s{}_leave_none.pth'.format(dom_src, dom_trg)
        enc, dec, actnet = sv.init_nets(5, 2, dom_src, dom_trg, device='cpu', verbose=False)
        enc, dec, [train_loss, valid_loss], _ = sv.load_net(enc, dec, net_path, '.valid', device='cpu', verbose=True)
        epoch_stopped = len(valid_loss)
        _, _, [train_loss, valid_loss], _ = sv.load_net(enc, dec, net_path, '.latest', device='cpu', verbose=False)
        plt.figure(fig_n).clf(); fig_n += 1
        plt.semilogy(train_loss, label='Training')
        plt.semilogy(valid_loss, label='Validation')
        ylim = plt.gca().get_ylim()
        plt.vlines([epoch_stopped], ylim[0], ylim[1], ls='dotted', colors='k')
        plt.semilogy(epoch_stopped, valid_loss[epoch_stopped-1], 'k*', ms=20)
        plt.ylim(ylim)
        plt.xlim(0, 50000)
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss (negative log-likelihood)')
        plt.savefig('figs/training/{}_{}.pdf'.format(dom_src, dom_trg), bbox_inches='tight', transparent=True)

    # =================================
    # On-policy evaluation
    # =================================
    at = pd.read_csv('tmp/at_on_policy.csv')
    pids = at['id'].unique()
    agent = 'org'
    for n_steps_src, n_steps_trg in [[3, 2], [2, 3]]:
        for agent in ['org', 'opp']:
            for dv in ['total_rew', 'percent_rewarding_act']:
                plt.figure(fig_n).clf()
                for pp, pid in enumerate(pids):
                    plt.plot(at.query('n_steps_src == @n_steps_src & n_steps_trg == @n_steps_trg & id == @pid & agent == "hum"')[dv],
                             at.query('n_steps_src == @n_steps_src & n_steps_trg == @n_steps_trg & id == @pid & agent == @agent')[dv],
                             'o', alpha=.5)
                plt.xlabel('Human'); plt.ylabel('EIDT')
                if dv == 'percent_rewarding_act':
                    plt.plot([0, 1], [0, 1], 'k--', zorder=-1)
                elif dv == 'total_rew':
                    plt.plot([15, 42], [15, 42], 'k--', zorder=-1)
                plt.axis('equal')
                plt.axis('square')
                plt.savefig('figs/on_policy/{}_{}to{}_{}.pdf'.format(dv, n_steps_src, n_steps_trg, agent), bbox_inches='tight', transparent=True)

    at1 = at.query('agent != "opp" & block in [2]')
    at1['ixb'] = at1['id'] + at1['block'].to_numpy().astype(str)
    for dv in ['total_rew', 'percent_rewarding_act']:
        print('\nANOVA and multicomparison cmparing HUM vs ORG in {} on-policy'.format(dv))
        print('[{}] hum {:.3f} vs org {:.3f}'.format(dv, at1[at1['agent'] == 'hum'][dv].mean(), at1[at1['agent'] == 'org'][dv].mean()))
        aov = pg.rm_anova(data=at1, dv=dv, subject='ixb', within=['agent', 'n_steps_trg'])
        print(aov.round(3))
        kwargs = {'padjust':'bonf', 'parametric':False, 'alternative':'two-sided'}
        post_hocs = pg.pairwise_tests(data=at1, dv=dv, within=['n_steps_trg', 'agent'], subject='ixb', **kwargs)
        print(post_hocs.round(3))

    at1 = at.query('agent != "opp"')
    for n_steps_src in [3, 2]:
        at2 = at1[at1['n_steps_src'] == n_steps_src]
        for dv in ['total_rew', 'percent_rewarding_act']:
            a = at2[at2['agent'] == 'hum'][dv]
            b = at2[at2['agent'] == 'org'][dv]
            tres = scipy.stats.pearsonr(a, b)
            print('[src: {}, {}] r={:.3f}, pvalue={:.3f}'.format(n_steps_src, dv, tres.statistic, tres.pvalue))
    fig_n += 1


    # ===============================
    # QRL parameters for human behaviours
    # ===============================
    qrl_bhv_path = './data/qrl_mdp.csv'
    dfp = pd.read_csv('tmp/param_qagents.csv')
    for param_n in ['alpha', 'beta', 'gamma', 'init_q']:
        plt.figure(fig_n).clf(); fig_n += 1
        plt.figure
        plt.hist(dfp[param_n], color='gray', bins=25)
        plt.xlabel(axlabels[param_n])
        plt.ylabel('Count')
        plt.savefig('figs/qrl/human_{}.pdf'.format(param_n), bbox_inches='tight', transparent=True)


    # =================================
    print('\nQL parameters and individual latent representation')
    # =================================
    dfz = pd.read_csv('tmp/z_qrl.csv')
    for n_steps_src in [3, 2]:
        dfz1 = dfz[dfz['n_steps'] == n_steps_src]
        for zz in ['z1', 'z2']:
            formula = '{} ~ np.log(alpha) * np.log(beta)'.format(zz)
            mdl = smf.glm(formula=formula, data=dfz1, family=sm.families.Gaussian(sm.families.links.InversePower())).fit()
            print(mdl.summary())
            y_ = mdl.predict(dfz1)

            for param_n in ['alpha', 'beta']:
                plt.figure(fig_n).clf(); fig_n += 1
                plt.plot(dfz1[param_n], dfz1[zz], 'k.', alpha=.5, mec=None)
                plt.ylabel(axlabels[zz])
                plt.xlabel(axlabels[param_n])

                plt.plot(dfz1[param_n], y_, '.', alpha=.5, mec=None)
                if (zz == 'z1') and (param_n == 'alpha'):
                    plt.legend(['EIDT projection', 'GLM fitting'])
                plt.savefig('figs/qrl/ns{}_{}_{}.pdf'.format(n_steps_src, param_n, zz), bbox_inches='tight', transparent=True)


    # =================================
    print('\nz_dist X prediction performance in QRL')
    # =================================
    cmbres = pd.read_csv('tmp/encdec_loss_b_qrl_cmb.csv')
    pids = cmbres['target_id'].unique()
    df = pd.DataFrame()
    for n_steps_src in [3, 2]:
        cmbres1 = cmbres[cmbres['n_steps_src'] == n_steps_src]
        for pp1, target_id in tqdm(enumerate(pids), total=len(pids)):
            cmbres2 = cmbres1[cmbres1['target_id'] == target_id]
            tsr = cmbres2[cmbres2['source_id'] == target_id].iloc[0]
            z0 = np.array([tsr['z1'], tsr['z2']])
            for pp2, source_id in enumerate(cmbres2['source_id'].unique()):
                ssr = cmbres2[cmbres2['source_id'] == source_id].iloc[0]
                z1 = np.array([ssr['z1'], ssr['z2']])
                z_dist = np.linalg.norm(z0 - z1)
                _df = pd.DataFrame({'n_steps_src':[n_steps_src], 'target_id':[target_id], 'z_dist':[z_dist], 'nllik':[ssr['nllik']], 'match':[ssr['match']]})
                df = pd.concat([df, _df], axis=0)
        df.to_csv('tmp/qrl_z_dist_pp.csv', index=False)

    df = pd.read_csv('tmp/qrl_z_dist_pp.csv')
    df = df.sample(20000)
    for n_steps_src in [3, 2]:
        df1 = df.query('n_steps_src == @n_steps_src')
        pids = df1['target_id'].unique()
        for vv, dv in enumerate(['nllik', 'match']):
            formula = '{} ~ target_id + z_dist'.format(dv)
            mdl = smf.glm(formula=formula, data=df1, family=sm.families.Gamma(sm.families.links.Log())).fit()
            # print(mdl.summary())
            print('[{}] {:.3f}, pvalue={:.3f}'.format(dv, mdl.params['z_dist'], mdl.pvalues['z_dist']))

            plt.figure(fig_n).clf(); fig_n += 1
            testdf = pd.DataFrame(np.array(['abcd'] * 1000), columns=['target_id'])
            testdf['z_dist'] = np.linspace(0, df1['z_dist'].max(), 1000)
            y = np.zeros([len(pids), 1000])
            for pp, pid in enumerate(pids):
                ms = plt.plot(df1[df1['target_id'] == pid]['z_dist'], df1[df1['target_id'] == pid][dv], '.', alpha=.05, mec=None)
                testdf['target_id'] = pid
                y_ = mdl.predict(testdf)
                # plt.plot(testdf['z_dist'], y_, '--', c=ms[0].get_color(), alpha=.2, zorder=100)
                y[pp] = y_
            plt.plot(testdf['z_dist'], y.mean(0), 'k', zorder=1000)
            plt.ylabel(axlabels[dv])
            plt.xlabel('Distance in individual latent rep.')
            plt.savefig('figs/qrl/z_dist_{}_{}.pdf'.format(dv, n_steps_src), bbox_inches='tight', transparent=True)


    # =================================
    print('\nQL parameters X prediction performance')
    # =================================
    cmbres = pd.read_csv('tmp/encdec_loss_b_qrl_cmb.csv')
    # cmbres = cmbres.sample(10000)
    qp = pd.read_csv('tmp/z_qrl.csv')
    pids = cmbres['target_id'].unique()
    df = pd.DataFrame()
    for n_steps_src in [3, 2]:
        cmbres1 = cmbres[cmbres['n_steps_src'] == n_steps_src]
        qp1 = qp.query('n_steps == @n_steps_src')
        for pp1, target_id in tqdm(enumerate(pids), total=len(pids)):
            cmbres2 = cmbres1[cmbres1['target_id'] == target_id]
            alpha0 = qp1[qp1['id'] == target_id].iloc[0]['alpha']
            beta0 = qp1[qp1['id'] == target_id].iloc[0]['beta']
            for pp2, source_id in enumerate(cmbres2['source_id'].unique()):
                ssr = cmbres2[cmbres2['source_id'] == source_id].iloc[0]
                alpha1 = qp1[qp1['id'] == source_id].iloc[0]['alpha']
                beta1 = qp1[qp1['id'] == source_id].iloc[0]['beta']
                _df = pd.DataFrame({'n_steps_src':[n_steps_src], 'target_id':[target_id], 'd_alpha':[np.abs(alpha0 - alpha1)], 'd_beta':[np.abs(beta0 - beta1)], 'nllik':[ssr['nllik']], 'match':[ssr['match']]})
                df = pd.concat([df, _df], axis=0)
        df.to_csv('tmp/qrl_qlp_pp.csv', index=False)

    df = pd.read_csv('tmp/qrl_qlp_pp.csv')
    for n_steps_src in [3, 2]:
        df1 = df.query('n_steps_src == @n_steps_src')
        pids = df1['target_id'].unique()
        for vv, dv in enumerate(['nllik', 'match']):
            formula = '{} ~ d_alpha * d_beta'.format(dv)
            mdl = smf.glm(formula=formula, data=df1, family=sm.families.Gamma(sm.families.links.Log())).fit()
            print('\n[source: {}, dv:{}]'.format(n_steps_src, dv))
            print(mdl.summary())
            _df1 = df1.sample(2500)
            # _y = mdl.predict(_df1)
            for uu, du in enumerate(['d_alpha', 'd_beta']):
                # print('[{}, {}] {:.3f}, pvalue={:.3f}'.format(dv, du, mdl.params[du], mdl.pvalues[du]))

                n_d = 1000
                testdf = pd.DataFrame(np.array(['abcd'] * n_d), columns=['fake'])
                if du == 'd_alpha':
                    testdf['d_alpha'] = np.linspace(0, df1['d_alpha'].max(), n_d)
                    testdf['d_beta'] = _df1['d_beta'].mean()
                elif du == 'd_beta':
                    testdf['d_beta'] = np.linspace(0, df1['d_beta'].max(), n_d)
                    testdf['d_alpha'] = _df1['d_alpha'].mean()
                _y = mdl.predict(testdf)

                plt.figure(fig_n).clf(); fig_n += 1
                m1 = plt.plot(_df1[du], _df1[dv], 'k.', alpha=.1, mec=None)
                m2 = plt.plot(testdf[du], _y, 'k', alpha=1., zorder=1000)
                xlim = plt.gca().get_xlim()
                ylim = plt.gca().get_ylim()
                ms1 = plt.plot(-1, 0, 'o', mfc=m1[0].get_mfc(), mec='w')
                ms2 = plt.plot(-1, 0, 'o', mfc=m2[0].get_mfc(), mec='w')
                plt.xlim(xlim)
                plt.ylim(ylim)
                plt.ylabel(axlabels[dv])
                plt.xlabel(axlabels[du])
                # if du == 'd_alpha' and dv == 'nllik':
                    # plt.legend([ms1[0], m2[0]], ['Predicted', 'Fitting'])
                # if dv == 'nllik':
                    # plt.ylim(.25, 1.1)
                # elif dv == 'match':
                    # plt.ylim(.4, .9)
                plt.savefig('figs/qrl/{}_{}_{}.pdf'.format(du, dv, n_steps_src), bbox_inches='tight', transparent=True)
                plt.pause(.1)


    # =================================
    # Plot z-plane
    # =================================
    zq = pd.read_csv('tmp/z_qrl.csv')
    zh = pd.read_csv('tmp/z_hum.csv')
    for n_steps_src in [3, 2]:
        zq1 = zq[zq['n_steps'] == n_steps_src]
        zh1 = zh[zh['n_steps'] == n_steps_src]

        plt.figure(fig_n).clf(); fig_n += 1
        plt.plot(zq1['z1'], zq1['z2'], 'k.', alpha=.1)
        for pid in zh1['id'].unique():
            zh2 = zh1[zh1['id'] == pid]
            l = plt.plot(zh2['z1'].mean(), zh2['z2'].mean(), 's')
        plt.xlabel(axlabels['z1'])
        plt.ylabel(axlabels['z2'])

        plt.savefig('figs/z/mdp{}.pdf'.format(n_steps_src), bbox_inches='tight', transparent=True)
