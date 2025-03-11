import sys
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
    dft = pd.concat([dft, dft_], axis=0)
    dft_['org_opp'] = 'opp'
    dft_['nllik'] = _df['nllik_opp']
    dft = pd.concat([dft, dft_], axis=0)
    return dft

def store_anova_table_b(dft, _df, model):
    dft_ = pd.DataFrame()
    dft_['id'] = _df['id']
    dft_['dom_src'] = dom_src
    dft_['dom_trg'] = dom_trg
    dft_['model'] = model
    dft_['org_opp'] = 'org'
    dft_['nllik'] = _df['nllik_org']
    dft = pd.concat([dft, dft_], axis=0)
    dft_['org_opp'] = 'opp'
    dft_['nllik'] = _df['nllik_opp']
    dft = pd.concat([dft, dft_], axis=0)
    return dft

def plot_paired(a, b, x, ticks=[], xlabel='Task transfer set', ylabel='Negative log-likelihood'):
    # b1 = plt.bar(x+0, a['nllik'].mean(), yerr=a['nllik'].std(), ec='k', fc='w')
    # b2 = plt.bar(x+1, b['nllik'].mean(), yerr=b['nllik'].std(), ec='k', fc='gray')
    boxplot_params = {'widths':[.8], 'whis':[0, 100], 'patch_artist':True, 'showfliers':False}
    b1 = plt.boxplot(a['nllik'], positions=[x], **boxplot_params)
    b2 = plt.boxplot(b['nllik'], positions=[x+1], **boxplot_params)
    b1['boxes'][0].set_facecolor('white')
    b2['boxes'][0].set_facecolor('gray')
    for id_ in a['id']:
        a_ = a[a['id'] == id_].iloc[0]['nllik']
        b_ = b[b['id'] == id_].iloc[0]['nllik']
        # plt.plot([x+.2, x+.8], [a_, b_], 'o-k', alpha=.2)
        plt.plot([x+.2, x+.8], [a_, b_], '-k', lw=.2, zorder=10)
        plt.plot([x+.2, x+.8], [a_, b_], 'ok', mfc='w', zorder=11)
    if len(ticks) == 2:
        plt.xticks([x+0, x+1], ticks)
    elif len(ticks) == 1:
        plt.xticks([x + .5], ticks)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    return b1, b2


if __name__=='__main__':
    sets = ['test']

    print('Set: {}'.format(sets))
    iddf = pd.read_csv('tmp/split_{}_train_valid_test.csv'.format('bhv'))
    ids = iddf.query('set in @sets')['id']

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

    # Evaluate Situation SA
    ylim = [.2, .85]
    kwargs = {'dv':'nllik', 'padjust':'bonf', 'parametric':True, 'alternative':'less'}
    print('\nANOVA and multicomparison cmparing CG vs TS')
    aov = pg.rm_anova(data=dfa[dfa['org_opp'] == 'org'], dv='nllik', subject='id', within=['model', 'dom'])
    print(aov.round(3))
    post_hocs = pg.pairwise_tests(data=dfa[dfa['org_opp'] == 'org'], within=['dom', 'model'], subject='id', **kwargs)
    print(post_hocs.round(3))

    plt.figure(0).clf()
    a, b = dfa.query('model=="cgm" & dom==2 & org_opp=="org"'), dfa.query('model=="act" & dom==2 & org_opp=="org"')
    plot_paired(a, b, 0)
    a, b = dfa.query('model=="cgm" & dom==3 & org_opp=="org"'), dfa.query('model=="act" & dom==3 & org_opp=="org"')
    b1, b2 = plot_paired(a, b, 2.5)
    plt.xticks([.5, 3.0], ['2-step', '3-step'])
    # plt.legend([b1[0], b2[0]], ['Cognitive model', 'Task solver'], loc='lower right')
    lg = plt.legend([b1['boxes'][0], b2['boxes'][0]], ['Cognitive model', 'Task solver'], loc='lower left'); lg.set_zorder(20)
    plt.xlabel('Task')
    plt.savefig('figs/loss_sa.pdf', bbox_inches='tight', transparent=True)

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

    # Evaluate Situation SB
    kwargs = {'dv':'nllik', 'padjust':'bonf', 'parametric':True, 'alternative':'greater'}
    print('\nANOVA and multicomparison cmparing CG vs ED')
    aov = pg.rm_anova(data=dfb[dfb['org_opp'] == 'org'], dv='nllik', subject='id', within=['model', 'dom_src'])
    print(aov.round(3))
    post_hocs = pg.pairwise_tests(data=dfb[dfb['org_opp'] == 'org'], within=['dom_src', 'model'], subject='id', **kwargs)
    print(post_hocs.round(3))

    plt.figure(1).clf()
    a, b = dfb.query('model=="cgm" & dom_src==3 & org_opp=="org"'), dfb.query('model=="edm" & dom_src==3 & org_opp=="org"')
    plot_paired(a, b, 0)
    a, b = dfb.query('model=="cgm" & dom_src==2 & org_opp=="org"'), dfb.query('model=="edm" & dom_src==2 & org_opp=="org"')
    b1, b2 = plot_paired(a, b, 2.5)
    plt.xticks([.5, 3.0], [r'3$\rightarrow$2-step', r'2$\rightarrow$3-step'])
    # plt.legend([b1[0], b2[0]], ['Cognitive model', 'Encoder-decoder'], loc='lower right')
    lg = plt.legend([b1['boxes'][0], b2['boxes'][0]], ['Cognitive model', 'Encoder-decoder'], loc='lower left'); lg.set_zorder(20)
    plt.savefig('figs/loss_org_model.pdf', bbox_inches='tight', transparent=True)


    kwargs = {'dv':'nllik', 'padjust':'bonf', 'parametric':True, 'alternative':'greater'}
    for mm, model in enumerate(['cgm', 'edm']):
        print('\n[{}] ANOVA and multicomparison for ORG vs OPP'.format(model))
        aov = pg.rm_anova(data=dfb[dfb['model'] == model], dv='nllik', subject='id', within=['org_opp', 'dom_src'])
        print(aov.round(3))
        post_hocs = pg.pairwise_tests(data=dfb[dfb['model'] == model], within=['dom_src', 'org_opp'], subject='id', **kwargs)
        print(post_hocs.round(3))

        plt.figure(2+mm).clf()
        a, b = dfb.query('model==@model & dom_src==3 & org_opp=="org"'), dfb.query('model==@model & dom_src==3 & org_opp=="opp"')
        plot_paired(a, b, 0)
        a, b = dfb.query('model==@model & dom_src==2 & org_opp=="org"'), dfb.query('model==@model & dom_src==2 & org_opp=="opp"')
        b1, b2 = plot_paired(a, b, 2.5)
        plt.xticks([.5, 3.0], [r'3$\rightarrow$2-step', r'2$\rightarrow$3-step'])
        if mm == 1:
            # plt.legend([b1[0], b2[0]], ['Original', 'Others'], loc='lower right')
            lg = plt.legend([b1['boxes'][0], b2['boxes'][0]], ['Original', 'Others'], loc='lower left'); lg.set_zorder(20)
        plt.savefig('figs/loss_opp_{}.pdf'.format(model), bbox_inches='tight', transparent=True)
