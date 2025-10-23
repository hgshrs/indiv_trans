import sys
import importlib
import itertools
import pandas as pd
import pingouin as pg
import numpy as np
import matplotlib.pylab as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"

import warnings
warnings.simplefilter('ignore', FutureWarning)
print('FutureWaning is diabled')

import train_it_mnist as ted
importlib.reload(ted)
import analyze_bhv_mnist as ab
importlib.reload(ab)

def screening_res_b(df, set_abbr_tasks, participants, verbose=False):
    df = df.query('subject in @participants')
    new_df = pd.DataFrame()
    cmbns = []
    for tasks in set_abbr_tasks:
        difficulty_src, sat_src, difficulty_trg, sat_trg = ted.translate_tasks(tasks)
        _df = df.query('difficulty_src==@difficulty_src & sat_src==@sat_src & difficulty_trg==@difficulty_trg & sat_trg==@sat_trg')
        new_df = pd.concat([new_df, _df], axis=0)
        if len(_df) > 0:
            cmbns.append([difficulty_src, sat_src, difficulty_trg, sat_trg])
            if verbose:
                print('There is {} results for\tsrc: {}, {}\ttrg: {}, {}'.format(len(_df), difficulty_src, sat_src, difficulty_trg, sat_trg))
        else:
            if verbose:
                print('There is no result for\tsrc: {}, {}\ttrg: {}, {}'.format(difficulty_src, sat_src, difficulty_trg, sat_trg))
    return new_df, cmbns

def mk_anova_frame_b(df):
    df1 = df.copy()
    df1['trans_to'] = 'org'
    df1 = df1.drop(['loss_opp', 'match_opp', 'correct_opp'], axis=1)
    df2 = df.copy()
    df2['trans_to'] = 'opp'
    df2 = df2.drop(['loss', 'match', 'correct'], axis=1)
    df2 = df2.rename(columns={'loss_opp':'loss', 'match_opp':'match', 'correct_opp':'correct'})
    return pd.concat([df1, df2], axis=0)

def redefine_task_b(df, cmbns):
    # df['state'] = 'aaaa'
    for cmbn in cmbns:
        df.loc[np.logical_and(
            np.logical_and(df['difficulty_src'] == cmbn[0], df['sat_src'] == cmbn[1]),
            np.logical_and(df['difficulty_trg'] == cmbn[2], df['sat_trg'] == cmbn[3])
            ), 'task'] = cmbn[0][0] + cmbn[1][0] + cmbn[2][0] + cmbn[3][0]
    df = df.drop(['difficulty_src', 'sat_src', 'difficulty_trg', 'sat_trg'], axis=1)
    return df

def plot_paired(a, b, x, dv, ticks=[], xlabel='Transfer direction', ylabel='Negative log-likelihood'):
    # b1 = plt.bar(x+0, a[dv].mean(), yerr=a[dv].std(), ec='k', fc='w')
    # b2 = plt.bar(x+1, b[dv].mean(), yerr=b[dv].std(), ec='k', fc='gray')
    boxplot_params = {'widths':[.8], 'whis':[0, 100], 'patch_artist':True, 'showfliers':False}
    b1 = plt.boxplot(a[dv], positions=[x], **boxplot_params)
    b2 = plt.boxplot(b[dv], positions=[x+1], **boxplot_params)
    b1['boxes'][0].set_facecolor('white')
    b2['boxes'][0].set_facecolor('gray')
    for pid in a['id']:
        a_ = a[a['id'] == pid].iloc[0][dv]
        b_ = b[b['id'] == pid].iloc[0][dv]
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

def text_trans_set(task, new_line=False):
    if new_line:
        return '{}\n'.format(task[:2].upper()) + r'$\downarrow$' + '\n{}'.format(task[-2:].upper())
    else:
        return r'{}$\rightarrow${}'.format(task[:2].upper(), task[-2:].upper())

def screening_res_a(df, tasks, participants):
    df = df.query('subject in @participants')
    new_df = pd.DataFrame()
    new_tasks = []
    for difficulty, sat in tasks:
        _df = df.query('difficulty==@difficulty & sat==@sat')
        new_df = pd.concat([new_df, _df])
        if len(_df) > 0:
            new_tasks.append((difficulty, sat))
    return new_df, new_tasks

def mk_anova_frame_a(res_rn, res_ts):
    res_rn['model'] = 'rn'
    res_ts['model'] = 'ts'
    return pd.concat([res_rn, res_ts], axis=0)

def redefine_task_a(df, tasks):
    # df['state'] = 'aaaa'
    for task in tasks:
        df.loc[np.logical_and(df['difficulty'] == task[0], df['sat'] == task[1]), 'task'] = task[0][0] + task[1][0]
    df = df.drop(['difficulty', 'sat'], axis=1)
    return df

if __name__ == '__main__':
    iddf = pd.read_csv('tmp/split_participants_train_valid_test.csv')
    participants = iddf.query('set == "test"')['id']
    plt.close('all')
    fig_n = 0
    set_abbr_tasks = [
            'esea', 'daea', 'dsea',
            'eaes', 'daes', 'dses',
            'eada', 'esda', 'dsda',
            'eads', 'esds', 'dads',
            ]
    axlabels = {'nllik': 'Negative log-likelihood', 'match': 'Rate for behaviour matched', 'correct': 'Rate for correct responses'}
    cmap = plt.colormaps.get_cmap('tab10')


    # =================================
    print('\nSituation SX (no task transfer)')
    # =================================
    tasks = ['ea', 'es', 'da', 'ds']
    res_rn = pd.read_csv('tmp/pred_rtn.csv')
    res_ts = pd.read_csv('tmp/mnist_ts.csv')
    res_rn['model'] = 'rn'
    res_ts['model'] = 'ts'
    pids = res_rn['id'].unique()

    res_hm = pd.DataFrame()
    bhv = ab.load_bhv(path='rtnet/behavioral data.csv', reject=True)
    for abbr_task in tasks:
        difficulty, sat = ted.translate_tasks(abbr_task)
        bhv1 = bhv.query('difficulty == @difficulty & sat == @sat')

        rp = np.zeros([len(pids)])
        for pp, pid in enumerate(pids):
            rp[pp] = bhv1[bhv1['subject'] == pid]['correct'].mean()
        res = pd.DataFrame({'task': [abbr_task] * len(pids), 'id': pids, 'correct': rp})
        res_hm = pd.concat([res_hm, res], axis=0)
    res_hm['model'] = 'hm'
    df = pd.concat([res_hm, res_rn, res_ts], axis=0)

    dv = 'correct'
    aov = pg.rm_anova(data=df, dv=dv, subject='id', within=['model', 'task'])
    print(aov.round(3))
    plt.figure(fig_n); plt.clf(); fig_n = fig_n + 1
    x = .0
    xticks, xticklabels = [], []
    for task in tasks:
        a, b, c = df.query('task==@task & model=="hm"'), df.query('task==@task & model=="rn"'), df.query('task==@task & model=="ts"')
        boxplot_params = {'widths':[.8], 'whis':[0, 100], 'patch_artist':True, 'showfliers':False}
        b1 = plt.boxplot(a[dv], positions=[x], **boxplot_params)
        b2 = plt.boxplot(b[dv], positions=[x+1], **boxplot_params)
        b3 = plt.boxplot(c[dv], positions=[x+2], **boxplot_params)
        b1['boxes'][0].set_facecolor(cmap(0))
        b2['boxes'][0].set_facecolor('white')
        b3['boxes'][0].set_facecolor('gray')
        for pid in a['id']:
            a_ = a[a['id'] == pid].iloc[0][dv]
            b_ = b[b['id'] == pid].iloc[0][dv]
            c_ = c[c['id'] == pid].iloc[0][dv]
            # plt.plot([x+.2, x+1.2, x+2.2], [a_, b_, c_], '-k', lw=.2, zorder=10)
            # plt.plot([x+.2, x+1.2, x+2.2], [a_, b_, c_], 'ok', mfc='w', zorder=11)
        xticks.append(x + 1)
        xticklabels.append(task.upper())
        # plt.ylim(ylim)
        x += 3.5
    lg = plt.legend([b1['boxes'][0], b2['boxes'][0], b3['boxes'][0]], ['Human', 'RTNet', 'Task solver']); lg.set_zorder(20)
    plt.xlabel('Task')
    plt.ylabel(axlabels[dv])
    plt.xticks(xticks, xticklabels)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.savefig('figs/mnist/human.pdf'.format(dv), bbox_inches='tight', transparent=True)

    df = pd.concat([res_rn, res_ts], axis=0)
    for dv in ['nllik', 'match']:
        print('[{}]'.format(axlabels[dv]))
        print('RN: {:3f} vs TS: {:.3f}'.format(df[df['model'] == 'rn'][dv].mean(), df[df['model'] == 'ts'][dv].mean()))
        aov = pg.rm_anova(data=df, dv=dv, subject='id', within=['model', 'task'])
        print(aov.round(3))
        # post_hocs = pg.pairwise_tests(data=df, dv=dv, subject='id', within=['task', 'model'], padjust='bonf')
        # print(post_hocs.round(3))

        plt.figure(fig_n); plt.clf(); fig_n = fig_n + 1
        xticks, xticklabels = [], []
        x = .0
        for task in tasks:
            a, b = df.query('task==@task & model=="rn"'), df.query('task==@task & model=="ts"')
            b1, b2 = plot_paired(a, b, x, dv, xlabel='Task', ylabel=axlabels[dv])
            xticks.append(x + .5)
            xticklabels.append(task.upper())
            x += 2.5
        if dv == 'nllik':
            lg = plt.legend([b1['boxes'][0], b2['boxes'][0]], ['RTNet', 'Task solver']); lg.set_zorder(20)
        plt.xticks(xticks, xticklabels) plt.savefig('figs/mnist/{}_sx.pdf'.format(dv), bbox_inches='tight', transparent=True)


    # =================================
    print('\nPlot Situation SY with task solvers')
    # =================================
    # make a dataframe
    edres = pd.read_csv('tmp/mnist_opp2.csv')
    pids = np.sort(edres['target_id'].unique())
    df = pd.DataFrame()
    for abbr_tasks in set_abbr_tasks:
        edres1 = edres[edres['tasks'] == abbr_tasks]
        for pp, pid in enumerate(pids):
            nllik = edres1.query('target_id == @pid & source_id == @pid').iloc[0]['nllik']
            match = edres1.query('target_id == @pid & source_id == @pid').iloc[0]['match']
            correct = edres1.query('target_id == @pid & source_id == @pid').iloc[0]['correct']
            ddict = {'tasks':abbr_tasks, 'id':pid, 'model':'ed', 'nllik':nllik, 'match':match, 'correct':correct}
            df = pd.concat([df, pd.Series(ddict).to_frame().T], axis=0)

    tsres = pd.read_csv('tmp/mnist_ts2.csv')
    for abbr_tasks in set_abbr_tasks:
        for pp, pid in enumerate(pids):
            tsres1 = tsres[tsres['tasks'] == abbr_tasks]
            nllik = tsres1.query('id == @pid & train_type == "given"').iloc[0]['nllik']
            match = tsres1.query('id == @pid & train_type == "given"').iloc[0]['match']
            correct = tsres1.query('id == @pid & train_type == "given"').iloc[0]['correct']
            ddict = {'tasks':abbr_tasks, 'id':pid, 'model':'ts1', 'nllik':nllik, 'match':match, 'correct':correct}
            df = pd.concat([df, pd.Series(ddict).to_frame().T], axis=0)

            tsres2 = tsres[tsres['tasks'] == abbr_tasks[-2:] + abbr_tasks[-2:]]
            nllik = tsres2.query('id == @pid & train_type == "given"').iloc[0]['nllik']
            match = tsres2.query('id == @pid & train_type == "given"').iloc[0]['match']
            correct = tsres2.query('id == @pid & train_type == "given"').iloc[0]['correct']
            ddict = {'tasks':abbr_tasks, 'id':pid, 'model':'ts2', 'nllik':nllik, 'match':match, 'correct':correct}
            df = pd.concat([df, pd.Series(ddict).to_frame().T], axis=0)

            nllik = tsres1.query('id == @pid & train_type == "leave"').iloc[0]['nllik']
            match = tsres1.query('id == @pid & train_type == "leave"').iloc[0]['match']
            correct = tsres1.query('id == @pid & train_type == "leave"').iloc[0]['correct']
            ddict = {'tasks':abbr_tasks, 'id':pid, 'model':'ts3', 'nllik':nllik, 'match':match, 'correct':correct}
            df = pd.concat([df, pd.Series(ddict).to_frame().T], axis=0)
    df.to_csv('tmp/mnist_sy_ts.csv', index=False)

    df = pd.read_csv('tmp/mnist_sy_ts.csv')
    for dv in ['nllik', 'match']:
        print('[{}]'.format(axlabels[dv]))
        print('TS: {:.3f} vs EIDT: {:.3f}'.format(df[df['model'] == 'ts1'][dv].mean(), df[df['model'] == 'ed'][dv].mean()))
        aov = pg.rm_anova(data=df, dv=dv, subject='id', within=['model', 'tasks'])
        print(aov.round(3))
        # post_hocs = pg.pairwise_tests(data=df, dv=dv, subject='id', within=['task', 'model'], padjust='bonf')
        # print(post_hocs.round(3))

        plt.figure(fig_n, figsize=[6.4*2, 4.8]); plt.clf(); fig_n = fig_n + 1
        xticks, xticklabels = [], []
        x = .0
        for abbr_tasks in set_abbr_tasks:
            a, b = df.query('tasks==@abbr_tasks & model=="ts1"'), df.query('tasks==@abbr_tasks & model=="ed"')
            b1, b2 = plot_paired(a, b, x, dv, xlabel='Transfer direction', ylabel=axlabels[dv])
            xticks.append(x + .5)
            xticklabels.append(text_trans_set(abbr_tasks, new_line=False))
            x += 2.5
        if dv == 'nllik':
            lg = plt.legend([b1['boxes'][0], b2['boxes'][0]], ['Task solver (source)', 'EIDT']); lg.set_zorder(20)
        plt.xticks(xticks, xticklabels, fontsize=15)
        plt.savefig('figs/mnist/{}_sy.pdf'.format(dv), bbox_inches='tight', transparent=True)
    plt.pause(.1)


    # =================================
    print('\nPlot individual latent representation')
    # =================================
    resdf = pd.read_csv('tmp/mnist_opp2.csv')
    pids = np.sort(resdf['target_id'].unique())
    resdf1 = resdf[resdf['target_id'] == 1]
    plt.figure(fig_n).clf()
    for abbr_tasks in set_abbr_tasks:
        resdf2 = resdf1[resdf1['tasks'] == abbr_tasks]
        plt.clf()
        for pid in pids:
            sr = resdf2[resdf2['source_id'] == pid].iloc[0]
            plt.plot(sr['z1'], sr['z2'], 's')
            # plt.text(sr['z1'], sr['z2'], pid, fontsize='small')
        plt.xlabel(r'$z_1$')
        plt.ylabel(r'$z_2$')
        plt.savefig('figs/mnist/z/{}.pdf'.format(abbr_tasks), bbox_inches='tight', transparent=True)
        plt.pause(.1)
    fig_n += 1


    # =================================
    print('\nPlot prediction performance and individual latent representation')
    # =================================
    resdf = pd.read_csv('tmp/mnist_opp2.csv')
    pids = np.sort(resdf['target_id'].unique())
    for dd, dv in enumerate(['nllik', 'match']):
        print(dv)
        for abbr_tasks in set_abbr_tasks:

            formula = '{} ~ C(target_id) + z_dist'.format(dv)
            resdf1 = resdf[resdf['tasks'] == abbr_tasks]
            mdl = smf.glm(formula=formula, data=resdf1, family=sm.families.Gamma(sm.families.links.Log())).fit()
            # print(mdl.summary())
            print('[{}] coefficients for z_dist={:.3f}, p={:.3f}'.format(abbr_tasks, mdl.params['z_dist'], mdl.pvalues['z_dist']))

            testdf = pd.DataFrame(np.array(['abcd'] * 1000), columns=['target_id'])
            testdf['z_dist'] = np.linspace(0, resdf1['z_dist'].max(), 1000)
            y = np.zeros([len(pids), 1000])

            plt.figure(fig_n + dd).clf()
            for pp, pid in enumerate(pids):
                dat = resdf1.query('target_id == @pid')
                ms = plt.plot(dat['z_dist'], dat[dv], '.', alpha=.05, mec=None)
                testdf['target_id'] = pid
                y_ = mdl.predict(testdf)
                y[pp] = y_
            plt.plot(testdf['z_dist'], y.mean(0), 'k', zorder=1000)
            plt.ylabel(axlabels[dv])
            plt.xlabel('Distance in individual latent rep.')
            plt.savefig('figs/mnist/ii2/{}_{}.pdf'.format(abbr_tasks, dv), bbox_inches='tight', transparent=True)
            plt.pause(.1)
    fig_n += 3


    # =================================
    print('\nConcrete actions')
    # =================================
    df = pd.read_csv('tmp/mnist/pred_real_act.csv')
    pids = df['id'].unique()
    pids = np.array([19, 23, 56, 63])
    set_abbr_tasks = ['eaes']
    digits = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int)
    for abbr_tasks in set_abbr_tasks:
        df1 = df[df['tasks'] == abbr_tasks]
        for pp, pid in enumerate(pids):
            df2 = df1[df1['id'] == pid]
            prob_hum = np.zeros(len(digits))
            prob_edm = np.zeros(len(digits))
            for dd, digit in enumerate(digits):
                true = df2[df2['true'] == digit]['true']
                real = df2[df2['true'] == digit]['real']
                pred = df2[df2['true'] == digit]['pred']
                prob_hum[dd] = (true == real).sum() / true.shape[0]
                prob_edm[dd] = (true == pred).sum() / true.shape[0]

            plt.figure(fig_n + dd).clf()
            plt.subplot(211)
            plt.bar(-.2 + digits, prob_hum * 100, width=.4, facecolor=cmap(0))
            plt.bar(+.2 + digits, prob_edm * 100, width=.4, facecolor='gray')
            plt.xticks(digits)
            plt.ylim([0, 100])
            if pid == pids[-1]:
                plt.legend(['Human', 'EIDT'], loc='lower right')
            plt.xlabel('Stimulus digit')
            plt.ylabel('Percent of correct')
            plt.title('Participant \\#{}'.format(pid))
            plt.savefig('figs/mnist/act/p{:02d}_{}.pdf'.format(pid, abbr_tasks), bbox_inches='tight', transparent=True)
    fig_n += 1


    # =================================
    print('Traning and validation curves')
    # =================================
    for aa, abbr_tasks in enumerate(set_abbr_tasks):
        net_path_init = 'tmp/edm_{}_mnistnone.pt'.format(abbr_tasks)
        enc, dec, evn = ted.init_nets(verbose=False)
        enc, dec, train_loss, valid_loss, train_match, valid_match = ted.load_net(enc, dec, net_path_init, state='.valid')
        epoch_stopped = len(valid_loss)
        _, _, train_loss, valid_loss, train_match, valid_match = ted.load_net(enc, dec, net_path_init, state='.latest')

        plt.figure(fig_n); plt.clf(); fig_n = fig_n + 1
        plt.semilogy(train_loss, label='Training')
        plt.semilogy(valid_loss, label='Validation')
        plt.ylim(.9, 3)
        ylim = plt.gca().get_ylim()
        plt.vlines([epoch_stopped], ylim[0], ylim[1], ls='dotted', colors='k')
        plt.semilogy(epoch_stopped, valid_loss[epoch_stopped-1], 'k*', ms=20)
        plt.ylim(ylim)
        # plt.xlim(0, 50000)
        if aa == 0:
            plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss (negative log-likelihood)')
        plt.savefig('figs/mnist/training/{}.pdf'.format(abbr_tasks), bbox_inches='tight', transparent=True)
        plt.pause(.1)
