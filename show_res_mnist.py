import sys
import importlib
import itertools
import pandas as pd
import pingouin as pg
import numpy as np
import matplotlib.pylab as plt
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"

import warnings
warnings.simplefilter('ignore', FutureWarning)
print('FutureWaning is diabled')

import train_it_mnist as ted
importlib.reload(ted)

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

def plot_paired(a, b, x, measure, ticks=[], xlabel='Task transfer set', ylabel='Negative log-likelihood'):
    # b1 = plt.bar(x+0, a[measure].mean(), yerr=a[measure].std(), ec='k', fc='w')
    # b2 = plt.bar(x+1, b[measure].mean(), yerr=b[measure].std(), ec='k', fc='gray')
    boxplot_params = {'widths':[.8], 'whis':[0, 100], 'patch_artist':True, 'showfliers':False}
    b1 = plt.boxplot(a[measure], positions=[x], **boxplot_params)
    b2 = plt.boxplot(b[measure], positions=[x+1], **boxplot_params)
    b1['boxes'][0].set_facecolor('white')
    b2['boxes'][0].set_facecolor('gray')
    for subject_ in a['subject']:
        a_ = a[a['subject'] == subject_].iloc[0][measure]
        b_ = b[b['subject'] == subject_].iloc[0][measure]
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

    print('=====================')
    print('Evaluate Situation SA')
    print('=====================')
    tasks = [
            ('easy', 'accuracy focus'),
            ('easy', 'speed focus'),
            ('difficult', 'accuracy focus'),
            ('difficult', 'speed focus'),
            ]
    res_rn = pd.read_csv('tmp/pred_rtn.csv')
    res_ts = pd.read_csv('tmp/pred_ts.csv')
    res_rn, _ = screening_res_a(res_rn, tasks, participants)
    res_ts, _ = screening_res_a(res_ts, tasks, participants)
    df = mk_anova_frame_a(res_rn, res_ts)
    df = redefine_task_a(df, tasks)

    for measure in ['loss',]:
        aov = pg.rm_anova(data=df, dv=measure, subject='subject', within=['task', 'model'])
        print(aov.round(3))
        post_hocs = pg.pairwise_tests(data=df, dv=measure, subject='subject', within=['task', 'model'], padjust='bonf')
        print(post_hocs.round(3))
        plt.figure(1).clf()
        xticks, xticklabels = [], []
        x = .0
        for task in df['task'].unique():
            a, b = df.query('task==@task & model=="rn"'), df.query('task==@task & model=="ts"')
            b1, b2 = plot_paired(a, b, x, measure)
            xticks.append(x + .5)
            xticklabels.append(task.upper())
            x += 2.5
        plt.xticks(xticks, xticklabels)
        # plt.legend([b1[0], b2[0]], ['RTNet', 'Task solver'], loc='lower right')
        lg = plt.legend([b1['boxes'][0], b2['boxes'][0]], ['RTNet', 'Task solver'], loc='lower right'); lg.set_zorder(20)
        plt.xlabel('Task')
        plt.savefig('figs/{}_sa.pdf'.format(measure), bbox_inches='tight', transparent=True)


    print('\n')
    print('=====================')
    print('Evaluate Situation SB')
    print('=====================')
    set_abbr_tasks = [
            'esea', 'daea', 'dsea',
            'eaes', 'daes', 'dses',
            'eada', 'esda', 'dsda',
            'eads', 'esds', 'dads',
            ]
    df = pd.read_csv('tmp/pred_ed.csv') # created by eval_enc_dec.py
    df, cmbns = screening_res_b(df, set_abbr_tasks, participants)
    df = mk_anova_frame_b(df)
    df = redefine_task_b(df, cmbns)

    fig_size = plt.figure(1).get_size_inches()
    fig_size[0] = fig_size[0] * 2
    plt.figure(2, figsize=list(fig_size))
    ylabel = {'loss': 'Negative log-likelihood', 'match': '\\%match to behavior', 'correct': '\\%correct'}
    for measure in ['loss', 'match', 'correct']:
    # for measure in ['loss',]:
        lg_loc = 'lower right' if measure == 'loss' else 'lower left'

        aov = pg.rm_anova(data=df, dv=measure, subject='subject', within=['trans_to', 'task'])
        print(aov.round(3))
        post_hocs = pg.pairwise_tests(data=df, dv=measure, subject='subject', within=['task', 'trans_to'], padjust='bonf')
        print(post_hocs.round(3))
        post_hocs.to_csv('tmp/others_{}.csv'.format(measure))
        plt.figure(2).clf()
        xticks, xticklabels = [], []
        x = .0
        for task in df['task'].unique():
            a, b = df.query('task==@task & trans_to=="org"'), df.query('task==@task & trans_to=="opp"')
            b1, b2 = plot_paired(a, b, x, measure, ylabel=ylabel[measure])
            xticks.append(x + .5)
            xticklabels.append(text_trans_set(task, new_line=False))
            x += 2.5
        plt.xticks(xticks, xticklabels, fontsize=15)
        # plt.legend([b1[0], b2[0]], ['Original', 'Others'], loc='lower right')
        lg = plt.legend([b1['boxes'][0], b2['boxes'][0]], ['Original', 'Others'], loc=lg_loc); lg.set_zorder(20)
        plt.savefig('figs/{}_org.pdf'.format(measure), bbox_inches='tight', transparent=True)
