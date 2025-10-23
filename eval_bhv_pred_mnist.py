import os
import importlib
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import scipy.stats

import analyze_bhv_mnist as ab
importlib.reload(ab)
import train_it_mnist as ted
importlib.reload(ted)

plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 20
plt.rcParams['text.latex.preamble'] = "\\usepackage{sfmath}"

if __name__ == '__main__':
    tasks = 'eads'
    difficulty_src, sat_src, difficulty_trg, sat_trg = ted.translate_tasks(tasks)

    bhv_df = ab.load_bhv(path='rtnet/behavioral data.csv', reject=True)
    pred_df = pd.read_csv('tmp/pred_ed.csv') # created by eval_enc_dec.py
    bhv_df = bhv_df.query('difficulty==@difficulty_trg & sat==@sat_trg')
    pred_df = pred_df.query('difficulty_src==@difficulty_src & sat_src==@sat_src & difficulty_src==@difficulty_src & sat_src==@sat_src')
    iddf = pd.read_csv('tmp/split_participants_train_valid_test.csv')
    try:
        os.system('mkdir figs/trial/{}'.format(tasks))
    except:
        pass

    participants = iddf.query('set == "test"')['id']
    n_partici = len(participants)
    p_correct = np.zeros([n_partici, 2])
    for ss, sbj in enumerate(participants):
        p_correct[ss, 0] = bhv_df[bhv_df['subject'] == sbj]['correct'].mean()
        p_correct[ss, 1] = pred_df[pred_df['subject'] == sbj]['correct'].mean()

    # ress = scipy.stats.pearsonr(*p_correct.T)
    # print('[{}, {}] C={:.3f}, p={:3f}'.format(difficulty_trg, sat_trg, ress.statistic, ress.pvalue))
    plt.figure(1).clf()
    plt.plot(p_correct[:, 0], p_correct[:, 1], 'o')

    # trial-wise analysis
    pred_trial_df = pd.read_csv('tmp/pred_{}.csv'.format(tasks)) # created by eval_enc_dec.py
    labels = np.sort(pred_trial_df['true_labels'].unique())
    p_correct = np.zeros([len(participants), len(labels), 2])
    for ss, sbj in enumerate(participants):
        for ll, true_label in enumerate(labels):
            p_correct[ss, ll, 0] = bhv_df.query('subject==@sbj & stim==@true_label')['correct'].mean()
            _df = pred_trial_df.query('subject==@sbj & true_labels==@true_label')
            p_correct[ss, ll, 1] = (_df['true_labels'] == _df['pred_labels']).sum() / len(_df)

        plt.figure(2).clf()
        width = .4
        plt.bar(labels - .5 * width, p_correct[ss, :, 0] * 100, width=width, ec='k', fc='w')
        plt.bar(labels + .5 * width, p_correct[ss, :, 1] * 100, width=width, ec='k', fc='gray')
        plt.xticks(labels, labels)
        plt.ylim([0, 100])
        plt.ylabel('Rate for correct response')
        plt.xlabel('Stimulus digit')
        plt.pause(.1)
        if sbj == 47:
            plt.legend(['Actual behaviour', 'Predicted'], loc='lower right')
        plt.savefig('figs/trial/{}/p{}.pdf'.format(tasks, sbj), bbox_inches='tight', transparent=True)
