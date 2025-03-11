import pandas as pd
import matplotlib.pylab as plt
import numpy as np

def load_bhv(path='rtnet/behavioral data.csv', reject=True):
    bhv_path = 'rtnet/behavioral data.csv'
    bhv = pd.read_csv(bhv_path)

    if reject:
        # Participant rejection
        part_idxs = bhv.subject.unique()
        n_parts = len(part_idxs)
        p_correct = np.zeros(n_parts)
        av_conf = np.zeros(n_parts)
        rt_af = np.zeros(n_parts)
        rt_sf = np.zeros(n_parts)
        for pp, part in enumerate(part_idxs):
            bhv_ = bhv[bhv['subject'] == part]
            correct = bhv_['correct']
            p_correct[pp] = correct.sum() / len(correct)
            rt_af[pp] = bhv_[bhv_['sat'] == 'accuracy focus']['resp_rt'].mean()
            rt_sf[pp] = bhv_[bhv_['sat'] == 'speed focus']['resp_rt'].mean()
            av_conf[pp] = bhv_['confidence'].mean()

        rm_parts = []
        # Remove: speed focus < accuracy focus in RT
        rm_parts += list(part_idxs[rt_af < rt_sf])
        # Remove: confidence > 3.85
        rm_parts += list(part_idxs[av_conf > 3.85])

        bhv = bhv.query('subject not in @rm_parts')
        part_idxs = bhv.subject.unique()
        print('Remove participants: {} - {} = {} ({:.2%}%)'.format(n_parts, n_parts - len(part_idxs), len(part_idxs), (n_parts - len(part_idxs)) / n_parts))

        # Individual trial rejection
        q1 = bhv['resp_rt'].quantile(.25)
        q3 = bhv['resp_rt'].quantile(.75)
        lb = q1 - 1.5 * (q3 - q1)
        ub = q3 + 1.5 * (q3 - q1)
        n_trials = len(bhv)
        bhv = bhv.query('resp_rt > @lb')
        bhv = bhv.query('resp_rt < @ub')
        print('Remove Individual trials: {} - {} = {} ({:.2%}%)'.format(n_trials, n_trials - len(bhv), len(bhv), (n_trials - len(bhv)) / n_trials))
    return bhv

if __name__ == '__main__':
    bhv = load_bhv(path='rtnet/behavioral data.csv', reject=True)
    part_idxs = bhv.subject.unique()
    n_parts = len(part_idxs)

    plt.figure(3).clf()
    plt.subplot(121)
    plt.hist(bhv[bhv['difficulty'] == 'easy']['resp_rt'], alpha=.5)
    plt.hist(bhv[bhv['difficulty'] == 'difficult']['resp_rt'], alpha=.5)
    plt.legend(['easy', 'difficult'])
    plt.xlabel('RT [s]')
    plt.ylabel('#samples')
    plt.subplot(122)
    plt.hist(bhv[bhv['sat'] == 'accuracy focus']['resp_rt'], alpha=.5)
    plt.hist(bhv[bhv['sat'] == 'speed focus']['resp_rt'], alpha=.5)
    plt.legend(['accuracy', 'speed'])
    plt.xlabel('RT [s]')

    fp = plt.figure(1); fp.clf()
    fr = plt.figure(2); fr.clf()
    subp_idx = 1
    for dd, difficulty in enumerate(['easy', 'difficult']):
        pc_two = np.zeros([n_parts, 2])
        rt_two = np.zeros([n_parts, 2])
        xylabels = []
        for ff, focus in enumerate(['accuracy focus', 'speed focus']):
            xylabels += [focus]
            for pp, part in enumerate(part_idxs):
                bhv_ = bhv.query('subject == @part & difficulty == @difficulty & sat == @focus')
                pc_two[pp, ff] = bhv_['correct'].mean()
                rt_two[pp, ff] = bhv_['resp_rt'].mean()

        ax = fp.add_subplot(2, 2, subp_idx)
        ax.plot(pc_two[:, 0], pc_two[:, 1], 'o')
        ax.set_xlabel(xylabels[0])
        ax.set_ylabel(xylabels[1])
        ax.set_title(difficulty)

        ax = fr.add_subplot(2, 2, subp_idx)
        ax.plot(rt_two[:, 0], rt_two[:, 1], 'o')
        ax.set_xlabel(xylabels[0])
        ax.set_ylabel(xylabels[1])
        ax.set_title(difficulty)
        ax.set_aspect('equal')
        subp_idx += 1

    for ff, focus in enumerate(['accuracy focus', 'speed focus']):
        pc_two = np.zeros([n_parts, 2])
        rt_two = np.zeros([n_parts, 2])
        xylabels = []
        for dd, difficulty in enumerate(['easy', 'difficult']):
            xylabels += [difficulty]
            for pp, part in enumerate(part_idxs):
                bhv_ = bhv.query('subject == @part & difficulty == @difficulty & sat == @focus')
                pc_two[pp, dd] = bhv_['correct'].mean()
                rt_two[pp, dd] = bhv_['resp_rt'].mean()

        ax = fp.add_subplot(2, 2, subp_idx)
        ax.plot(pc_two[:, 0], pc_two[:, 1], 'o')
        ax.set_xlabel(xylabels[0])
        ax.set_ylabel(xylabels[1])
        ax.set_title(focus)

        ax = fr.add_subplot(2, 2, subp_idx)
        ax.plot(rt_two[:, 0], rt_two[:, 1], 'o')
        ax.set_xlabel(xylabels[0])
        ax.set_ylabel(xylabels[1])
        ax.set_title(focus)
        ax.set_aspect('equal')
        subp_idx += 1
