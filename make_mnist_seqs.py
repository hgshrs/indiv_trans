import argparse
import importlib
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import uuid

import rtnet as rtn
importlib.reload(rtn)
import analyze_bhv_mnist as ab
importlib.reload(ab)

def give_evidence(det_model, guide, x):
    det_model.load_state_dict(guide(x, None))
    det_model.eval()
    return torch.exp(det_model(x).detach())

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='Simuate RTNet')
    parser.add_argument('--init', action='store_true', default=False,
                        help='initializes network paramters')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--n-generated-seqs', type=int, default=100000, metavar='N',
                        help='number of generated sequences (default: 0)')
    parser.add_argument('--fast-bnn', action='store_true', default=False,
                        help='use fast computation of BNN using preloaded weights')
    parser.add_argument('--light', action='store_true', default=False,
                        help='Use a light-weight model or not')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = 'cuda:0' if use_cuda else 'cpu'
    print('Light weight: {}'.format(args.light))
    # print('Dry run: {}'.format(args.dry_run))
    print('Device: {}'.format(device))
    print('Fast BNN: {}'.format(args.fast_bnn))


    ########################
    # Create marged sequence datasets (cnn_score and bhv)
    ########################
    seq_path = 'tmp/seqs.csv'
    print('Sequence file is loaded from and saved to {}.'.format(seq_path))
    try:
        dfseqs = pd.read_csv(seq_path)
        n_seqs = len(dfseqs['uid'].unique())
    except:
        dfseqs = pd.DataFrame()
        n_seqs = 0

    if n_seqs < args.n_generated_seqs:
        # Load pretrained model
        net_path, net_param_path, model, guide, _, _, _, _ = rtn.mk_pyro_model(path='tmp/bnn{}.pt', light=args.light, device=device, guide_hide=True)
        det_model = rtn.Net(use_pyro=False, light_weight=args.light).to(device)

        # Load behavioral data
        bhv = ab.load_bhv(path='rtnet/behavioral data.csv', reject=True)
        part_idxs = bhv.subject.unique()
        n_resps = len(bhv)
        n_parts = len(part_idxs)

        # Load mnist data
        test_loader = rtn.create_mnist_loaders(sets='test', test_batch_size=16)

        # Marge
        rt2frame_resol = .1 # RT resolution for conversion from sec to frame
        max_n_frames = int(np.round(bhv['resp_rt'].max() / rt2frame_resol))
        noise_lvs = {'easy': 2.1, 'difficult': 2.9}
        for ii in tqdm(range(n_seqs, args.n_generated_seqs)):

            uid = str(uuid.uuid4())[:8]
            tt = np.random.randint(n_resps)
            s = bhv.iloc[tt]
            img = test_loader.dataset[s['mnist_index'] - 1][0]
            img = img.view(1, 1, 227, 227)
            true_label = test_loader.dataset[s['mnist_index'] - 1][1]
            if s['stim'] != true_label:
                print('Warning: the true labels on MNIST and bhv are not match!')

            n_frames = int(np.round(s['resp_rt'] / rt2frame_resol))
            imgn = img + noise_lvs[s['difficulty']] * torch.rand(img.shape)
            imgn = imgn.to(device)

            if args.fast_bnn:
                n_det_models = 100
                if (ii == n_seqs) or (ii % 1000 == 0):
                    det_models = []
                    for jj in range(n_det_models):
                        _det_model = rtn.Net(use_pyro=False, light_weight=args.light).to(device)
                        _det_model.load_state_dict(guide(imgn, None))
                        _det_model.eval()
                        det_models.append(_det_model)

            df_seq = pd.DataFrame()
            for ff in range(max_n_frames):

                if args.fast_bnn:
                    evid = torch.exp(det_models[np.random.randint(n_det_models)](imgn).detach())[0]
                else:
                    evid = give_evidence(det_model, guide, imgn)[0]

                sd = s.to_frame().T
                sd['frame'] = [ff]
                sd['evidence0'] = [evid[0].item()]
                sd['evidence1'] = [evid[1].item()]
                sd['evidence2'] = [evid[2].item()]
                sd['evidence3'] = [evid[3].item()]
                sd['evidence4'] = [evid[4].item()]
                sd['evidence5'] = [evid[5].item()]
                sd['evidence6'] = [evid[6].item()]
                sd['evidence7'] = [evid[7].item()]
                sd['evidence8'] = [evid[8].item()]
                sd['evidence9'] = [evid[9].item()]
                df_seq = pd.concat([df_seq, sd], axis=0)
            df_seq['n_frames'] = [n_frames] * max_n_frames
            df_seq['uid'] = [uid] * max_n_frames
            dfseqs = pd.concat([dfseqs, df_seq], axis=0)

            if args.fast_bnn:
                if (ii == (args.n_generated_seqs - 1)) or (ii % 100 == 0):
                    dfseqs.to_csv(seq_path, index=False)
