import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
from pyro.infer.autoguide import AutoDiagonalNormal, AutoNormal
from pyro.infer import SVI, Trace_ELBO, Predictive
from tqdm.auto import trange, tqdm
import numpy as np

class Net(PyroModule):
    def __init__(self, use_pyro=True, light_weight=False, device='cpu'):
        super(Net, self).__init__()
        self.use_pyro = use_pyro
        self.light_weight = light_weight
        if self.use_pyro:
            self.conv1 = PyroModule[nn.Conv2d](in_channels=1, out_channels=96, kernel_size=11, stride=4, padding=0)
            self.conv1.weight = PyroSample(dist.Normal(torch.tensor(0., device=device), 1.).expand(self.conv1.weight.shape).to_event(self.conv1.weight.dim()))
            self.conv1.bias = PyroSample(dist.Normal(torch.tensor(0., device=device), 1.).expand(self.conv1.bias.shape).to_event(self.conv1.bias.dim()))

            self.conv2 = PyroModule[nn.Conv2d](in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2)
            self.conv2.weight = PyroSample(dist.Normal(torch.tensor(0., device=device), 1.).expand(self.conv2.weight.shape).to_event(self.conv2.weight.dim()))
            self.conv2.bias = PyroSample(dist.Normal(torch.tensor(0., device=device), 1.).expand(self.conv2.bias.shape).to_event(self.conv2.bias.dim()))

            if not self.light_weight:
                self.conv3 = PyroModule[nn.Conv2d](in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1)
                self.conv3.weight = PyroSample(dist.Normal(torch.tensor(0., device=device), 1.).expand(self.conv3.weight.shape).to_event(self.conv3.weight.dim()))
                self.conv3.bias = PyroSample(dist.Normal(torch.tensor(0., device=device), 1.).expand(self.conv3.bias.shape).to_event(self.conv3.bias.dim()))

                self.conv4 = PyroModule[nn.Conv2d](in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
                self.conv4.weight = PyroSample(dist.Normal(torch.tensor(0., device=device), 1.).expand(self.conv4.weight.shape).to_event(self.conv4.weight.dim()))
                self.conv4.bias = PyroSample(dist.Normal(torch.tensor(0., device=device), 1.).expand(self.conv4.bias.shape).to_event(self.conv4.bias.dim()))

                self.conv5 = PyroModule[nn.Conv2d](in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
                self.conv5.weight = PyroSample(dist.Normal(torch.tensor(0., device=device), 1.).expand(self.conv5.weight.shape).to_event(self.conv5.weight.dim()))
                self.conv5.bias = PyroSample(dist.Normal(torch.tensor(0., device=device), 1.).expand(self.conv5.bias.shape).to_event(self.conv5.bias.dim()))

            self.fc1 = PyroModule[nn.Linear](256*6*6, 4096)
            self.fc1.weight = PyroSample(dist.Normal(torch.tensor(0., device=device), 1.).expand(self.fc1.weight.shape).to_event(self.fc1.weight.dim()))
            self.fc1.bias = PyroSample(dist.Normal(torch.tensor(0., device=device), 1.).expand(self.fc1.bias.shape).to_event(self.fc1.bias.dim()))

            if not self.light_weight:
                self.fc2 = PyroModule[nn.Linear](4096, 4096)
                self.fc2.weight = PyroSample(dist.Normal(torch.tensor(0., device=device), 1.).expand(self.fc2.weight.shape).to_event(self.fc2.weight.dim()))
                self.fc2.bias = PyroSample(dist.Normal(torch.tensor(0., device=device), 1.).expand(self.fc2.bias.shape).to_event(self.fc2.bias.dim()))

            self.fc3 = PyroModule[nn.Linear](4096, 10)
            self.fc3.weight = PyroSample(dist.Normal(torch.tensor(0., device=device), 1.).expand(self.fc3.weight.shape).to_event(self.fc3.weight.dim()))
            self.fc3.bias = PyroSample(dist.Normal(torch.tensor(0., device=device), 1.).expand(self.fc3.bias.shape).to_event(self.fc3.bias.dim()))
        else:
            self.conv1 = nn.Conv2d(1, 96, 11, 4, 0)
            self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
            if not self.light_weight:
                self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
                self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
                self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)
            self.fc1 = nn.Linear(256*6*6, 4096)
            if not self.light_weight:
                self.fc2 = nn.Linear(4096, 4096)
            self.fc3 = nn.Linear(4096, 10)

    def forward(self, x, y=None):
        # conv
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 3, 2)
        if not self.light_weight:
            x = self.conv3(x)
            x = F.relu(x)
            x = self.conv4(x)
            x = F.relu(x)
            x = self.conv5(x)
            x = F.relu(x)
        x = F.max_pool2d(x, 3, 2)

        # linear
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, .5)
        if not self.light_weight:
            x = F.relu(self.fc2(x))
        x = F.dropout(x, .5)
        x = self.fc3(x)

        output = F.log_softmax(x, dim=1)
        if self.use_pyro:
            with pyro.plate('data', x.shape[0]):
                obs = pyro.sample('obs', dist.Categorical(logits=output), obs=y)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total_n = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_n += len(data)
            if args.dry_run:
                break

    test_loss /= total_n

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total_n, 100. * correct / total_n))

    return test_loss

def create_mnist_loaders(sets=['train', 'test'], train_batch_size=64, test_batch_size=1000, use_cuda=False):
    train_kwargs = {'batch_size': train_batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    if type(sets) == list:
        dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
        dataset2 = datasets.MNIST('./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
        return train_loader, test_loader
    elif sets == 'train':
        dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
        return train_loader
    elif sets == 'test':
        dataset2 = datasets.MNIST('./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
        return test_loader

def mk_model(path='tmp/cnn{}.pt', light=False, lr=1e-4, device='cpu'):
    net_kwargs = {'use_pyro': False, 'light_weight': light, 'device':device}
    pa = 'light' if light else ''
    net_path = path.format(pa)
    model = Net(**net_kwargs).to(device)
    try:
        loaded = torch.load(net_path, weights_only=False)
        model.load_state_dict(loaded['model'], map_location=device)
        epoch_loss_train = loaded['train_loss']
        epoch_loss_test = loaded['test_loss']
        print('Succeeded load: {} at {} epoch'.format(net_path, len(epoch_loss_train)))
    except:
        print('Failed to load: {}'.format(net_path))
        epoch_loss_train = []
        epoch_loss_test = []
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    return net_path, model, optimizer, epoch_loss_train, epoch_loss_test

def mk_pyro_model(path='tmp/bnn{}.pt', light=False, lr=1e-4, device='cpu', guide_hide=False):
    net_kwargs = {'use_pyro': True, 'light_weight': light, 'device': device}
    pa = 'light' if light else ''
    net_path = path.format(pa)
    net_param_path = path.format(pa) + '.params'
    model = Net(**net_kwargs).to(device)
    try:
        pyro.clear_param_store()
        loaded = torch.load(net_path, weights_only=False)
        model.load_state_dict(loaded['model'])
        pyro.get_param_store().load(net_param_path, map_location=device)
        epoch_loss_train = loaded['train_loss']
        epoch_loss_test = loaded['test_loss']
        print('Succeeded to load: {}, {} at {} epoch'.format(net_path, net_param_path, len(epoch_loss_train)))
    except:
        print('Failed to load: {}, {}'.format(net_path, net_param_path))
        pyro.clear_param_store()
        epoch_loss_train = []
        epoch_loss_test = []
    if guide_hide:
        guide = AutoDiagonalNormal(pyro.poutine.block(model, hide=['obs']))
    else:
        guide = AutoDiagonalNormal(model)
    # guide = AutoNormal(model)
    adam = pyro.optim.Adam({"lr": lr})
    loss_fn_ = Trace_ELBO()
    svi = SVI(model, guide, adam, loss=loss_fn_)
    loss_fn = loss_fn_(model, guide)
    return net_path, net_param_path, model, guide, loss_fn, svi, epoch_loss_train, epoch_loss_test

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')
    # parser.add_argument('--save-model', action='store_true', default=False,
                        # help='For Saving the current Model')
    parser.add_argument('--pyro', action='store_true', default=False,
                        help='Use pyro or not')
    parser.add_argument('--light', action='store_true', default=False,
                        help='Use a light-weight model or not')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()
    print('Use pyro: {}'.format(args.pyro))
    print('Light weight: {}'.format(args.light))
    print('Dry run: {}'.format(args.dry_run))

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda:0")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load data
    train_loader, test_loader = create_mnist_loaders(sets=['train', 'test'], train_batch_size=args.batch_size, test_batch_size=args.test_batch_size, use_cuda=use_cuda)

    # Load model
    net_kwargs = {'use_pyro': args.pyro, 'light_weight': args.light}
    pa = 'light' if args.light else ''
    if args.pyro:
        net_path = 'tmp/bnn{}.pt'
        net_path, net_param_path, model, guide, loss_fn, svi, epoch_loss_train, epoch_loss_test = mk_pyro_model(path=net_path, light=args.light, lr=args.lr, device=device)
    else:
        net_path = 'tmp/cnn{}.pt'
        net_path, model, optimizer, epoch_loss_train, epoch_loss_test = mk_model(path=net_path, light=args.light, lr=args.lr, device=device)

    # train
    for epoch in range(1, args.epochs + 1):
        batch_ave_loss = []
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            if args.pyro:
                loss = svi.step(data, target)
                batch_ave_loss += [loss / (len(data))]
            else:
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                batch_ave_loss += [loss.item() / len(data)]

            if ((batch_idx+1) % args.log_interval == 0) or (batch_idx == 0) or (batch_idx == len(train_loader)-1):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3e}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), np.mean(batch_ave_loss)))
            if args.dry_run:
                break
        epoch_loss_train += [np.mean(batch_ave_loss)]

        if args.pyro:
            n_samplings = 4
            predictive = Predictive(model, guide=guide, num_samples=n_samplings)
            correct = 0
            total_sampling = 0
            batch_ave_loss = []
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                sampling_pred = predictive(data)['obs']
                for ss in range(n_samplings):
                    correct += (sampling_pred[ss] == target).sum()
                    total_sampling += len(target)
                batch_ave_loss += [loss_fn(data, target).item() / len(data)]
                if args.dry_run:
                    break
            print("Test set: Accuracy: {}/{} ({:.2f}%)\tLoss: {:.3e}\n".format(
                    correct, total_sampling, 100.0 * correct / total_sampling, np.mean(batch_ave_loss)))
            epoch_loss_test += [np.mean(batch_ave_loss)]
        else:
            loss = test(model, device, test_loader)
            epoch_loss_test += [loss]

        torch.save({'model': model.state_dict(), 'train_loss': epoch_loss_train, 'test_loss': epoch_loss_test}, net_path)
        if args.pyro:
            pyro.get_param_store().save(net_param_path)
