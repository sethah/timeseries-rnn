import mxnet as mx
import mxnet.gluon as gluon

import numpy as np


class GenSequenceDataset(gluon.data.Dataset):
    def __init__(self, in_seq_len, out_seq_len=1, period=1., spacing=10., length=1000, lam=0.3):
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.period = period
        self.spacing = period / spacing
        self.length = length
        self.lam = lam

    def __getitem__(self, idx):
        offset = np.random.random() * self.period
        n = self.in_seq_len + self.out_seq_len
        x = np.linspace(0, n * self.spacing, n)
        noise = np.random.normal(0, 1, n) * 0.1
        y = np.sin(2 * np.pi / self.period * (x - offset)) + noise
        exog = mx.nd.zeros((self.out_seq_len, 1))
        if self.lam != 0.0:
            nswitches = np.random.poisson(lam=self.lam)
            idx = np.sort(np.random.choice(np.arange(n), nswitches, replace=False))
            idx = np.concatenate([idx, np.array([n])])
            start = np.random.randint(0, 2)
            bin_exog = np.zeros(n) + start
            switch = start == 0
            for a, b in zip(idx, np.roll(idx, -1)):
                bin_exog[a:b] = int(switch)
                switch = not switch
            y += bin_exog * 1
            exog = mx.nd.array(bin_exog[self.in_seq_len:].reshape((-1, 1)))
        y = y.reshape((-1, 1))
        return mx.nd.array(y[:self.in_seq_len]), exog, mx.nd.array(y[self.in_seq_len:])

    def __len__(self):
        return self.length


class GenSequenceFullDataset(gluon.data.Dataset):
    def __init__(self, in_seq_len, out_seq_len=1, return_sequences=False,
                 period=1., spacing=10., length=1000, lam=0.3):
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.period = period
        self.spacing = period / spacing
        self.length = length
        self.lam = lam
        self.return_sequences = return_sequences

    def __getitem__(self, idx):
        offset = np.random.random() * self.period
        n = self.in_seq_len + self.out_seq_len
        x = np.linspace(0, n * self.spacing, n)
        noise = np.random.normal(0, 1, n) * 0.1
        y = np.sin(2 * np.pi / self.period * (x - offset)) + noise
        out_len = n - 1 if self.return_sequences else self.out_seq_len
        out_start = 1 if self.return_sequences else self.in_seq_len
        exog = mx.nd.zeros((out_len, 1))
        if self.lam != 0.0:
            nswitches = np.random.poisson(lam=self.lam)
            idx = np.sort(np.random.choice(np.arange(n), nswitches, replace=False))
            idx = np.concatenate([idx, np.array([n])])
            start = np.random.randint(0, 2)
            bin_exog = np.zeros(n) + start
            switch = start == 0
            for a, b in zip(idx, np.roll(idx, -1)):
                bin_exog[a:b] = int(switch)
                switch = not switch
            y += bin_exog * 1
            exog = mx.nd.array(bin_exog[out_start:].reshape((-1, 1)))
        y = y.reshape((-1, 1))
        return mx.nd.array(y[:self.in_seq_len]), exog, mx.nd.array(y[out_start:])

    def __len__(self):
        return self.length
