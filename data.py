import mxnet as mx
import mxnet.gluon as gluon

import pandas as pd
import numpy as np


def _gen_stationary(n, ar_coefs, seasonal_period=10, seasonal_coef=0.0):
    if isinstance(ar_coefs, list) or isinstance(ar_coefs, np.ndarray):
        ar_coefs = mx.nd.array(ar_coefs)
    alpha = seasonal_coef
    d = ar_coefs.shape[0]
    h = seasonal_period
    x = mx.nd.random.normal(0, 1, h + d + n)
    for i in range(h + d, h + d + n):
        x[i] += mx.nd.dot(ar_coefs, x[i - d:i]) + \
                alpha * (x[i - h] - mx.nd.dot(ar_coefs, x[i - h - d:i - h]))
    return x[h + d:]

def make_index(seq, start=pd.to_datetime('2016-03-01')):
    end = start + pd.Timedelta(seq.shape[0] - 1, 'd')
    idx = pd.DatetimeIndex(start=start, end=end, freq='d')
    return pd.DataFrame(seq, index=idx)

class TSArrayDataset(gluon.data.Dataset):

    def __init__(self, endog, in_seq_len, exog_lags=None, out_seq_len=1):
        self.endog = endog
        self.exog_lags = exog_lags
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len

    def __getitem__(self, idx):
        data = self.endog[idx:idx + self.in_seq_len]
        label = self.endog[idx + self.in_seq_len:idx + self.in_seq_len + self.out_seq_len]

        if self.exog_lags is not None:
            exogs = []
            for lag in self.exog_lags:
                if idx + self.in_seq_len - lag < 0:
                    lag_seq = mx.nd.zeros(self.out_seq_len)
                else:
                    lag_seq = self.endog[idx + self.in_seq_len - lag:idx + self.in_seq_len + self.out_seq_len - lag]
                exogs.append(lag_seq.reshape((-1, 1)))
            exog = mx.nd.concat(*exogs, dim=1)
            return (data.reshape((-1, 1)), exog), label.reshape((-1, 1))
        else:
            return data.reshape((-1, 1)), label.reshape((-1, 1))

    def __len__(self):
        return self.endog.shape[0] - self.in_seq_len - self.out_seq_len


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
            y += bin_exog * 2
            exog = mx.nd.array(bin_exog[out_start:].reshape((-1, 1)))
        y = y.reshape((-1, 1))
        return (mx.nd.array(y[:self.in_seq_len]), exog), mx.nd.array(y[out_start:])

    def __len__(self):
        return self.length

class PeriodicExogDataset(gluon.data.Dataset):
    def __init__(self, in_seq_len, out_seq_len=1, ar_coefs=[0.3],
                 period=1, spacing=10., length=1000, lam=0.3):
        self.in_seq_len = in_seq_len
        self.out_seq_len = out_seq_len
        self.period = period
        self.spacing = period / spacing
        self.length = length
        self.lam = lam
        self.ar_coefs = ar_coefs

    def _gen_exog(self, n):
        nswitches = np.random.poisson(lam=self.lam)
        idx = np.sort(np.random.choice(np.arange(n), nswitches, replace=False))
        idx = np.concatenate([idx, np.array([n])])
        start = np.random.randint(0, 2)
        bin_exog = np.zeros(n) + start
        switch = start == 0
        for a, b in zip(idx, np.roll(idx, -1)):
            bin_exog[a:b] = int(switch)
            switch = not switch
        return mx.nd.array(bin_exog)

    def _gen_stationary(self, n, coefs, c=0.0, eps=0.1, trend=0.0):
        if isinstance(coefs, list) or isinstance(coefs, np.ndarray):
            coefs = mx.nd.array(coefs)
        alpha = 0.0
        h = self.period
        d = coefs.shape[0]
        x = mx.nd.random.normal(0, 1, h + d + n)
        for i in range(h + d, h + d + n):
            x[i] += mx.nd.dot(coefs, x[i - d:i]) + \
                   alpha * (x[i - h] - mx.nd.dot(coefs, x[i - h - d:i - h]))
        return x[h + d:]

    def __getitem__(self, idx):
        # offset = np.random.random() * self.period
        n = self.in_seq_len + self.out_seq_len
        z = self._gen_stationary(n, self.ar_coefs)
        # exog = self._gen_exog(n + self.period)
        # for i in range(self.period, n + self.period):
        #     z[i] += 0.5 * z[i - self.period] + 0.2 * z[i - 365]

        # x = np.linspace(0, n * self.spacing, n)
        # noise = np.random.normal(0, 1, n) * 0.1
        # y = np.sin(2 * np.pi / self.period * (x - offset)) + noise
        # out_len = n - 1 if self.return_sequences else self.out_seq_len
        # out_start = 1 if self.return_sequences else self.in_seq_len
        out_start = self.in_seq_len
        # if self.lam != 0.0:
        #     y += bin_exog * 2
        #     exog = mx.nd.array(bin_exog[out_start:].reshape((-1, 1)))
        y = z.reshape((-1, 1))
        return mx.nd.array(y[:self.in_seq_len]), mx.nd.array(y[out_start:])

    def __len__(self):
        return self.length
