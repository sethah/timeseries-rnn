import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt

import mxnet as mx

import seaborn as sns


def tsplot(y, lags=None, figsize=(10, 8)):
    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    y.plot(ax=ts_ax)
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    [ax.set_xlim(0.5) for ax in [acf_ax, pacf_ax]]
    sns.despine()
    plt.tight_layout()
    return ts_ax, acf_ax, pacf_ax

def predict_dynamic(model, seq_input, other_input, predict_seq_len):

    pred_batch_size, input_seq_len, feature_dim = seq_input.shape

    # this buffer holds the input and output as we fill things in
    inp_buffer = mx.nd.zeros((pred_batch_size, input_seq_len + predict_seq_len, feature_dim),
                             ctx=seq_input.context)
    inp_buffer[:, :input_seq_len, :] = seq_input
    t_buffer = mx.nd.zeros((pred_batch_size, predict_seq_len, feature_dim),
                             ctx=seq_input.context)

    num_predictions = 1
    inp = inp_buffer[:, :input_seq_len, :]
    for j, other_inp in enumerate(other_input):
        t_output = model.tcn.forward(inp)
        output = model.forward((inp, other_inp))
        # for train_sequences, we need to grab every num_predictionth prediction
        # output_idx = mx.nd.arange(0, output.shape[0], num_predictions,
        #                           ctx=seq_input.context) + num_predictions - 1
        # last_outputs = output[output_idx, :]
        inp_buffer[:, input_seq_len + j:input_seq_len + j + 1, :] = \
            output.reshape((pred_batch_size, 1, -1))
        t_buffer[:, j:j + 1, :] = t_output.reshape((pred_batch_size, 1, -1))
        inp = inp_buffer[:, j + 1:input_seq_len + j + 1, :]
    outputs = inp_buffer[:, input_seq_len:, :]
    return outputs, t_buffer
