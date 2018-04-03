import mxnet as mx
import mxnet.gluon as gluon
import mxnet.gluon.nn as nn


class CutRight(gluon.Block):
    def __init__(self, cut_size):
        """
        We only want to pad the left side (beginning) of a sequence. Use this
        to cut off the padded elements from the right side.
        :param cut_size:
        """
        super(CutRight, self).__init__()
        self.cut_size = cut_size

    def forward(self, x):
        return x[:, :, :-self.cut_size]


class TemporalBlock(gluon.Block):
    def __init__(self, k, d, out_channels, in_channels=1, dropout=0.4, prefix=None):
        super(TemporalBlock, self).__init__(prefix=prefix)
        with self.name_scope():
            self.in_channels = in_channels
            self.kernel_size = k
            self.dilation = d
            self.out_channels = out_channels

            self.net = gluon.nn.Sequential()
            self.conv1 = gluon.nn.Conv1D(out_channels, k, in_channels=in_channels, dilation=d,
                                         padding=d * (k - 1))
            self.cut1 = CutRight(d * (k - 1))
            self.bn1 = gluon.nn.BatchNorm()
            self.relu1 = gluon.nn.Activation('relu')
            self.dropout1 = gluon.nn.Dropout(dropout)
            self.conv2 = gluon.nn.Conv1D(out_channels, k, in_channels=out_channels, dilation=d,
                                         padding=d * (k - 1))
            self.cut2 = CutRight(d * (k - 1))
            self.bn2 = gluon.nn.BatchNorm()
            self.relu2 = gluon.nn.Activation('relu')
            self.dropout2 = gluon.nn.Dropout(dropout)
            layers = [self.conv1, self.cut1, self.bn1, self.relu1, self.dropout1, self.conv2, self.cut2,
                      self.bn2, self.relu2, self.dropout2]
            for layer in layers:
                self.net.add(layer)
            self.relu = nn.Activation('relu')
            self.downsample = nn.Conv1D(out_channels, 1) if in_channels != out_channels else None

    def forward(self, inputs):
        out = self.net(inputs)
        res = inputs if self.downsample is None else self.downsample(inputs)
        return self.relu(out + res)


class TemporalConvNet(gluon.Block):
    def __init__(self, channel_list, in_channels, dropout=0.0, kernel_size=2, prefix=None):
        super(TemporalConvNet, self).__init__(prefix=prefix)
        with self.name_scope():
            self.net = nn.Sequential()
            num_levels = len(channel_list)
            for i in range(num_levels):
                dilation = 2 ** i
                in_channels = in_channels if i == 0 else channel_list[i - 1]
                out_channels = channel_list[i]
                self.net.add(
                    TemporalBlock(kernel_size, dilation, out_channels, in_channels=in_channels,
                                  dropout=dropout))

    def forward(self, inputs):
        return self.net(inputs)


class TCN(gluon.Block):
    def __init__(self, channel_list, in_channels, input_seq_len, output_dim, train_sequences=False,
                 prefix=None):
        """
        :param channel_list: A list of output channels for each TemporalBlock layer.
        :param in_channels: The number of input sequences.
        :param input_seq_len: Length of the input sequence.
        :param output_dim: The number of output sequences to predict.
        :param train_sequences: Whether to backprop the predictions for the entire sequence
                                or just the last prediction.
        """
        super(TCN, self).__init__(prefix=prefix)
        self.input_seq_len = input_seq_len
        self.output_dim = output_dim
        self.train_sequences = train_sequences
        self.channel_list = channel_list
        with self.name_scope():
            self.tcn = TemporalConvNet(channel_list, in_channels=in_channels)
            self.dense = gluon.nn.Dense(output_dim)

    def forward(self, inputs, exog=None):
        """
        :param inputs: shape (batch_size, seq_len, in_channels)
        :param exog: shape (batch_size, seq_len, exog_dim)
        :return: NDArray, shape (batch_size * num_predictions, output_dim)
        """
        # tcn_out: (batch_size, out_channels, seq_len)
        tcn_out = self.tcn.forward(inputs.transpose((0, 2, 1)))
        if not self.train_sequences:
            tcn_out = tcn_out[:, :, self.input_seq_len - 1:self.input_seq_len]
        combined = tcn_out
        if exog is not None:
            # exog_t: (batch_size, exog_dim, seq_len)
            exog_t = exog.transpose((0, 2, 1))
            # combined: (batch_size, seq_len, exog_dim + out_channels)
            combined = mx.nd.concat(combined, exog_t, dim=1)
        combined = combined.transpose((0, 2, 1))
        # flatten all predictions
        # preds: (batch_size * seq_len, out_channels)
        preds = combined.reshape((-1, combined.shape[2]))
        out = self.dense(preds)
        return out

    def predict_dynamic(self, predict_input, predict_seq_len, exog_input=None, ctx=mx.cpu()):
        pred_batch_size, input_seq_len, feature_dim = predict_input.shape

        # this buffer holds the input and output as we fill things in
        inp_buffer = mx.nd.zeros((pred_batch_size, input_seq_len + predict_seq_len, feature_dim),
                                 ctx=ctx)
        inp_buffer[:, :input_seq_len, :] = predict_input

        num_predictions = 1 if not self.train_sequences else input_seq_len
        inp = inp_buffer[:, :input_seq_len, :]
        for j in range(predict_seq_len):
            if exog_input is not None:
                output = self.forward(inp, exog=exog_input[:, j:j + num_predictions, :])
            else:
                output = self.forward(inp, exog=None)
            # for train_sequences, we need to grab every num_predictionth prediction
            output_idx = mx.nd.arange(0, output.shape[0], num_predictions, ctx=ctx)\
                         + num_predictions - 1
            last_outputs = output[output_idx, :]
            inp_buffer[:, input_seq_len + j:input_seq_len + j + 1, :] = \
                last_outputs.reshape((pred_batch_size, 1, -1))
            inp = inp_buffer[:, j + 1:input_seq_len + j + 1, :]
        outputs = inp_buffer[:, input_seq_len:, :]
        assert outputs.shape == (pred_batch_size, predict_seq_len, self.output_dim)
        return outputs

    def predict_batch(self, predict_input, predict_steps, ctx, pred_batch_size=5000):
        pred_outs = []
        for i in range(0, predict_input.shape[0], pred_batch_size):
            end = min([i + pred_batch_size, predict_input.shape[0]])
            predict_input_batch = predict_input[i:end, :]
            predict_input_batch = mx.nd.expand_dims(predict_input_batch, 2)
            pred_outputs = self.predict_dynamic(predict_input_batch, predict_steps, exog_input=None,
                                                ctx=ctx)
            pred_outs.append(pred_outputs)

        out_rows = sum([x.shape[0] for x in pred_outs])
        final_preds = mx.nd.zeros((out_rows, predict_steps), ctx=ctx)
        i = 0
        for arr in pred_outs:
            final_preds[i:i + arr.shape[0], :] = arr[:, :, 0]
            i += arr.shape[0]
        return final_preds

