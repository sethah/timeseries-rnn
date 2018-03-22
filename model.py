import mxnet.gluon as gluon
import mxnet as mx


# class Seq2Seq(gluon.Block):
#
#     def __init__(self, hidden_size, enc_num_layers, dec_num_layers,
#                  enc_dropout=0.0, dec_dropout=0.0):
#         super(Seq2Seq, self).__init__()
#         self.encoder = Encoder(hidden_size=hidden_size, num_layers=enc_num_layers,
#                                dropout=enc_dropout)
#         self.decoder = Decoder(hidden_size=hidden_size, num_layers=dec_num_layers,
#                                dropout=dec_dropout)
#
#     def forward(self, inputs, hidden, exog=None):
#         enc_out, dec_hidden = self.encoder.forward(inputs, hidden)
#
#     def begin_state(self, *args, **kwargs):
#         return self.encoder.begin_state(*args, **kwargs)


class Encoder(gluon.Block):
    """
    Encoder portion of Seq2Seq time series model
    """
    def __init__(self, hidden_size, num_layers, dropout=0.0):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = gluon.rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers,
                                  dropout=dropout, layout="NTC")

    def forward(self, inputs, hidden):
        """
        # :param inputs: shape (feature_dim, batch_size, input_seq_len)
        :param inputs: shape (batch_size, input_seq_len, feature_dim)
        :param hidden:
        :return:
        """
        output, hidden = self.rnn.forward(inputs, hidden)
        return output, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


class Decoder(gluon.Block):
    """
    Decoder portion of Seq2Seq model
    """
    def __init__(self, hidden_size, num_layers, output_dim, dropout=0.0):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = gluon.rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers,
                                  dropout=dropout, layout='NTC')
        self.out = gluon.nn.Dense(output_dim)
        self.bn1 = gluon.nn.BatchNorm()

    def forward(self, inputs, hidden, exog=None):
        """
        # :param inputs: shape (feature_dim, batch_size, input_seq_len)
        :param inputs: shape (batch_size, input_seq_len, feature_dim)
        :param hidden:
        :param exog: shape (batch_size, exog_dim)
        :return:
        """
        output, hidden = self.rnn.forward(inputs, hidden)
        output = output.reshape((-1, self.hidden_size))
        if exog is not None:
            assert (exog.shape[0] == output.shape[0])
            output = mx.nd.concat(output, exog, dim=1)
        output = self.bn1.forward(output)
        output = self.out.forward(output)
        return output, hidden

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

    def predict(self, inputs, hidden, exog=None, nsteps=1):
        dec_inp = inputs
        dec_hidden = hidden
        preds = []
        for i in range(nsteps):
            dec_out, dec_hidden = self.forward(dec_inp, dec_hidden, exog[:, i, :])
            preds.append(dec_out)
            dec_inp = mx.nd.expand_dims(dec_out, axis=1)#.transpose((1, 0, 2))

        return mx.nd.concatenate(preds, axis=1)

    @staticmethod
    def start_sequence(sos, feature_dim, batch_size):
        return mx.nd.ones((batch_size, 1, feature_dim)) * sos


class LSTMExogModel(gluon.Block):
    """
    LSTM model that concatenates exogenous feature to the output.
    """
    def __init__(self, hidden_size, num_layers, output_seq_len):
        super(LSTMExogModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = gluon.rnn.LSTM(hidden_size=hidden_size, num_layers=num_layers,
                                   dropout=0.4)
        self.out = gluon.nn.Dense(output_seq_len, activation=None)
        self.bn1 = gluon.nn.BatchNorm(axis=1)

    def forward(self, inputs, hidden, exog=None):
        """
        :param inputs: shape (feature_dim, batch_size, input_seq_len)
        :param hidden:
        :param exog: shape (batch_size, exog_dim)
        :return:
        """
        output, hidden = self.lstm.forward(inputs, hidden)
        output = output.reshape((-1, self.hidden_size))
        if exog is not None:
            assert (exog.shape[0] == output.shape[0])
            output = mx.nd.concat(output, exog, dim=1)
        output = self.bn1.forward(output)
        output = self.out.forward(output)
        return output

    def begin_state(self, *args, **kwargs):
        return self.lstm.begin_state(*args, **kwargs)

    def predict_dynamic(self, predict_input, predict_seq_len, exog_input=None):
        """
        Here we need to repeatedly append the predictions to the input sequence and
        keep on predicting.
        y1 = predict([x0, x1, x2])
        y2 = predict([x1, x2, y1])
        ...
        yn = predict([yn-3, yn-2, yn-1])

        :param predict_input: The input sequence to seed predictions with
                              shape (feature_dim, pred_batch_size, input_seq_len)
        :param exog_input: The exogenous input features, we need them for every
                           prediction step. Shape (exog_dim, pred_batch_size, predict_seq_len)
        :param predict_seq_len: The number of time steps ahead to predict.
        :return: Predictions, shape (output_dim, pred_batch_size, predict_seq_len)
        """

        feature_dim, pred_batch_size, input_seq_len = predict_input.shape
        # will fail if weights not yet initialized
        lstm_input_len = self.lstm.i2h_weight[0].data().shape[1]
        assert input_seq_len == lstm_input_len, "model expects input sequence length %d, but " \
                                                "got %d" % (lstm_input_len, input_seq_len)
        if exog_input is not None:
            exog_dim = exog_input.shape[0]
            assert exog_input.shape == (pred_batch_size, exog_dim, predict_seq_len)

        # this buffer holds the input and output as we fill things in
        inp_buffer = mx.nd.zeros((feature_dim, pred_batch_size, input_seq_len + predict_seq_len))
        inp_buffer[:, :, :input_seq_len] = predict_input[:, 0, :]

        inp = inp_buffer[:, :, :input_seq_len]
        hidden = self.begin_state(batch_size=pred_batch_size)
        for j in range(predict_seq_len):
            if exog_input is None:
                output = self.forward(inp, hidden, exog=None)
            else:
                output = self.forward(inp, hidden, exog_input[:, :, j].transpose((1, 0)))
            assert output.shape == (pred_batch_size, self.out.weight.shape[0])
            inp_buffer[0, :, input_seq_len + j:input_seq_len + j + 1] = output
            inp = inp_buffer[:, :, j + 1:input_seq_len + j + 1]
        outputs = inp_buffer[:, :, input_seq_len:]
        assert outputs.shape == (self.out.weight.shape[0], pred_batch_size, predict_seq_len)
        return outputs
