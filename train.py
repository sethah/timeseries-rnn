import mxnet as mx
import mxnet.gluon as gluon

import numpy as np


def maybe_print_summary(epoch, log_interval, loss, total_samples, label='train'):
    if (epoch + 1) % log_interval == 0:
        cur_loss = loss / total_samples
        print("[Epoch %d] %s loss = %0.3f" % (epoch + 1, label, cur_loss))


def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [i.detach() for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden


def train_epochs(model, loaders, loss, trainer, num_epochs, log_interval):
    batch_size = loaders['train']._batch_sampler._batch_size
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_samples = 0
        hidden = model.begin_state(batch_size=batch_size)
        for data, exog, target in loaders['train']:
            bsize, out_seq_len, out_dim = target.shape
            # assert data.shape == (batch_size, input_seq_len, feature_dim)
            exog_dim = exog.shape[2]
            assert exog.shape == (batch_size, out_seq_len, exog_dim)
            data = data
            hidden = detach(hidden)
            with mx.autograd.record():
                # assum only one prediction step
                output = model.forward(data, hidden, exog=exog.reshape((bsize, exog_dim)))
                L = loss(output, target.reshape((batch_size, -1)))
                L.backward()
            trainer.step(batch_size)
            total_loss += mx.nd.sum(L).asscalar()
            total_samples += target.shape[0]

        test_loss = 0.0
        test_samples = 0
        for data, exog, target in loaders['valid']:
            bsize, out_seq_len, out_dim = target.shape
            exog_dim = exog.shape[2]
            # data = data.transpose((2, 0, 1))
            hidden = detach(hidden)
            output = model.forward(data, hidden, exog=exog.reshape((batch_size, exog_dim)))
            # output = model.forward(data, hidden)
            L = loss(output, target.reshape((batch_size, -1)))
            test_loss += mx.nd.sum(L).asscalar()
            test_samples += target.shape[0]
        maybe_print_summary(epoch, log_interval, total_loss, total_samples)
        maybe_print_summary(epoch, log_interval, test_loss, test_samples, label='valid')
        print(model.out.weight.data()[0, 32].asscalar())


def train_epochs_tcn(model, loaders, loss, trainer, num_epochs, log_interval, ctx,
                     valid_func=None, valid_interval=1):
    batch_size = loaders['train']._batch_sampler._batch_size
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_samples = 0
        for batch in loaders['train']:
            data, target = batch
            if isinstance(data, tuple) or isinstance(data, list):
                datas = zip(*[gluon.utils.split_and_load(d, ctx) for d in data])
            else:
                datas = gluon.utils.split_and_load(data, ctx)
            targets = gluon.utils.split_and_load(target, ctx)
            with mx.autograd.record():
                outs = [model.forward(data) for data, exog in datas]
                losses = [loss(output, target.reshape((target.shape[0] * target.shape[1], -1)))
                          for output, target in zip(outs, targets)]
                for l in losses:
                    l.backward()
            trainer.step(batch_size)
            for target, l in zip(targets, losses):
                total_loss += mx.nd.sum(l).asscalar()
                total_samples += target.shape[0]

        maybe_print_summary(epoch, log_interval, total_loss, total_samples)
        if (epoch + 1) % valid_interval == 0 and valid_func:
            val_loss = valid_func(model)
            print("[Epoch %d] valid loss: %0.3f" % (epoch + 1, val_loss))


def train_batch_seq2seq(inp, target, encoder, decoder, enc_opt,
                        dec_opt, criterion, exog=None, teacher_forcing_prob=0.5):
    batch_size, output_seq_len, output_dim = target.shape
    enc_hidden = encoder.begin_state(batch_size=batch_size)

    feature_dim = inp.shape[2]
    assert feature_dim == output_dim, "Input feature dimension doesn't" \
                                      " match output feature dimension"

    with mx.autograd.record():
        batch_loss = 0.0
        loss_sum = mx.nd.zeros((1,))
        # enc_out, dec_hidden = encoder.forward(inp.transpose((2, 0, 1)), enc_hidden)
        enc_out, dec_hidden = encoder.forward(inp, enc_hidden)
        dec_inp = mx.nd.zeros(batch_size * output_dim * 1).reshape((batch_size, 1, output_dim))

        use_teacher_forcing = np.random.random() < teacher_forcing_prob
        if use_teacher_forcing:
            for di in range(output_seq_len):
                dec_out, dec_hidden = decoder.forward(dec_inp, dec_hidden, exog[:, di, :])
                loss = criterion(dec_out, target[:, di, :].reshape((batch_size, -1)))
                # target is (batch_size, out_seq_len, feature_dim)
                dec_inp = target[:, di:di + 1, :]#.transpose((2, 0, 1))
                loss_sum = mx.nd.add(loss_sum, loss)
        else:
            for di in range(output_seq_len):
                dec_out, dec_hidden = decoder.forward(dec_inp, dec_hidden, exog[:, di, :])
                # dec_out is shape (batch_size, feature_dim)
                loss = criterion(dec_out, target[:, di, :].reshape((batch_size, -1)))
                loss_sum = mx.nd.add(loss_sum, loss)
                dec_inp = mx.nd.expand_dims(dec_out, axis=1)
                # should be (output_dim, batch_size, out_seq_len)
                # dec_inp = dec_inp.transpose((1, 0, 2))
        loss_sum.backward()
        batch_loss += mx.nd.sum(loss_sum).asscalar()
    enc_opt.step(batch_size)
    dec_opt.step(batch_size)
    return batch_loss


def train_seq2seq(num_epochs, encoder, decoder, loader, print_every=10, **kwargs):

    optimizer = kwargs.get('opt', 'adam')
    opt_params = kwargs.get('opt_params', {})
    enc_opt = gluon.Trainer(encoder.collect_params(), optimizer, opt_params)
    dec_opt = gluon.Trainer(decoder.collect_params(), optimizer, opt_params)
    criterion = gluon.loss.L2Loss()

    teacher_forcing_prob = kwargs.get('teacher_forcing_prob', 0.5)
    loss_total = 0.0
    total_samples = 0
    for epoch in range(num_epochs):
        for inputs, exog, target in loader:
            batch_size = inputs.shape[0]

            if exog is not None:
                exog_dim = exog.shape[2]
                output_seq_len = target.shape[1]
                assert exog.shape == (batch_size, output_seq_len, exog_dim)
                batch_loss = train_batch_seq2seq(inputs, target, encoder, decoder, enc_opt, dec_opt,
                                                 criterion, exog,
                                                 teacher_forcing_prob=teacher_forcing_prob)
            else:
                batch_loss = train_batch_seq2seq(inputs, target, encoder, decoder, enc_opt, dec_opt,
                                                 criterion,
                                                 teacher_forcing_prob=teacher_forcing_prob)
            loss_total += batch_loss
            total_samples += target.shape[0]
        if epoch % print_every == print_every - 1:
            print("[%d/%d] Avg. loss per epoch: %0.3f" % (epoch + 1, num_epochs,
                                                          loss_total / total_samples))
            loss_total = 0.0
            total_samples = 0
