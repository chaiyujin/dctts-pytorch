import torch
import torch.nn as nn
import numpy as np
from pkg.networks import Text2Mel
from pkg.data import BatchMaker, load_data
from pkg.hyper import Hyper
from pkg.utils import PrettyBar, plot_spectrum, plot_attention


class MovingAverage(object):
    def __init__(self):
        self.sum_ = 0.0
        self.num_ = 0.0

    def add(self, x):
        self.sum_ += x
        self.num_ += 1

    def __call__(self):
        return "%4f" % (self.sum_ / self.num_)


def train(module):
    print("train:", "Text to Mel" if module=="Text2Mel" else "Mel Spectrum Super Resolution")
    graph = Text2Mel().to(Hyper.device)
    # set the training flag
    graph.train()
    # load data and get batch maker
    names, lengths, texts = load_data()
    batch_maker = BatchMaker(Hyper.batch_size, names, lengths, texts)

    criterion_mels = nn.L1Loss().to(Hyper.device)
    criterion_bd1 = nn.BCEWithLogitsLoss().to(Hyper.device)
    optimizer = torch.optim.Adam(
        graph.parameters(),
        lr=Hyper.adam_alpha,
        betas=Hyper.adam_betas,
        eps=Hyper.adam_eps
    )

    global_step = 0
    for loop_cnt in range(int(Hyper.num_batches / batch_maker.num_batches() + 0.5)):
        print("loop", loop_cnt)
        bar = PrettyBar(batch_maker.num_batches())
        bar.set_description("training...")
        loss_str0 = MovingAverage()
        loss_str1 = MovingAverage()
        for bi in bar:
            batch = batch_maker.next_batch()
            # make batch
            texts = torch.LongTensor(batch["texts"]).to(Hyper.device)
            # shift mel
            shift_mels = torch.FloatTensor(
                np.concatenate(
                    (np.zeros((batch["mels"].shape[0], batch["mels"].shape[1], 1)),
                     batch["mels"][:, :, :-1]),
                    axis=2)).to(Hyper.device)
            # ground truth
            mels = torch.FloatTensor(batch["mels"]).to(Hyper.device)
            # forward
            pred_logits, pred_mels = graph(texts, mels)
            # loss
            loss_mels = criterion_mels(pred_mels, mels)
            loss_bd1 = criterion_bd1(pred_logits, mels)
            loss = loss_mels + loss_bd1
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            loss_str0.add(loss_mels.cpu().data.mean())
            loss_str1.add(loss_bd1.cpu().data.mean())
            bar.set_description("loss_mels: {}, loss_bd1: {}".format(loss_str0(), loss_str1()))

            # plot
            plot_spectrum(mels[0].cpu().data, "mel_true", global_step)
            plot_spectrum(pred_mels[0].cpu().data, "mel_pred", global_step)
            plot_spectrum(graph.query[0].cpu().data, "query", global_step)
            plot_attention(graph.attention[0].cpu().data, "atten", global_step)

            # increase global step
            # global_step += 1
