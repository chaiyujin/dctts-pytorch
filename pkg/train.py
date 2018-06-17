from __future__ import division

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

    def val(self):
        return float(self.sum_ / self.num_)


def guide_attention(text_lengths, mel_lengths):
    b = len(text_lengths)
    r = np.max(text_lengths)
    c = np.max(mel_lengths)
    guide = np.ones((b, r, c), dtype=np.float32)
    mask = np.zeros((b, r, c), dtype=np.float32)
    for i in range(b):
        W = guide[i]
        M = mask[i]
        N = float(text_lengths[i])
        T = float(mel_lengths[i])
        for n in range(r):
            for t in range(c):
                W[n][t] = 1.0 - np.exp(-(float(n) / N - float(t) / T) ** 2 / (2.0 * (Hyper.guide_g ** 2)))
                M[n][t] = 1.0
    return guide, mask


def train(module):
    print("train:", "Text to Mel" if module == "Text2Mel" else "Mel Spectrum Super Resolution")
    graph = Text2Mel().to(Hyper.device)
    # set the training flag
    graph.train()
    # load data and get batch maker
    names, lengths, texts = load_data()
    batch_maker = BatchMaker(Hyper.batch_size, names, lengths, texts)

    criterion_mels = nn.L1Loss().to(Hyper.device)
    criterion_bd1 = nn.BCEWithLogitsLoss().to(Hyper.device)
    criterion_atten = nn.L1Loss().to(Hyper.device)
    optimizer = torch.optim.Adam(
        graph.parameters(),
        lr=Hyper.adam_alpha,
        betas=Hyper.adam_betas,
        eps=Hyper.adam_eps
    )

    dynamic_guide = 10000.0
    global_step = 0
    for loop_cnt in range(int(Hyper.num_batches / batch_maker.num_batches() + 0.5)):
        print("loop", loop_cnt)
        bar = PrettyBar(batch_maker.num_batches())
        bar.set_description("training...")
        loss_str0 = MovingAverage()
        loss_str1 = MovingAverage()
        loss_str2 = MovingAverage()
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
            pred_logits, pred_mels = graph(texts, shift_mels)
            # loss
            loss_mels = sum(
                criterion_mels(torch.narrow(pred_mels[i], -1, 0, batch["mel_lengths"][i]),
                               torch.narrow(mels[i], -1, 0, batch["mel_lengths"][i]))
                for i in range(batch_maker.batch_size())) / float(batch_maker.batch_size())
            loss_bd1 = sum(
                criterion_bd1(torch.narrow(pred_logits[i], -1, 0, batch["mel_lengths"][i]),
                              torch.narrow(mels[i], -1, 0, batch["mel_lengths"][i]))
                for i in range(batch_maker.batch_size())) / float(batch_maker.batch_size())
            # guide attention
            atten_guide, atten_mask = guide_attention(batch["text_lengths"], batch["mel_lengths"])
            atten_guide = torch.FloatTensor(atten_guide).to(Hyper.device)
            atten_mask = torch.FloatTensor(atten_mask).to(Hyper.device)
            loss_atten = criterion_atten(
                atten_guide * graph.attention * atten_mask,
                torch.zeros_like(graph.attention)) * dynamic_guide
            loss = loss_mels + loss_bd1 + loss_atten

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            loss_str0.add(loss_mels.cpu().data.mean())
            loss_str1.add(loss_bd1.cpu().data.mean())
            loss_str2.add(loss_atten.cpu().data.mean())
            # adjust dynamic_guide
            # dynamic_guide = float((loss_mels + loss_bd1).cpu().data.mean() / loss_atten.cpu().data.mean())
            if global_step > 70:
                dynamic_guide = 100.0
            bar.set_description("loss_mels: {}, loss_bd1: {}, loss_atten: {}, scale: {}".format(loss_str0(), loss_str1(), loss_str2(), "%4f" % dynamic_guide))

            # plot
            plot_spectrum(mels[0].cpu().data, "mel_true", 0)
            plot_spectrum(pred_mels[0].cpu().data, "mel_pred", 0)
            plot_spectrum(graph.query[0].cpu().data, "query", 0)
            plot_attention(graph.attention[0].cpu().data[:, :batch["mel_lengths"][0]], "atten", 0)
            plot_attention(atten_guide[0].cpu().data[:, :batch["mel_lengths"][0]], "atten_guide", 0)

            # increase global step
            global_step += 1
