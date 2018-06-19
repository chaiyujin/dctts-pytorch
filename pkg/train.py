from __future__ import division

import os
import torch
import torch.nn as nn
import numpy as np
from shutil import copyfile
from pkg.networks import Text2Mel, SuperRes
from pkg.data import BatchMaker, load_data
from pkg.hyper import Hyper
from pkg.utils import PrettyBar, plot_spectrum, plot_attention, plot_loss


class LogHelper(object):
    def __init__(self, loss_name, logdir):
        self.val_ = []
        self.idx_ = []
        self.name_ = loss_name
        self.dir_ = os.path.join(logdir, "loss")
        if not os.path.exists(self.dir_):
            os.makedirs(self.dir_)

    def add(self, loss, step):
        self.val_.append(float(loss))
        self.idx_.append(int(step))

    def plot(self):
        if len(self.val_) == 0 or len(self.idx_) == 0:
            return
        path = os.path.join(self.dir_, "loss_{}-start_at_{}.png".format(self.name_, self.idx_[0]))
        plot_loss(self.val_, self.idx_, self.name_, path)


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


def train(module, load_trained):
    module = str(module)
    print("train:", "Text to Mel" if module == "Text2Mel" else "Mel Spectrum Super Resolution")
    if module == "Text2Mel":
        train_text2mel(load_trained)
    else:
        train_superres(load_trained)


def save(save_path, graph, criterion_dict, optimizer, global_step, is_best=False):
    state = {
        "global_step": global_step,
        "graph": graph.state_dict(),
        "optim": optimizer.state_dict()
    }
    for k in criterion_dict:
        state[k] = criterion_dict[k].state_dict()
    torch.save(state, save_path)
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path),
                                 "trained.pkg")
        copyfile(save_path, best_path)


def load(save_path, graph, criterion_dict=None, optimizer=None, device=None):
    state = torch.load(save_path, map_location=device)
    global_step = state["global_step"]
    graph.load_state_dict(state["graph"])
    if optimizer:
        optimizer.load_state_dict(state["optim"])
    if criterion_dict:
        for k in criterion_dict:
            criterion_dict[k].load_state_dict(state[k])
    return global_step


def train_text2mel(load_trained):
    # create log dir
    logdir = os.path.join(Hyper.logdir, "text2mel")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(os.path.join(logdir, "pkg")):
        os.mkdir(os.path.join(logdir, "pkg"))

    # device
    device = Hyper.device_text2mel

    graph = Text2Mel().to(device)
    # set the training flag
    graph.train()
    # load data and get batch maker
    names, lengths, texts = load_data()
    batch_maker = BatchMaker(Hyper.batch_size, names, lengths, texts)

    criterion_mels = nn.L1Loss().to(device)
    criterion_bd1 = nn.BCEWithLogitsLoss().to(device)
    criterion_atten = nn.L1Loss().to(device)
    optimizer = torch.optim.Adam(
        graph.parameters(),
        lr=Hyper.adam_alpha,
        betas=Hyper.adam_betas,
        eps=Hyper.adam_eps
    )

    lossplot_mels = LogHelper("mel_l1", logdir)
    lossplot_bd1 = LogHelper("mel_BCE", logdir)
    lossplot_atten = LogHelper("atten", logdir)

    dynamic_guide = float(Hyper.guide_weight)
    global_step = 0

    # check if load
    if load_trained > 0:
        print("load model trained for {}k batches".format(load_trained))
        global_step = load(os.path.join(logdir, "pkg/save_{}k.pkg".format(load_trained)),
                           graph, {"mels": criterion_mels, "bd1": criterion_bd1, "atten": criterion_atten}, optimizer)
        dynamic_guide *= Hyper.guide_decay ** (load_trained * 1000)

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
            texts = torch.LongTensor(batch["texts"]).to(device)
            # shift mel
            shift_mels = torch.FloatTensor(
                np.concatenate(
                    (np.zeros((batch["mels"].shape[0], batch["mels"].shape[1], 1)),
                     batch["mels"][:, :, :-1]),
                    axis=2)).to(device)
            # ground truth
            mels = torch.FloatTensor(batch["mels"]).to(device)

            # forward
            pred_logits, pred_mels = graph(texts, shift_mels)
            # loss
            if False:
                loss_mels = sum(
                    criterion_mels(torch.narrow(pred_mels[i], -1, 0, batch["mel_lengths"][i]),
                                   torch.narrow(mels[i], -1, 0, batch["mel_lengths"][i]))
                    for i in range(batch_maker.batch_size())) / float(batch_maker.batch_size())
                loss_bd1 = sum(
                    criterion_bd1(torch.narrow(pred_logits[i], -1, 0, batch["mel_lengths"][i]),
                                  torch.narrow(mels[i], -1, 0, batch["mel_lengths"][i]))
                    for i in range(batch_maker.batch_size())) / float(batch_maker.batch_size())
            else:
                loss_mels = criterion_mels(pred_mels, mels)
                loss_bd1 = criterion_bd1(pred_logits, mels)
            # guide attention
            atten_guide = torch.FloatTensor(batch["atten_guides"]).to(device)
            atten_mask = torch.FloatTensor(batch["atten_masks"]).to(device)
            atten_mask = torch.ones_like(graph.attention)
            loss_atten = criterion_atten(
                atten_guide * graph.attention * atten_mask,
                torch.zeros_like(graph.attention)) * dynamic_guide
            loss = loss_mels + loss_bd1 + loss_atten

            # backward
            graph.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            # clip grad
            nn.utils.clip_grad_value_(graph.parameters(), 1)
            optimizer.step()
            # log
            loss_str0.add(loss_mels.cpu().data.mean())
            loss_str1.add(loss_bd1.cpu().data.mean())
            loss_str2.add(loss_atten.cpu().data.mean())
            lossplot_mels.add(loss_str0(), global_step)
            lossplot_bd1.add(loss_str1(), global_step)
            lossplot_atten.add(loss_str2(), global_step)

            # adjust dynamic_guide
            # dynamic_guide = float((loss_mels + loss_bd1).cpu().data.mean() / loss_atten.cpu().data.mean())
            dynamic_guide *= Hyper.guide_decay
            if dynamic_guide < Hyper.guide_lowbound:
                dynamic_guide = Hyper.guide_lowbound
            bar.set_description("gs: {}, mels: {}, bd1: {}, atten: {}, scale: {}".format(global_step, loss_str0(), loss_str1(), loss_str2(), "%4f" % dynamic_guide))

            # plot
            if global_step % 100 == 0:
                gs = 0
                plot_spectrum(mels[0].cpu().data, "mel_true", gs, dir=logdir)
                plot_spectrum(shift_mels[0].cpu().data, "mel_input", gs, dir=logdir)
                plot_spectrum(pred_mels[0].cpu().data, "mel_pred", gs, dir=logdir)
                plot_spectrum(graph.query[0].cpu().data, "query", gs, dir=logdir)
                plot_attention(graph.attention[0].cpu().data, "atten", gs, True, dir=logdir)
                plot_attention((atten_guide)[0].cpu().data, "atten_guide", gs, True, dir=logdir)
                if global_step % 500 == 0:
                    lossplot_mels.plot()
                    lossplot_bd1.plot()
                    lossplot_atten.plot()

                if global_step % 10000 == 0:
                    save(os.path.join(logdir, "pkg/save_{}k.pkg").format(global_step // 1000),
                         graph,
                         {"mels": criterion_mels, "bd1": criterion_bd1, "atten": criterion_atten},
                         optimizer,
                         global_step,
                         True)

            # increase global step
            global_step += 1


def train_superres(load_trained):
    # logdir
    logdir = os.path.join(Hyper.logdir, "superres")
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    if not os.path.exists(os.path.join(logdir, "pkg")):
        os.mkdir(os.path.join(logdir, "pkg"))
    # device
    device = Hyper.device_superres
    # graph
    graph = SuperRes().to(device)
    graph.train()
    # load data
    names, lengths, texts = load_data()
    batch_maker = BatchMaker(Hyper.batch_size, names, lengths, texts)
    # loss
    criterion_mags = nn.L1Loss().to(device)
    criterion_bd2 = nn.BCEWithLogitsLoss().to(device)
    lossplot_mags = LogHelper("mag_l1", logdir)
    lossplot_bd2 = LogHelper("mag_BCE", logdir)
    # optim
    optimizer = torch.optim.Adam(
        graph.parameters(),
        lr=Hyper.adam_alpha,
        betas=Hyper.adam_betas,
        eps=Hyper.adam_eps
    )
    # load 
    global_step = 0
    if load_trained > 0:
        print("load model trained for {}k batches".format(load_trained))
        global_step = load(os.path.join(logdir, "pkg/save_{}k.pkg".format(load_trained)),
                           graph, {"mags": criterion_mags, "bd2": criterion_bd2}, optimizer)

    for loop_cnt in range(int(Hyper.num_batches / batch_maker.num_batches() + 0.5)):
        print("loop", loop_cnt)
        bar = PrettyBar(batch_maker.num_batches())
        bar.set_description("training...")
        loss_str0 = MovingAverage()
        loss_str1 = MovingAverage()

        for bi in bar:
            batch = batch_maker.next_batch()
            # low res
            mels = torch.FloatTensor(batch["mels"]).to(device)
            # high res
            mags = torch.FloatTensor(batch["mags"]).to(device)

            # forward
            mag_logits, mag_pred = graph(mels)

            # loss
            loss_mags = criterion_mags(mag_pred, mags)
            loss_bd2 = criterion_bd2(mag_logits, mags)
            loss = loss_mags + loss_bd2

            # backward
            graph.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            # clip grad
            nn.utils.clip_grad_value_(graph.parameters(), 1)
            optimizer.step()

            # log
            loss_str0.add(loss_mags.cpu().data.mean())
            loss_str1.add(loss_bd2.cpu().data.mean())
            lossplot_mags.add(loss_str0(), global_step)
            lossplot_bd2.add(loss_str1(), global_step)
            bar.set_description("gs: {}, mags: {}, bd2: {}".format(global_step, loss_str0(), loss_str1()))

            # plot
            if global_step % 100 == 0:
                gs = 0
                plot_spectrum(mag_pred[0].cpu().data, "pred", gs, dir=logdir)
                plot_spectrum(mags[0].cpu().data, "true", gs, dir=logdir)
                plot_spectrum(mels[0].cpu().data, "input", gs, dir=logdir)
                if global_step % 100 == 0:
                    lossplot_mags.plot()
                    lossplot_bd2.plot()

                if global_step % 10000 == 0:
                    save(os.path.join(logdir, "pkg/save_{}k.pkg").format(global_step // 1000),
                         graph,
                         {"mags": criterion_mags, "bd2": criterion_bd2},
                         optimizer,
                         global_step,
                         True)

            global_step += 1
