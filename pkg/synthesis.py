import os
import torch
import numpy as np
import scipy.io.wavfile as wavfile
from pkg.train import load
from pkg.networks import Text2Mel, SuperRes
from pkg.utils import plot_spectrum, plot_attention, PrettyBar, spectrogram2wav
from pkg.hyper import Hyper
from pkg.data import process_text


def synthesis(text_list, plot=True):
    device = "cuda:0"
    # load graphs
    graph0 = Text2Mel()
    graph1 = SuperRes()
    load(os.path.join(Hyper.logdir, "text2mel/pkg/trained.pkg"), graph0, device='cpu')
    load(os.path.join(Hyper.logdir, "superres/pkg/trained.pkg"), graph1, device='cpu')
    graph0.eval()
    graph1.eval()
    # make dir
    syn_dir = os.path.join(Hyper.root_dir, "synthesis")
    if not os.path.exists(syn_dir):
        os.makedirs(syn_dir)

    # phase1: text to mel
    graph0.to(device)
    texts = [process_text(text, padding=True) for text in text_list]
    texts = torch.LongTensor(np.asarray(texts)).to(device)
    mels = torch.FloatTensor(np.zeros((len(texts), Hyper.dim_f, Hyper.data_max_mel_length))).to(device)
    prev_atten = None
    bar = PrettyBar(Hyper.data_max_mel_length - 1)
    bar.set_description("Text to Mel")
    for t in bar:
        _, new_mel = graph0(texts, mels, None if t == 0 else t - 1, prev_atten)
        mels[:, :, t + 1].data.copy_(new_mel[:, :, t].data)
        prev_atten = graph0.attention
    for i in range(len(text_list)):
        # mels[:, :, :-1].data.copy_(mels[:, :, 1:].data)
        if plot:
            plot_attention(graph0.attention[i].cpu().data, "atten", i, True, syn_dir)
            plot_spectrum(mels[i].cpu().data, "mels", i, True, syn_dir)
    del graph0

    # phase2: super resolution
    graph1.to(device)
    _, mags = graph1(mels)
    bar = PrettyBar(len(text_list))
    bar.set_description("Super Resolution")
    for i in bar:
        wav = spectrogram2wav(mags[i].cpu().data.numpy())
        wavfile.write(os.path.join(syn_dir, "syn_{}.wav".format(i)), Hyper.audio_samplerate, wav)
        if plot:
            plot_spectrum(mags[i].cpu().data, "mags", i, True, syn_dir)
    del graph1


if __name__ == "__main__":
    synthesis(
        ["in being comparatively modern.",
         "The birch canoe slid on the smooth planks",
         "I can't believe you any more.",
         "This is bullshit.",
         "Give me 10101, because it's .123 times better than h110..."])
