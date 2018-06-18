from __future__ import division
import time
import math
import os, copy
import re
import unicodedata
import numpy as np
import librosa
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pkg.hyper import Hyper


def get_spectrum(wav_path):
    '''
    :param wav_path: the path of wav file
    :return:
        mel: mel spectrum (n_mels, T) float32 numpy array
        mag: magnitude spectrum (nfft/2 + 1, T) float32 numpy array
    '''
    audio, rate = librosa.load(wav_path, sr=Hyper.audio_samplerate)
    audio, _ = librosa.effects.trim(audio)
    # pre-emphasis
    audio = np.append(audio[0], audio[1:] - Hyper.audio_preemph * audio[:-1])
    # stft
    spec = librosa.stft(y=audio,
                        n_fft=int(Hyper.audio_nfft),
                        hop_length=int(Hyper.audio_winstep * Hyper.audio_samplerate),
                        win_length=int(Hyper.audio_winlen * Hyper.audio_samplerate))
    mag = np.absolute(spec)
    mel_filters = librosa.filters.mel(Hyper.audio_samplerate, Hyper.audio_nfft, Hyper.audio_melfilters)
    mel = np.dot(mel_filters, mag)

    # to dB
    mag[mag < 1e-10] = 1e-10
    mel[mel < 1e-10] = 1e-10
    mel = 20 * np.log10(mel)
    mag = 20 * np.log10(mag)

    # normalize
    mel = np.clip((mel - Hyper.audio_refdB + Hyper.audio_maxdB) / Hyper.audio_maxdB, 1e-8, 1)
    mag = np.clip((mag - Hyper.audio_refdB + Hyper.audio_maxdB) / Hyper.audio_maxdB, 1e-8, 1)

    return mel, mag


def spectrogram2wav(mag):
    '''# Generate wave file from linear magnitude spectrogram

    Args:
      mag: A numpy array of (T, 1+n_fft//2)

    Returns:
      wav: A 1-D numpy array.
    '''

    # de-noramlize
    mag = (np.clip(mag, 0, 1) * Hyper.audio_maxdB) - Hyper.audio_maxdB + Hyper.audio_refdB

    # to amplitude
    mag = np.power(10.0, mag * 0.05)

    # wav reconstruction
    wav = griffin_lim(mag**Hyper.audio_power)

    # de-preemphasis
    wav = signal.lfilter([1], [1, -Hyper.audio_preemph], wav)

    # trim
    wav, _ = librosa.effects.trim(wav)

    return wav.astype(np.float32)


def griffin_lim(spectrogram):
    '''Applies Griffin-Lim's raw.'''
    X_best = copy.deepcopy(spectrogram)
    for i in range(Hyper.audio_niter):
        X_t = invert_spectrogram(X_best)
        est = librosa.stft(X_t, Hyper.audio_nfft,
                           hop_length=int(Hyper.audio_winstep * Hyper.audio_samplerate),
                           win_length=int(Hyper.audio_winlen * Hyper.audio_samplerate))
        phase = est / np.maximum(1e-8, np.abs(est))
        X_best = spectrogram * phase
    X_t = invert_spectrogram(X_best)
    y = np.real(X_t)

    return y


def invert_spectrogram(spectrogram):
    '''Applies inverse fft.
    Args:
      spectrogram: [1+n_fft//2, t]
    '''
    return librosa.istft(spectrogram,
                         hop_length=int(Hyper.audio_winstep * Hyper.audio_samplerate),
                         win_length=int(Hyper.audio_winlen * Hyper.audio_samplerate),
                         window="hann")


def plot_spectrum(spectrum, name, gs, colorbar=False, dir=Hyper.logdir):
    if not os.path.exists(dir):
        os.mkdir(dir)

    fig, ax = plt.subplots()
    im = ax.imshow(np.flip(spectrum, 0), cmap="jet", aspect=0.2 * spectrum.shape[1] / spectrum.shape[0])
    if colorbar:
        fig.colorbar(im)
    plt.title('{} Steps'.format(gs))
    plt.savefig('{}/{}_{}.png'.format(dir, name, gs), format='png')
    plt.close(fig)


def plot_attention(attention, name, gs, colorbar=False, dir=Hyper.logdir):
    """Plots the alignment.

    Args:
      alignment: A numpy array with shape of (encoder_steps, decoder_steps)
      gs: (int) global step.
      dir: Output path.
    """
    if not os.path.exists(dir):
        os.mkdir(dir)

    fig, ax = plt.subplots()
    im = ax.imshow(attention)
    if colorbar:
        fig.colorbar(im)
    plt.title('{} Steps'.format(gs))
    plt.savefig('{}/{}_{}.png'.format(dir, name, gs), format='png')
    plt.close(fig)


def plot_loss(loss, idx, name, path):
    fig, ax = plt.subplots()
    plt.title(name)
    plt.plot(idx, loss)
    plt.savefig(path, format="png")
    plt.close(fig)


def find_files(path, target_ext=None):
    if target_ext is not None:
        if not isinstance(target_ext, list):
            target_ext = [target_ext]
        for i in range(len(target_ext)):
            if target_ext[i][0] != '.':
                target_ext[i] = '.' + target_ext[i]
    result_list = []
    for parent, dirs, files in os.walk(path):
        for file in files:
            if file[0] == '.' and file[1] == '_':
                continue
            if target_ext is not None:
                the_path = os.path.join(parent, file).replace('\\', '/')
                name, ext = os.path.splitext(the_path)
                if ext in target_ext:
                    result_list.append(name + ext)
            else:
                the_path = os.path.join(parent, file).replace('\\', '/')
                result_list.append(the_path)
    return result_list


def guide_attention(text_lengths, mel_lengths, r=None, c=None):
    b = len(text_lengths)
    if r is None:
        r = np.max(text_lengths)
    if c is None:
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
                if n < N and t < T:
                    W[n][t] = 1.0 - np.exp(-(float(n) / N - float(t) / T) ** 2 / (2.0 * (Hyper.guide_g ** 2)))
                    M[n][t] = 1.0
                elif t >= T and n < N:
                    W[n][t] = 1.0 - np.exp(-((float(n - N - 1) / N)** 2 / (2.0 * (Hyper.guide_g ** 2))))
    return guide, mask


def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents

    text = text.lower()
    text = re.sub("[\"\-()[\]“”]", " ", text)
    text = re.sub("[,;:!]", ".", text)
    text = re.sub("[’]", "'", text)
    text = re.sub("[^{}]".format(Hyper.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    text = text.strip()
    if text[-1] >= 'a' and text[-1] <= 'z':
        text += '.'
    return text


class PrettyBar:
    grid_list = ['\u2596', '\u2598', '\u259D', '\u2597']

    def __init__(self, low, high=None, step=1):
        if high is None:
            high = low
            low = 0
        if step == 0:
            high = low
        self.sign = -1 if step < 0 else 1
        self.current = low
        self.low = low
        self.high = high
        self.total = int(math.ceil((high - low) / step))
        self.step = step
        self.percent = 0
        self.eta = -1
        # tick
        self.first_tick = -1
        self.last_tick = -1
        self.per = -1
        self.desc = 'in progress'
        self.block_idx = 0
        self._len = 0
        self._block_tick = -1

    def __iter__(self):
        return self

    def __next__(self):
        if self.total <= 0:
            raise StopIteration
        if self.current * self.sign >= self.high * self.sign:
            self.progress_block(self.current, max(self.high, self.low), self.desc, suffix='eta ' + self.__time_to_str(0), end=True)
            print("Total time:", self.__time_to_str((time.time() - self.first_tick) * 1000))
            raise StopIteration
        else:
            iter = int((self.current - self.low) / self.step)
            # eta
            if self.first_tick < 0:
                eta = -1
                self.first_tick = time.time()
                self.last_tick = self.first_tick
            else:
                cur_tick = time.time()
                dura_per_iter = (cur_tick - self.first_tick) * 1000 / iter
                if self.per < 0:
                    self.per = dura_per_iter
                else:
                    self.per = 0.5 * self.per + (0.5) * dura_per_iter
                eta = self.per * (self.total - iter)
                self.last_tick = cur_tick
            self.eta = eta
            self.percent = ("{0:." + str(1) + "f}").format(
                    100 * (iter / float(self.total)))
            self.progress_block(self.current, max(self.high, self.low), self.percent, self.desc, suffix='eta ' + self.__time_to_str(self.eta))
            self.current += self.step
            return self.current - self.step

    def progress_block(self, iteration, total, percent, prefix='',
                       suffix='', end=False):
        # calc block idx
        if (time.time() - self._block_tick) > 0.2:
            self._block_tick = time.time()
            self.block_idx += self.sign
        print_str = '%s[%d/%d] |%s| [%s%% %s]' % (PrettyBar.grid_list[int(self.block_idx%len(PrettyBar.grid_list))], iteration, total, prefix, percent, suffix)
        if len(print_str) < self._len:
            print("\r%s" % (' ' * self._len), end='')
        self._len = len(print_str)
        print('\r%s' % (print_str), end='')
        if (end):
            print("\r%s" % (' ' * self._len), end='\r')

    def set_description(self, desc):
        self.desc = desc
        self.progress_block(self.current, max(self.high, self.low), self.percent, self.desc, suffix='eta ' + self.__time_to_str(self.eta))

    def __time_to_str(self, t):
        t = int(t)
        if t < 0:
            return 'ETA unknown'
        sec = int(t / 1000)
        ms = t % 1000
        min = int(sec / 60)
        sec = sec % 60
        h = int(min / 60)
        min = min % 60
        if h > 99:
            return '99:' + str(min).zfill(2) + ':' + str(sec).zfill(2)# + ':' + str(ms).zfill(3)
        else:
            return '' + str(h).zfill(2) + ':' + str(min).zfill(2) + ':' + str(sec).zfill(2)# + ':' + str(ms).zfill(3)


if __name__ == '__main__':
    bar = PrettyBar(100)
    print(bar.total)
    for i in bar:
        time.sleep(0.2)