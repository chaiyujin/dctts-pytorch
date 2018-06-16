import os
import numpy as np
from pkg.utils import get_spectrum, spectrogram2wav, plot_spectrum, PrettyBar, find_files
from pkg.hyper import Hyper
import scipy.io.wavfile as wavfile


def process_file(path):
    fname, ext = os.path.splitext(os.path.basename(path))
    if ext != ".wav":
        raise Exception("[preprocess]: only support wav file")

    mel, mag = get_spectrum(path)
    t = mel.shape[1]

    # right padding, reduce shape
    pad = Hyper.temporal_rate - (t % Hyper.temporal_rate) if t % Hyper.temporal_rate != 0 else 0
    mel = np.pad(mel, [[0, 0], [0, pad]], mode="constant")
    mag = np.pad(mag, [[0, 0], [0, pad]], mode="constant")

    # temporal reduction
    mel = mel[..., ::Hyper.temporal_rate]

    mel_path = os.path.join(Hyper.feat_dir, "mels")
    mag_path = os.path.join(Hyper.feat_dir, "mags")
    if not os.path.exists(mel_path):
        os.makedirs(mel_path)
    if not os.path.exists(mag_path):
        os.makedirs(mag_path)
    np.save(os.path.join(mel_path, fname + ".npy"), mel.astype(np.float32))
    np.save(os.path.join(mag_path, fname + ".npy"), mag.astype(np.float32))


def preprocess():
    print("pre-processing...")
    flist = find_files(Hyper.data_dir, "wav")
    bar = PrettyBar(len(flist))
    for i in bar:
        fpath = flist[i]
        fname = os.path.basename(fpath)
        bar.set_description(fname)
        process_file(os.path.join(Hyper.data_dir, fpath))


if __name__ == "__main__":
    preprocess()
