import os
import codecs
import numpy as np
from pkg.hyper import Hyper
from pkg.utils import text_normalize


def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(Hyper.vocab)}
    idx2char = {idx: char for idx, char in enumerate(Hyper.vocab)}
    return char2idx, idx2char


def load_data():
    char2idx, _ = load_vocab()

    csv = os.path.join(Hyper.data_dir, "metadata.csv")
    names, lengths, texts = [], [], []
    with codecs.open(csv, 'r', "utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            fname, _, text = line.split('|')
            text = text_normalize(text) + 'E'  # append the end of string mark
            text = [char2idx[char] for char in text]

            names.append(fname)
            lengths.append(len(text))
            texts.append(text)
            if len(text) > Hyper.data_max_text_length:
                raise Exception("[load data]: length of text is out of range")

    return names, lengths, texts


def process_text(text, padding=False):
    char2idx, _ = load_vocab()
    text = text_normalize(text) + 'E'  # append the end of string mark
    text = [char2idx[char] for char in text]
    if padding:
        text = np.concatenate((text, np.zeros(Hyper.data_max_text_length - len(text))))
    return text


class BatchMaker(object):
    def __init__(self, batch_size, names, lengths, texts):
        '''
        Set a batch generator for training
        :param batch_size: training batch size
        :param names: the list of all wav file names
        :param lengths: the list of text length
        :param texts: the list of text
        '''

        self.bs_ = batch_size
        # load all data
        padi = Hyper.vocab.find('P')
        self.names_ = np.asarray(names)
        self.lengs_ = np.asarray(lengths, dtype=np.int32)
        self.total_ = len(names)
        max_len = np.max(self.lengs_)
        self.texts_ = []
        for text in texts:
            self.texts_.append(np.append(np.asarray(text, dtype=np.int32),
                                         np.full((max_len - len(text)), padi, dtype=np.int32)))
        self.texts_ = np.asarray(self.texts_, dtype=np.int32)
        # reset for loop
        self.idx_= 0
        self.indexes_ = None
        self.reshuffle_ = True

    def num_batches(self):
        return (self.total_ + self.bs_ - 1) // self.bs_

    def batch_size(self):
        return self.bs_

    def next_batch(self):
        # check if need re-shuffle
        if self.indexes_ is None or self.reshuffle_ == True:
            self.indexes_ = np.arange(0, self.total_)
            np.random.shuffle(self.indexes_)
            self.idx_ = 0
            self.reshuffle_ = False
        # load a batch
        indices = [self.indexes_[int(i % self.total_)] for i in range(self.idx_, self.idx_ + self.bs_)]
        self.idx_ += self.bs_
        if self.idx_ >= self.total_:
            self.reshuffle_ = True
        names = self.names_[indices]
        # texts
        text_lengths = list(self.lengs_[indices])
        maxlen_texts = np.max([d for d in text_lengths])
        texts = np.asarray(list(map(lambda x: x[:maxlen_texts], self.texts_[indices])))
        # feature
        mels = list(map(lambda x: np.load(os.path.join(Hyper.feat_dir, "mels/" + x + ".npy")), names))
        mags = list(map(lambda x: np.load(os.path.join(Hyper.feat_dir, "mags/" + x + ".npy")), names))
        mel_lengths = [d.shape[1] for d in mels]
        mag_lengths = [d.shape[1] for d in mags]
        maxlen_mels = np.max(mel_lengths)
        maxlen_mags = np.max(mag_lengths)
        mels = np.asarray(list(map(lambda x: np.pad(x, [[0, 0], [0, maxlen_mels - x.shape[1]]], mode="constant"), mels)))
        mags = np.asarray(list(map(lambda x: np.pad(x, [[0, 0], [0, maxlen_mags - x.shape[1]]], mode="constant"), mags)))
        # attention guide and mask
        guides = np.asarray(list(map(lambda x: np.load(os.path.join(Hyper.feat_dir, "guides/" + x + ".npy"))[:maxlen_texts, :maxlen_mels], names)))
        masks = np.asarray(list(map(lambda x: np.load(os.path.join(Hyper.feat_dir, "masks/" + x + ".npy"))[:maxlen_texts, :maxlen_mels], names)))
        return {
            "names": names,
            "texts": texts, "text_lengths": text_lengths,
            "mels": mels, "mel_lengths": mel_lengths,
            "mags": mags, "mag_lengths": mag_lengths,
            "atten_guides": guides,
            "atten_masks": masks
        }


if __name__ == "__main__":
    names, lengths, texts = load_data()
    batch_maker = BatchMaker(16, names, lengths, texts)
    batch = batch_maker.next_batch()
    print(batch["texts"].shape, batch["texts"].dtype)
    print(batch["mels"].shape, batch["mels"].dtype)
    print(batch["mags"].shape, batch["mags"].dtype)
    print(batch["atten_guides"].shape)
    print(batch["atten_masks"].shape)
