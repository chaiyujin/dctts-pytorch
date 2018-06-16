import os
import re
import codecs
import unicodedata
import numpy as np
from pkg.hyper import Hyper


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
        mels = list(map(lambda x: np.load(os.path.join(Hyper.feat_dir, "mels/" + x + ".npy")), self.names_[indices]))
        mags = list(map(lambda x: np.load(os.path.join(Hyper.feat_dir, "mags/" + x + ".npy")), self.names_[indices]))
        maxlen_mels = np.max([d.shape[1] for d in mels])
        maxlen_mags = np.max([d.shape[1] for d in mags])
        mels = np.asarray(list(map(lambda x: np.pad(x, [[0, 0], [0, maxlen_mels - x.shape[1]]], mode="constant"), mels)))
        mags = np.asarray(list(map(lambda x: np.pad(x, [[0, 0], [0, maxlen_mags - x.shape[1]]], mode="constant"), mags)))
        return {
            "names": self.names_[indices],
            "texts": self.texts_[indices],
            "mels": mels,
            "mags": mags
        }


if __name__ == "__main__":
    names, lengths, texts = load_data()
    batch_maker = BatchMaker(16, names, lengths, texts)
    batch = batch_maker.next_batch()
    print(batch["mels"].shape, batch["mels"].dtype)
    print(batch["mags"].shape, batch["mags"].dtype)
