
class Hyper:
    # audio
    audio_preemph = 0.97
    audio_nfft = 2048
    audio_samplerate = 22050
    audio_winlen = 0.05
    audio_winstep = 0.0125
    audio_melfilters = 80
    audio_refdB = 20
    audio_maxdB = 100
    audio_power = 1.5  # Exponent for amplifying the predicted magnitude
    audio_niter = 50  # Number of inversion iterations
    temporal_rate = 4

    # data
    vocab = "P abcdefghijklmnopqrstuvwxyz'.?"  # P: padding
    data_dir = "/media/chaiyujin/FE6C78966C784B81/Linux/Dataset/LJSpeech-1.1"
    feat_dir = "/home/chaiyujin/Documents/Speech/dctts-pytorch/features/"

    # net
    dim_f = 80  # the dim of audio feature
    dim_e = 128
    dim_d = 256  # the hidden layer of Text2Mel
    # dropout
    dropout = 0.05
    # train
    device = "cuda:0"
    logdir = "/home/chaiyujin/Documents/Speech/dctts-pytorch/logdir"
