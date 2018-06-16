import torch
import numpy as np
from pkg.hyper import Hyper
from pkg.modules import CharEmbed
from pkg.networks import TextEncoder, AudioEncoder, AudioDecoder


texts = torch.LongTensor(np.zeros([16, 180]))
mels = torch.FloatTensor(np.zeros([16, Hyper.dim_f, 500]))


text_encoder = TextEncoder()
text_encoder.print_shape([16, 180])
audio_encoder = AudioEncoder()
audio_encoder.print_shape([16, Hyper.dim_f, 500])
audio_decoder = AudioDecoder()
audio_decoder.print_shape([16, Hyper.dim_d * 2, 500])
texts = texts.to(Hyper.device)
mels = mels.to(Hyper.device)
text_encoder.to(Hyper.device)
audio_encoder.to(Hyper.device)

VK = text_encoder(texts)
Q = audio_encoder(mels)
# print(VK)
# print(Q)
