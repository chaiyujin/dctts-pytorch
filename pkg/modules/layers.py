import torch.nn as nn


class CharEmbed(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx):
        super(CharEmbed, self).__init__(num_embeddings, embedding_dim, padding_idx)

    def forward(self, inputs):
        output = super(CharEmbed, self).forward(inputs)
        return output.transpose(1, 2)
