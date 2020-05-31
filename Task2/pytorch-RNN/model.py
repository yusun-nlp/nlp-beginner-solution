import torch
from torch import nn
import torch.nn.functional as F


class TextRNN(nn.Module):
    """文本分类，RNN模型"""

    def __init__(self):
        super(TextRNN, self).__init__()
        # embedding layer, output tensor is (batch, seq, feature)
        self.embedding = nn.Embedding(5000, 64)  # word embedding
        # LSTM network
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, bidirectional=True)
        # full connected layer
        self.f1 = nn.Sequential(nn.Linear(256, 10), nn.Softmax())

    def forward(self, x):
        # batch_size x text_len x embedding_size 64*600*64
        x = self.embedding(x)
        # text_len x batch_size x embedding_size 600*64*64
        x = x.permute(1, 0, 2)
        x, (h_n, c_n) = self.rnn(x)
        final_feature_map = F.dropout(h_n, 0.8)
        # 64*256 Batch_size * (hidden_size * hidden_layers * 2)
        feature_map = torch.cat([final_feature_map[i, :, :] for i in range(final_feature_map.shape[0])], dim=1)
        final_out = self.f1(feature_map)  # 64*10 batch_size * class_num
        return final_out
