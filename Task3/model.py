from torch.utils.data import Dataset
import torch
import torch.nn as nn
from layers import *


def _init_esim_weight(module):
    """
    Initialise the weights of the ESIM model
    :param module: ESIM model
    :return:
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0]
        module.bias_hh_l0.data[hidden_size:(2 * hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2 * hidden_size)] = 1.0


def get_mask(sequences_batch, sequences_lengths):
    """
    Get the mask for a batch of padded variable length sequences
    :param sequences_batch: a batch of padded variable length sequences containing word indices
    :param sequences_lengths: a tensor containing the lengths of the sequence in "sequences_batch"
    :return: a mask of size (batch, max_sequence_length)
    """
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    return mask


class SnliDataSet(Dataset):
    def __init__(self, data, max_premises_len=None, max_hypothesis_len=None):
        self.seq_len = len(data["premises"])

        # set size of tensor
        self.premises_len = [len(seq) for seq in data["premises"]]
        self.max_premises_len = max_premises_len
        if self.max_premises_len is None:
            self.max_premises_len = max(self.premises_len)

        self.hypothesis_len = [len(seq) for seq in data["hypothesis"]]
        self.max_hypothesis_len = max_hypothesis_len
        if max_hypothesis_len is None:
            self.max_hypothesis_len = max(self.hypothesis_len)
        print(self.seq_len, self.max_premises_len)
        print(self.seq_len, self.max_hypothesis_len)

        # set as tensor
        self.data = {
            "premises": torch.zeros((self.seq_len, self.max_premises_len), dtype=torch.long),
            "hypothesis": torch.zeros((self.seq_len, self.max_hypothesis_len), dtype=torch.long),
            "labels": torch.tensor(data["labels"])
        }

        for i, premises in enumerate(data["premises"]):
            l = len(data["premises"][i])
            self.data["premises"][i][:l] = torch.tensor(data["premises"][i][:l])
            l2 = len(data["hypothesis"][i])
            self.data["hypothesis"][i][:l2] = torch.tensor(data["hypothesis"][i][:l2])

    def __len__(self):
        return self.seq_len

    def __getitem__(self, index):
        return {"premises": self.data["premises"][index],
                "premises_len": min(self.premises_len[index], self.max_premises_len),
                "hypothesis": self.data["hypothesis"][index],
                "hypothesis_len": min(self.hypothesis_len[index], self.max_hypothesis_len),
                "labels": self.data["labels"][index]}


class ESIM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, embeddings=None, padding_idx=0,
                 dropout=0.5, num_classes=3, device="cpu"):
        """

        :param vocab_size: size of the vocabulary of embeddings in the model
        :param embedding_dim: dimension of the word embeddings
        :param hidden_size: size of all hidden layers in the network
        :param embeddings: a tensor of size (vocab_size, embedding_dim) containing pretrained word
                           embeddings.
        :param padding_idx: the index of the padding token in the premises and hypothesis passed as
                            input to the model.
        :param dropout: the dropout rate to use between the layers of the network
        :param num_classes: number of classes in the output of the network
        :param device: name of device
        """
        super(ESIM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        # set layers
        self._word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx, _weight=embeddings)
        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
        self._encoding = Seq2SeqEncoder(nn.LSTM, self.embedding_dim, self.hidden_size, bidirectional=True)
        self._attention = SoftmaxAttention()
        self._projection = nn.Sequential(nn.Linear(4 * 2 * self.hidden_size, self.hidden_size), nn.ReLU())
        self._composition = Seq2SeqEncoder(nn.LSTM, self.hidden_size, self.hidden_size, bidirectional=True)
        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2 * 4 * self.hidden_size, self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size, self.num_classes))

        # Initialize all weights and biases in the model.
        self.apply(_init_esim_weight)

    def forward(self, premises, premises_lengths, hypotheses, hypotheses_lengths):
        """

        :param premises: a batch of variable length sequences of word indices representing premises,
                         with size of (batch, premises_length)
        :param premises_lengths: a 1D tensor containing the lengths of the premises
        :param hypotheses: a batch of variable length sequences of word indices representing hypotheses,
                         with size of (batch, hypotheses_length)
        :param hypotheses_lengths: a 1D tensor containing the lengths of the hypotheses
        :return:
        """
        premises_mask = get_mask(premises, premises_lengths).to(self.device)
        hypotheses_mask = get_mask(hypotheses, hypotheses_lengths).to(self.device)

        embedded_premises = self._word_embedding(premises)
        embedded_hypotheses = self._word_embedding(hypotheses)

        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)

        encoded_premises = self._encoding(embedded_premises,
                                          premises_lengths)
        encoded_hypotheses = self._encoding(embedded_hypotheses,
                                            hypotheses_lengths)

        attended_premises, attended_hypotheses = \
            self._attention(encoded_premises, premises_mask,
                            encoded_hypotheses, hypotheses_mask)

        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses -
                                         attended_hypotheses,
                                         encoded_hypotheses *
                                         attended_hypotheses],
                                        dim=-1)

        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)

        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)
            projected_hypotheses = self._rnn_dropout(projected_hypotheses)

        v_ai = self._composition(projected_premises, premises_lengths)
        v_bj = self._composition(projected_hypotheses, hypotheses_lengths)

        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) \
                  / torch.sum(premises_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) \
                  / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities
