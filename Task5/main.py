import os
import math
import torch
import torch.nn as nn
from torchtext import data
from torchtext.data import BucketIterator
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import trange, tqdm
from model import LSTM_LM

# hyper parameters
BATCH_SIZE = 64
HIDDEN_DIM = 512
LAYER_NUM = 1
EPOCHS = 200
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.01
EMBEDDING_SIZE = 200
MAX_SEQ_LEN = 64
MOMENTUM = 0.9
DECAY_RATE = 0.05  # learning rate decay rate
CLIP = 5
TEMPERATURE = 0.8  # Higher temperature means more diversity

EOS_TOKEN = "[EOS]"
DATA_PATH = "data"


def read_data(input_file, max_seq_len):
    with open(input_file, encoding='utf-8') as f:
        poetries = []
        poetry = []
        for line in f:
            contents = line.strip()
            if len(poetry) + len(contents) <= max_seq_len:
                if contents:
                    poetry.extend(contents)
                else:
                    poetries.append(poetry)
                    poetry = []
            else:
                poetries.append(poetry)
                poetry = list(contents)
        if poetry:
            poetries.append(poetry)
        return poetries


class PoetryData(data.Dataset):
    def __init__(self, text_field, datafile, max_seq_len, **kwargs):
        fields = [("text", text_field)]
        datas = read_data(datafile, max_seq_len)
        examples = []
        for text in datas:
            examples.append(data.Example.fromlist([text], fields))
        super(PoetryData, self).__init__(examples, fields, **kwargs)


def load_iters(eos_token="[EOS]", batch_size=32, device="cpu", data_path="data", max_seq_len=128):
    # Configuration information for data preprocessing
    TEXT = data.Field(eos_token="[EOS]", batch_first=32, include_lengths=True)
    datas = PoetryData(TEXT, os.path.join(data_path, "poetryFromTang.txt"), max_seq_len)
    train_data, dev_data, test_data = datas.split([0.8, 0.1, 0.1])

    TEXT.build_vocab(train_data)

    train_iter, dev_iter, test_iter = BucketIterator.splits((train_data, dev_data, test_data),
                                                            batch_sizes=(batch_size, batch_size, batch_size),
                                                            device=device, sort_key=lambda x: len(x.text),
                                                            sort_within_batch=True, repeat=False, shuffle=True)
    return train_iter, dev_iter, test_iter, TEXT


def eval(data_iter, is_dev=False, epoch=None):
    model.eval()
    with torch.no_grad():
        total_words = 0
        total_loss = 0
        for i, batch in enumerate(data_iter):
            text, lens = batch.text
            inputs = text[:, :-1]
            targets = text[:, 1:]
            model.zero_grad()
            init_hidden = model.lstm.init_hidden(inputs.size(0))
            logits, _ = model(inputs, lens - 1, init_hidden)
            loss = loss_func(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            total_loss += loss.item()
            total_words += lens.sum().item()
    if epoch is not None:
        tqdm.write("Epoch: %d, %s perplexity %.3f" % (
            epoch + 1, "Dev" if is_dev else "Test", math.exp(total_loss / total_words)))
        writer.add_scalar("Dev_loss", total_loss, epoch)
    else:
        tqdm.write("%s perplexity %.3f" % ("Dev" if is_dev else "Test", math.exp(total_loss / total_words)))


def train(train_iter, dev_iter, loss_func, optimizer, epochs, clip):
    for epoch in trange(epochs):
        model.train()
        total_loss = 0
        total_words = 0
        for i, batch in enumerate(tqdm(train_iter)):
            text, lens = batch.text
            if epoch == 0 and i == 0:
                tqdm.write(' '.join([TEXT.vocab.itos[i] for i in text[0]]))
                tqdm.write(' '.join([str(i.item()) for i in text[0]]))
            inputs = text[:, :-1]
            targets = text[:, 1:]
            init_hidden = model.lstm.init_hidden(inputs.size(0))
            logits, _ = model(inputs, lens - 1, init_hidden)
            loss = loss_func(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            model.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            total_loss += loss.item()
            total_words += lens.sum().item()
        tqdm.write("Epoch: %d, Train perplexity: %d" % (epoch + 1, math.exp(total_loss / total_words)))
        writer.add_scalar("Train_loss", total_loss, epoch)
        eval(dev_iter, True, epoch)

        lr = LEARNING_RATE / (1 + DECAY_RATE * (epoch + 1))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def generate(eos_idx, word, temperature=0.8):
    model.eval()
    with torch.no_grad():
        if word in TEXT.vocab.stoi:
            idx = TEXT.vocab.stoi[word]
            inputs = torch.tensor([idx])
        else:
            print("%s is not in vocabulary, choose by random." % word)
            prob = torch.ones(len(TEXT.vocab.stoi))
            inputs = torch.multinomial(prob, 1)
            idx = inputs[0].item()

        inputs = inputs.unsqueeze(1).to(device)
        lens = torch.tensor([1]).to(device)
        hidden = tuple([h.to(device) for h in model.lstm.init_hidden(1)])
        poetry = [TEXT.vocab.itos[idx]]

        while idx != eos_idx:
            logits, hidden = model(inputs, lens, hidden)
            word_weights = logits.squeeze().div(temperature).exp().cpu()
            idx = torch.multinomial(word_weights, 1)[0].item()
            inputs.fill_(idx)
            poetry.append(TEXT.vocab.itos[idx])
        print("".join(poetry[:-1]))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_iter, dev_iter, test_iter, TEXT = load_iters(EOS_TOKEN, BATCH_SIZE, device, DATA_PATH, MAX_SEQ_LEN)
    pad_idx = TEXT.vocab.stoi[TEXT.pad_token]
    eos_idx = TEXT.vocab.stoi[EOS_TOKEN]

    model = LSTM_LM(len(TEXT.vocab), EMBEDDING_SIZE, HIDDEN_DIM, DROPOUT_RATE, LAYER_NUM).to(device)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='sum')
    writer = SummaryWriter("logs")
    train(train_iter, dev_iter, loss_func, optimizer, EPOCHS, CLIP)
    eval(test_iter, is_dev=False)
    try:
        while True:
            word = input("Input the first word or press Ctrl-C to exit: ")
            generate(eos_idx, word.strip(), TEMPERATURE)
    except:
        pass
