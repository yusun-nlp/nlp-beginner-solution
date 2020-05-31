from data_helper import read_category, read_vocab, data_preprocess, batch_iter
from model import TextRNN
import torch
from torch import nn, optim
import numpy as np


def evaluate(model, Loss, X_val, Y_val):
    """
    evaluate the accuracy on validate set
    """
    batch_val = batch_iter(X_val, Y_val)
    acc = 0
    los = 0
    for X_batch, Y_batch in batch_val:
        size = len(X_batch)
        X = np.array(X_batch)
        Y = np.array(Y_batch)
        X = torch.LongTensor(X)
        Y = torch.Tensor(Y)
        out = model(X)
        loss = Loss(out, Y)
        loss_val = np.mean(loss.detach().numpy())
        accuracy = np.mean((torch.argmax(out, 1) == torch.argmax(Y, 1)).numpy())
        acc += accuracy * size
        los += loss_val * size
    return los / len(X_val), acc / len(X_val)


def training():
    # get train set and validate set
    X_train, Y_train = data_preprocess('./data/cnews.train.txt', word_id, cat_id, 600)
    X_val, Y_val = data_preprocess('./data/cnews.val.txt', word_id, cat_id, 600)
    print('train size: ', len(X_train))
    print('Load data')

    # build model
    model = TextRNN()

    # set loss and optimizer
    Loss = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = 0
    for epoch in range(100):
        i = 0
        print('epoh:{}'.format(epoch))
        batch_train = batch_iter(X_train, Y_train)
        for X_batch, Y_batch in batch_train:
            i = i + 1
            X = np.array(X_batch)
            Y = np.array(Y_batch)
            X = torch.LongTensor(X)
            Y = torch.Tensor(Y)
            out = model(X)
            loss = Loss(out, Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 90 == 0:
                los, accuracy = evaluate(model, Loss, X_val, Y_val)
                print('loss:{},accuracy:{}'.format(los, accuracy))
                if accuracy > best_val_acc:
                    torch.save(model.state_dict(), 'model_params.pkl')
                    best_val_acc = accuracy


if __name__ == '__main__':
    # get text category id
    categories, cat_id = read_category()
    # set text word id
    words, word_id = read_vocab('./data/cnews.vocab.txt')
    vocab_size = len(words)
    print('start training')
    training()
