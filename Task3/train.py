import json
import pickle
from model import *
from torch.utils.data import DataLoader
import torch
import time


def getCorrectNum(probs, targets):
    _, out_classes = probs.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


def train(model, data_loader, optimizer, criterion, max_gradient_norm):
    model.train()
    device = model.device

    time_epoch_start = time.time()
    running_loss = 0
    correct_cnt = 0
    batch_cnt = 0

    for index, batch in enumerate(data_loader):
        time_batch_start = time.time()
        # load data from data_loader
        premises = batch["premises"].to(device)
        premises_len = batch["premises_len"].to(device)
        hypothesis = batch["hypothesis"].to(device)
        hypothesis_len = batch["hypothesis_len"].to(device)
        labels = batch["labels"].to(device)

        # set gradient to 0
        optimizer.zero_grad()

        # forward
        logits, probs = model(premises, premises_len, hypothesis, hypothesis_len)

        # calculate loss
        loss = criterion(logits, labels)

        # backward
        loss.backward()

        # cut gradient and update weights
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()

        running_loss += loss.item()
        correct_cnt += getCorrectNum(probs, labels)
        batch_cnt += 1
        print("Training -----> Batch count：{:d}/{:d}, batch_time: {:.4f}s, batch average loss: {:.4f}"
              .format(batch_cnt, len(data_loader), time.time() - time_batch_start, running_loss / (index + 1)))

    epoch_time = time.time() - time_epoch_start
    epoch_loss = running_loss / len(data_loader)
    epoch_accuracy = correct_cnt / len(data_loader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy


def validate(model, data_loader, criterion):
    model.eval()
    device = model.device

    time_epoch_start = time.time()
    running_loss = 0
    correct_cnt = 0
    batch_cnt = 0

    for index, batch in enumerate(data_loader):
        time_batch_start = time.time()
        # load data from data loader
        premises = batch["premises"].to(device)
        premises_len = batch["premises_len"].to(device)
        hypothesis = batch["hypothesis"].to(device)
        hypothesis_len = batch["hypothesis_len"].to(device)
        labels = batch["labels"].to(device)

        # set gradient to 0
        optimizer.zero_grad()

        # forward
        logits, probs = model(premises, premises_len, hypothesis, hypothesis_len)

        # calculate loss
        loss = criterion(logits, labels)

        running_loss += loss.item()
        correct_cnt += getCorrectNum(probs, labels)
        batch_cnt += 1
        print("Testing -----> Batch count：{:d}/{:d}, batch_time: {:.4f}s, batch average loss: {:.4f}"
              .format(batch_cnt, len(data_loader), time.time() - time_batch_start, running_loss / (index + 1)))

    epoch_time = time.time() - time_epoch_start
    epoch_loss = running_loss / len(data_loader)
    epoch_accuracy = correct_cnt / len(data_loader.dataset)
    return epoch_time, epoch_loss, epoch_accuracy


if __name__ == '__main__':
    # load configuration file
    with open("config.json", 'r') as f:
        config = json.load(f)

    # set values
    vocab_path = config["vocab_path"]
    train_id_file = config["train_id_file"]
    dev_id_file = config["dev_id_file"]
    embedding_matrix_file = config["embedding_matrix_file"]
    model_train_path = config["model_train_path"]

    # set hyper parameters
    batch_size = config["batch_size"]
    patience = config["patience"]
    hidden_size = config["hidden_size"]
    dropout = config["dropout"]
    num_classes = config["num_classes"]
    lr = config["lr"]
    epochs = config["epochs"]
    max_grad_norm = config["max_grad_norm"]
    device = torch.device("cpu")

    # load data
    with open(train_id_file, 'rb') as f:
        train_data = SnliDataSet(pickle.load(f))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    with open(dev_id_file, 'rb') as f:
        dev_data = SnliDataSet(pickle.load(f))
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)

    # load embedding
    with open(embedding_matrix_file, 'rb') as f:
        embeddings = torch.tensor(pickle.load(f), dtype=torch.float).to(device)

    # build model
    model = ESIM(embeddings.shape[0], embeddings.shape[1], hidden_size, embeddings, dropout=dropout,
                 num_classes=num_classes, device=device).to(device)

    # prepare training
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=0)

    # set parameters
    best_score = 0.0
    train_losses = []
    valid_losses = []
    patience_cnt = 0

    for epoch in range(epochs):
        # training
        print("Training epoch %d" % (epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer, criterion, max_grad_norm)
        train_losses.append(epoch_loss)
        print("Training time: {:.4f}s, loss :{:.4f}, accuracy: {:.4f}%".format(epoch_time, epoch_loss,
                                                                               (epoch_accuracy * 100)))

        # validation
        print("Validating epoch %d" % (epoch))
        epoch_time_dev, epoch_loss_dev, epoch_accuracy_dev = validate(model, dev_loader, criterion)
        valid_losses.append(epoch_loss_dev)
        print("Validating time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%\n".format(epoch_time_dev, epoch_loss_dev,
                                                                                   (epoch_accuracy_dev * 100)))

        # update learning rate
        scheduler.step(epoch_accuracy)

        # early stopping
        if epoch_accuracy_dev < best_score:
            patience_cnt += 1
        else:
            best_score = epoch_accuracy_dev
            patience_cnt = 0
        if patience_cnt >= patience:
            print("Early Stopping")
            break

        # save model
        torch.save({"epoch": epoch, "model": model.state_dict(), "best_score": best_score,
                    "train_losses": train_losses, "valid_losses": valid_losses},
                   model_train_path + str(epoch) + ".dir")
