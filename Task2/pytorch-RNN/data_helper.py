import tensorflow.keras as kr
import numpy as np


def read_file(filename):
    """
    Read the file
    :param filename:
    :return: contents and labels
    """
    contents, labels = [], []
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            label, content = line.strip().split('\t')
            if content:
                contents.append(list(content))
                labels.append(label)
    return contents, labels


def read_category():
    """
    Read the category to index
    :return: categories, cat_id
    """
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    categories = [x for x in categories]
    cat_id = dict(zip(categories, range(len(categories))))
    return categories, cat_id


def read_vocab(vocab_dir):
    """
    Read the vocabulary
    :param vocab_dir: vocabulary dictionary
    :return: word and the index
    """
    with open(vocab_dir, 'r', encoding='utf-8', errors='ignore') as f:
        words = [_.strip() for _ in f.readlines()]
    word_id = dict(zip(words, range(len(words))))
    return words, word_id


def batch_iter(X, Y, batch_size=64):
    data_len = len(X)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    X_shuffle = X[indices]
    Y_shuffle = Y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield X_shuffle[start_id:end_id], Y_shuffle[start_id:end_id]


def data_preprocess(filename, word_id, cat_id, max_len=600):
    """
    exchange file to id
    :param filename: filename
    :param word_id: word index
    :param cat_id: category index
    :param max_len: max length
    :return:
    """
    contents, labels = read_file(filename)
    data_id, label_id = [], []
    for i in range(min(len(contents), 10000)):
        data_id.append([word_id[x] for x in contents[i] if x in word_id])
        label_id.append(cat_id[labels[i]])

    # set text as fixed length
    X_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_len)
    # change label to one-hot vector
    Y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_id))

    return X_pad, Y_pad
