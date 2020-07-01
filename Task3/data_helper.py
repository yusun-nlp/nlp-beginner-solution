import json
import string
import pickle
import numpy as np


def load_data(data_path):
    """
    load data from data_path
    :param data_path: data path
    :return: dictionary of data
    """
    premise = []
    hypothesis = []
    labels = []
    labels_vocab = {"entailment": 0, "neutral": 1, "contradiction": 2}
    trans_table = str.maketrans({key: " " for key in string.punctuation})
    with open(data_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            line = line.strip().split('\t')
            if line[0] not in labels_vocab:  # ignore instances without labels
                continue
            premise.append(line[5].translate(trans_table).lower())
            hypothesis.append(line[6].translate(trans_table).lower())
            labels.append(line[0])

    return {"premise": premise, "hypothesis": hypothesis, "labels": labels}


def build_vocab(data, vocab_path):
    """
    build vocabulary dictionary
    :param data: processed data
    :param vocab_path: file path to save vocabs
    :return: dictionaries
    """
    # get vocabularies
    words = []
    # add special token
    # PAD: padding, OOV: out of vocabulary, BOS: begin of sentence, EOS: end of sentence
    words.extend(["_PAD_", "_OOV_", "_BOS_", "_EOS_"])
    for sentence in data["premise"]:
        words.extend(sentence.strip().split(" "))
    for sentence in data["hypothesis"]:
        words.extend(sentence.strip().split(" "))

    # build vocabulary dictionary
    word_id = {}
    id_word = {}
    index = 0
    for i, word in enumerate(words):
        if word not in word_id:
            word_id[word] = index
            id_word[index] = word
            index += 1

    # save the dictionary
    with open(vocab_path, 'w', encoding='utf-8') as f:
        for word in word_id:
            f.write("%s\t%d\n" % (word, word_id[word]))
    return word_id, id_word


def sentence2id(sentence, word_id):
    """
    Change sentence to id list based on word id
    :param sentence: sentence
    :param word_id: word id list
    :return: sentence id list
    """
    ids = []
    ids.append(word_id["_BOS_"])  # add begin label
    sentence = sentence.strip().split(" ")
    for word in sentence:
        if word not in word_id:
            ids.append(word_id["_OOV_"])
        else:
            ids.append(word_id[word])
    ids.append(word_id["_EOS_"])  # add end label
    return ids


def data2id(data, word_id):
    """
    Change data to id based on word id
    :param data: data
    :param word_id: word id list
    :return: data id dictionary
    """
    premise_id = []
    hypothesis_id = []
    labels_id = []
    labels_vocab = {"entailment": 0, "neutral": 1, "contradiction": 2}
    for i, label in enumerate(data["labels"]):
        if label not in labels_vocab:
            continue
        premise_id.append(sentence2id(data["premise"][i], word_id))
        hypothesis_id.append(sentence2id(data["hypothesis"][i], word_id))
        labels_id.append(labels_vocab[label])

    return {"premises": premise_id, "hypothesis": hypothesis_id, "labels": labels_id}


def build_embedding(embedding_file, word_id):
    """
    build embedding matrix based on word id
    :param embedding_file: embedding file
    :param word_id: word id list
    :return: embedding matrix
    """
    embedding_map = {}
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split()
            word = line[0]
            if word in word_id:
                embedding_map[word] = line[1:]
    words_num = len(word_id)
    embedding_dim = len(embedding_map['a'])
    embedding_matrix = np.zeros((words_num, embedding_dim))
    print("size of embedding matrix: ", words_num, embedding_dim)

    missed_num = 0
    for i, word in enumerate(word_id):
        if word in embedding_map:
            embedding_matrix[i] = embedding_map[word]
        else:
            if word == "_PAD_":
                continue
            missed_num += 1
            embedding_matrix[i] = np.random.normal(size=embedding_dim)
    print("missed word number: %d" % (missed_num))
    return embedding_matrix


if __name__ == '__main__':
    # load configuration file
    with open("config.json", 'r') as f:
        config = json.load(f)

    # set values
    train_data_path = config["train_data_path"]
    dev_data_path = config["dev_data_path"]
    embedding_path = config["embedding_path"]

    vocab_path = config["vocab_path"]
    train_data_file = config["train_data_file"]
    dev_data_file = config["dev_data_file"]
    train_id_file = config["train_id_file"]
    dev_id_file = config["dev_id_file"]
    embedding_matrix_file = config["embedding_matrix_file"]

    print("Start processing data.")

    # load data
    train_data = load_data(train_data_path)
    dev_data = load_data(dev_data_path)

    # build the vocabulary dictionary
    word_id, id_word = build_vocab(train_data, vocab_path)

    # clean data and change to id
    data_id_train = data2id(train_data, word_id)
    data_id_dev = data2id(dev_data, word_id)

    # save processed data
    with open(train_data_file, 'wb') as f:
        pickle.dump(train_data, f)
    with open(dev_data_file, 'wb') as f:
        pickle.dump(dev_data, f)
    with open(train_id_file, 'wb') as f:
        pickle.dump(data_id_train, f)
    with open(dev_id_file, 'wb') as f:
        pickle.dump(data_id_dev, f)

    # build embedding matrix
    embedding_matrix = build_embedding(embedding_path, word_id)
    with open(embedding_matrix_file, 'wb') as f:
        pickle.dump(embedding_matrix, f)

    print("preprocessing data finished!")
