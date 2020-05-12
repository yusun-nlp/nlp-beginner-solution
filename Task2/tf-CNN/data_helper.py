import csv

classes_num = 5


def load_data(path):
    """
    Load data from tsv files
    :param path: the path of the file
    :return: list of data
    """
    tsvfile = open(path, 'r')  # read only
    tsvreader = csv.reader(tsvfile, delimiter='\t')  # read row into a list, every row divided by \t
    data = []
    for row in tsvreader:
        data.append(row)
    tsvfile.close()
    return data  # switch list to Ndarray


def save_data(path, data):
    """
    Save data to a csv file
    :param path: the path of the file
    :param data: prediction data
    :return: null
    """
    csvfile = open(path, 'w', newline='')
    csvwriter = csv.writer(csvfile, dialect='excel')
    for result in data:
        csvwriter.writerow(result)
    csvfile.close()


def get_dicts(data_raw):
    """
    Generate the dictionary of words in the sample
    :param data_raw: raw input data
    :return: list of dictionary, max size of sentences
    """
    dicts = set()  # eliminate
    dicts.add('')  # use as the bias position
    max_len = 0  # max length of a sentence
    for row in data_raw:
        sentence = row[2].split(' ')
        max_len = max(max_len, len(sentence))
        dicts.update(set([word.lower() for word in sentence]))

    return dicts, max_len


def label_to_one_hot(label):
    """
    convert list to one hot vector
    :param label: label of list
    :return: label of one hot vector
    """
    vector = [0] * classes_num
    vector[label] = 1
    return vector


def get_features(data_raw, dicts, seq_len, tag):
    """
    convert raw sentences to feature vectors
    :param data_raw: raw data
    :param dicts: the dictionary of n-gram
    :param seq_len: length of input sequence
    :param tag: 'train' or 'test'
    :return: [[[feature vector],[label vector]], ... ] or [[feature vector], ...]
    """
    input = []
    for row in data_raw:  # go through all rows instead of the first one
        sentence = row[2].split(' ')
        # the input and output of CNN are fixed, but the length of each sentence is different, insufficient supplement 0
        while (len(sentence) < seq_len):
            sentence.append(' ')
        features = []
        labels = []
        for word in sentence:
            features.append(dicts.index(word.lower()))  # change every word in dicts to integer
        if tag == 'train':
            label = int(row[3])  # sentiment number
            labels = label_to_one_hot(label)
            input.append([features, labels])
        else:
            input.append(features)
    return input


if __name__ == '__main__':
    # load data
    train_raw = load_data('./data/train.tsv')[1:]
    test_raw = load_data('./data/test.tsv')[1:]
    print('data loaded')

    # get the dictionary
    dicts_train, train_len = get_dicts(train_raw)
    dicts_test, test_len = get_dicts(test_raw)
    dicts = dicts_train.union(dicts_test)  # union train and test dictionary
    dicts = [' '] + list(dicts)
    seq_len = max(train_len, test_len)
    save_data('./data/dicts.csv', dicts)
    print('dictionary ready. Sequence length: ', seq_len)
    print('Dictionary size: ', len(dicts))

    # convert raw data to training data
    train_input = get_features(train_raw, dicts, seq_len, tag='train')
    print('train data ready')
    test_input = get_features(test_raw, dicts, seq_len, tag='test')
    print('test data ready')

    # save the preprocessing data to csv file
    save_data('./data/train_in.csv', train_input)
    save_data('./data/test_in.csv', test_input)
