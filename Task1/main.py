import csv
import numpy as np

classes_num = 5  # number of classes of result


def load_data(path):
    """
    Load data from tsv files
    :param path: the path of the file
    :return: Ndararry of data
    """
    tsvfile = open(path, 'r')  # read only
    tsvreader = csv.reader(tsvfile, delimiter='\t')  # read row into a list, every row divided by \t
    data = []
    for row in tsvreader:
        data.append(row)
    tsvfile.close()
    return np.array(data)  # switch list to Ndarray


def get_dicts(text, n):
    """
    Generate the dictionary of n_gram model
    :param text: list of words, e.g. [['Are','you','OK'],['Yes']
    :param n: the dimension of n_gram
    :return: Ndararry of dictionary
    """
    assert (n >= 1)  # assure n is bigger than 1
    dicts = set()  # eliminate
    dicts.add('')  # use as the bias position
    for sentence in text:
        for i in range(len(sentence)):
            for length in range(n):
                if i - length >= 0:  # generate n-gram dictionary
                    dicts.add(' '.join([word.lower() for word in sentence[i - length:i + 1]]))
    dicts = sorted(list(dicts))
    return np.array(dicts)


def label_to_one_hot(label):
    """
    convert list to one hot vector
    :param label: label of list
    :return: label of one hot vector
    """
    vector = np.zeros(classes_num)
    vector[label] = 1
    return vector


def get_features(data_raw, dicts, tag):
    """
    convert raw sentences to feature vectors
    :param data_raw: raw data
    :param dicts: the dictionary of n-gram
    :param tag: 'train' or 'test'
    :return: feature matrix and label matrix, each column is a sample and a label
    """
    features = []
    labels = []
    n = len(data_raw)
    for i in range(1, n):  # go through all rows instead of the first one
        sentence = data_raw[i][2].lower()  # change all sentences to lowercase
        feature = np.array([word in sentence for word in dicts], np.int32)  # change every word in dicts to integer
        features.append(feature)
        if tag == 'train':
            label = int(data_raw[i][3])  # sentiment number
            labels.append(label_to_one_hot(label))
    return np.array(features).T, np.array(labels).T


def softmax(Y):
    """
    Compute the softmax for each column of matrix Y
    :param Y: matrix
    :return: softmax value
    """
    exp_x = np.exp(Y)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


def predict(X, W):
    """
    Generate the prediction of given dataset X with the weight W
    :param X: input training set
    :param W: weight matrix
    :return: predicted label matrix
    """
    Y = W.dot(X)
    return softmax(Y)


def loss(train_Y, pred_Y):
    """
    Calculate the loss for the prediction with train
    :param train_Y: wanted result
    :param pred_Y: predicted result
    :return: loss function value
    """
    m = train_Y.shape[1]  # batch size
    Y = (np.log(pred_Y) / m) * train_Y
    return -np.sum(Y)


def vector_to_labels(Y):
    """
    Convert prediction matrix to a vector of label, that is change on-hot vector to a label number
    :param Y: prediction matrix
    :return: a vector of label
    """
    labels = []
    Y = list(Y.T)  # each row of Y.T is a sample
    for vec in Y:
        vec = list(vec)
        labels.append(vec.index(max(vec)))  # find the index of 1
    return np.array(labels)


def cal_acc(train_Y, pred_Y):
    """
    Calculate the accuracy of prediction
    :param train_Y: standard result
    :param pred_Y: predicted result
    :return: accuracy
    """
    return np.sum(train_Y == pred_Y) / train_Y.shape[0]


def training(X, Y, epochs, method='mini batch', W=None, learning_rate=0.003, batch_size=64):
    """
    Train the weight vector W on the dataset(X, Y)
    :param X: train data matrix [n, m]
    :param Y: labels matrix [classes_num, m]
    :param epochs: times go through the whole dataset
    :param method: 'SGD', 'BGD' or 'mini batch'
    :param W: initial value of weight vector [classes_num, n]
    :param learning_rate: learning rate for gradient descent
    :param batch_size: update W every batch size
    :return: trained weight vector and loss value
    """
    m = Y.shape[1]  # number of training examples
    n = X.shape[0]  # number of features
    print(X.shape, Y.shape)

    # decide the value of batch size
    if method == 'SGD':
        batch_size = 1
    elif method == 'BGD':
        batch_size = m
    assert (method in ['SGD', 'BGD', 'mini batch'])  # assure method is available

    # initialize W if W is null
    if not W:
        W = 0.1 * np.random.randn(classes_num, n)
        W[:, 0] = np.zeros(classes_num)

    loss_cache = []

    # train the dataset
    for k in range(epochs + 1):
        X_Y = list(zip(X.T, Y.T))  # zip every couple of data and label
        np.random.shuffle(X_Y)
        X, Y = zip(*X_Y)  # unzip
        X, Y = np.array(X).T, np.array(Y).T
        blocks = m // batch_size  # the integer part of m/batch_size, refer to the number of W changes
        for i in range(blocks + 1):  # the last block may not be full
            begin = i * batch_size
            end = (i + 1) * batch_size
            if end > m:
                end = m
            if begin == end:
                break
            train_X, train_Y = X[:, begin:end], Y[:, begin:end]  # train set and label in this block
            pred_Y = predict(train_X, W)
            dW = (pred_Y - train_Y).dot(train_X.T)  # gradient descent
            W = W - learning_rate * dW
        if k % 100 == 0:  # every 100 times output the result
            pred_Y = predict(X, W)
            L = loss(Y, pred_Y)
            loss_cache.append(L)
            acc = cal_acc(vector_to_labels(Y), vector_to_labels(pred_Y))
            print('epochs ' + str(k) + ': loss: ' + str(L) + ' accuracy: ' + str(acc))

    return W, loss_cache


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


if __name__ == "__main__":
    # load data
    train_raw = load_data('./data/train.tsv')
    test_raw = load_data('./data/test.tsv')
    print('data loaded')

    # limit the data size, in case of running out of the memory
    train_size = 10000
    test_size = 5000

    # get dictionary
    phrase = [train_raw[i][2].split(' ') for i in range(1, train_size)]  # get words in phrase
    dicts = get_dicts(phrase, 3)
    print('dictionary ready')

    # convert raw data to training data
    train_X, train_Y = get_features(train_raw[:train_size], dicts, tag='train')
    print('train data ready')
    test_X, _ = get_features(test_raw[:test_size], dicts, tag='test')
    print('test data ready')

    # finish loading data
    del dicts
    del train_raw

    # train W using softmax regression
    W, loss_cache = training(train_X, train_Y, epochs=2000, learning_rate=0.005, batch_size=32)
    print('train ending')

    # make prediction on test dataset
    test_Y = vector_to_labels(predict(test_X, W))

    # save the prediction result to csv file
    output = [['Phrase', 'Sentiment']]
    for i in range(1, test_size):
        output.append([test_raw[i][0], test_Y[i - 1]])  # Phrase in test_raw + prediction sentiment
    save_data('result.csv', output)
