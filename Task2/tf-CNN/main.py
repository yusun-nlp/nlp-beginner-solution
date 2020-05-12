import tensorflow as tf
import numpy as np
import csv
import model
import datetime


def load_data(path):
    """
    Load input data from csv files
    :param path: the path of the file
    :return: the list of data
    """
    csvfile = open(path, 'r')
    csvreader = csv.reader(csvfile, dialect='excel')
    data = []
    for row in csvreader:
        data.append(list(row))
    csvfile.close()
    return data


def clean_data(data):
    """
    Clean raw data into train data
    :param data: raw data
    :return: clean data of Ndarray
    """
    train_data = []
    for row in data:
        train_data.append(row.strip('[]').split(','))
    return np.array(train_data, np.int32)


def train_pred(X, Y, test_X, batch_size=64, epochs=200, embedding_size=128, filter_sizes=[3, 4, 5], num_filters=128,
               dropout_keep_prob=0.5):
    """
    Train TextCNN model and make prediction
    :param X: train data matrix [m, n]
    :param Y: labels matrix [m, classes_num]
    :param test_X: test data matrix [m, n]
    :param batch_size: batch size
    :param epochs: number of training epochs
    :param embedding_size: dimensionality of character embedding, that is the length of the word vector
    :param filter_sizes: list of filter sizes, similar to n-gram
    :param num_filters: number of filters per filter size
    :param dropout_keep_prob: dropout keep probability
    :return: prediction on test_X by trained model
    """
    # build the model
    dicts = load_data('./data/dicts.csv')
    cnn = model.TextCNN(X.shape[1], Y.shape[1], len(dicts), embedding_size, filter_sizes, num_filters)

    # define training procedure
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-4)  # use Adam algorithm to find global optimization
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)  # a gradient update on our parameters

    sess = tf.Session()  # Launch the graph in a session
    sess.run(tf.global_variables_initializer())  # initialize the model parameters

    def train_step(X_batch, Y_batch):
        """
        A single training step
        :param X_batch: a batch of input data
        :param Y_batch: same batch of output label
        :return: null
        """
        # feed_dict contains the data for the placeholder nodes we pass to our network
        feed_dict = {cnn.input_X: X_batch, cnn.input_Y: Y_batch, cnn.dropout_keep_prob: dropout_keep_prob}
        _, step, loss, accuracy = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)

        # output process
        time_str = datetime.datetime.now().isoformat()
        if (step % 1000 == 0):
            print('{}: step {}, loss {:g}, acc {:g}'.format(time_str, step, loss, accuracy))

    def batch_iter(input_X, input_Y, batch_size):
        """
        Generate batches
        :param input_X:
        :param input_Y:
        :param batch_size: batch size
        :return: list of batches
        """
        m = input_X.shape[0]
        blocks = m // batch_size  # the number of W changes
        batches = []
        for i in range(blocks):
            begin = i * batch_size
            end = (i + 1) * batch_size
            batches.append((input_X[begin: end], input_Y[begin: end]))
        if m % batch_size != 0:
            batches.append((input_X[blocks * batch_size:], input_Y[blocks * batch_size:]))
        return batches

    batches = batch_iter(X, Y, batch_size)
    # training loop
    for k in range(epochs):
        for batch in batches:
            X_batch, Y_batch = batch
            train_step(X_batch, Y_batch)
            cur_step = tf.train.global_step(sess, global_step)

    def make_prediction(X_batch, Y_batch):
        """
        A prediction step
        :param X_batch: a batch of input test data
        :param Y_batch: [0]*classes_num
        :return: predictions
        """
        # use feed_dict to feed data
        feed_dict = {cnn.input_X: X_batch, cnn.input_Y: Y_batch, cnn.dropout_keep_prob: 1.0}
        predictions = sess.run(cnn.predictions, feed_dict)
        return predictions

    test_Y = []
    for test_case in test_X:
        test_Y.append(make_prediction([test_case], [[0, 0, 0, 0, 0]])[0])

    sess.close()
    return test_Y


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


if __name__ == '__main__':
    # load data
    train_raw = load_data('./data/train_in.csv')
    test_raw = load_data('./data/test_in.csv')

    # limit the data size, in case of running out of the memory
    train_size = 10000
    test_size = 5000

    # convert raw data to training data (Ndarray)
    train_X, train_Y = zip(*train_raw)  # unzip the train data into input data and label
    train_X, train_Y = clean_data(train_X[:train_size]), clean_data(train_Y[:train_size])
    test_X = np.array(test_raw[:test_size], np.int32)
    print(train_X.shape, train_Y.shape, test_X.shape)

    test_Y = train_pred(train_X, train_Y, test_X)

    # save the prediction result to csv file
    output = [['PhraseId', 'Sentiment']]
    for i in range(1, test_size):
        output.append([test_raw[i][0], test_Y[i - 1]])  # Phrase in test_raw + prediction sentiment
    save_data('result.csv', output)
