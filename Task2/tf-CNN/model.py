import tensorflow as tf


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """

    def __init__(self, seq_len, classes_num, dict_size, embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        """
        :param seq_len: length of given sentence sequence
        :param classes_num: number of classification
        :param dict_size: size of dictionary
        :param embedding_size: kernel width
        :param filter_sizes: list of kernel height
        :param num_filters: out_channels, that is the number of kernels
        :param l2_reg_lambda: lambda value of l2 regularization
        """
        # placeholders for input, output and dropout
        self.input_X = tf.placeholder(tf.int32, [None, seq_len], name='input_X')  # seq_len columns
        self.input_Y = tf.placeholder(tf.float32, [None, classes_num], name='input_Y')  # classes_num columns
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')  # dropped out probability

        # Keep track of l2 regularization loss
        l2_loss = tf.constant(0.0)

        # embedding layer, convert the input word id into a word vector
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            # randomly generate matrix with size [dict_size, embedding_size], value from -1.0 to 1.0
            self.W = tf.Variable(tf.random_uniform([dict_size, embedding_size], -1.0, 1.0), name='W')
            # select the element corresponding to the index(input_X) in a tensor(W)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_X)
            # add a dimension to the word vector result
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # create a convolution and max-pool layer for each filter size
        pooled_outputs = [] # save the results of each convolution
        for i, filter_size in enumerate(filter_sizes):  # index sequence (i, filter_size)
            with tf.name_scope('conv-maxpool-%s' % filter_size):
                # convolution layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]  # text's in_channel is 1
                # generate random values from a truncated normal distribution, with 0.1 standard deviation
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                # generate a constant vector with value of 0.1
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
                # kernel is W, stride [batch, height, width, channels]=[1,1,1,1], no padding
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding='VALID',
                                    name='conv')
                # activation function, return [batch, height, width, channels]
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')  # add b into conv, then compute relu function
                # max-pooling over the outputs, ksize is size of pool, return [batch, height, width, channels]
                pooled = tf.nn.max_pool(h, ksize=[1, seq_len - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                        padding='VALID', name='pool')
                pooled_outputs.append(pooled)

        # combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)  # concat pooled_outputs in axis 3
        # reshape pooled features into [len(input_x), num_filters*len(filters_size)]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # add dropout to prevent overfitting
        with tf.name_scope('dropout'):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # output layer, feature map to probabilities on different classes
        with tf.name_scope('dropout'):
            # generate random values from a truncated normal distribution, with 0.1 standard deviation
            W = tf.Variable(tf.truncated_normal([num_filters_total, classes_num], stddev=0.1), name='W')
            # generate a constant vector with value of 0.1
            b = tf.Variable(tf.constant(0.1, shape=[classes_num]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # compute h_drop*W+b
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name='scores')
            # get the index number of the maximum value
            self.predictions = tf.argmax(self.scores, 1, name='predictions')

        # calculate mean cross-entropy loss
        with tf.name_scope('loss'):
            # scores is predicted result, input_Y is actual result
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_Y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss  # compute mean loss

        # calculate accuracy
        with tf.name_scope('accuracy'):
            correct_predicitons = tf.equal(self.predictions, tf.argmax(self.input_Y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predicitons, 'float'), name='accuracy')
