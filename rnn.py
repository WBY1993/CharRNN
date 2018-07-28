import tensorflow as tf


class CharRNN:
    def __init__(self, batch_size, time_step, input_size, class_num, lr):
        # input data
        self.inputs, self.targets, self.keep_prob = self.get_input(batch_size, time_step)
        embedding = tf.get_variable("embedding", shape=[class_num, input_size])
        rnn_input = tf.nn.embedding_lookup(embedding, self.inputs)

        # initial
        self.global_step = tf.Variable(0, trainable=False)
        self.initial_state = None

        # building graph
        self.logits, self.new_state = self.inference(rnn_input, class_num, self.keep_prob)
        self.pred = tf.nn.softmax(self.logits)
        self.loss = self.losses(self.logits, self.targets)
        self.acc = self.evaluation(self.logits, self.targets)
        self.train_op = self.training(self.loss, lr, self.global_step)

    def get_input(self, batch_size, time_step):
        inputs = tf.placeholder(tf.int32, shape=[batch_size, time_step], name="inputs")
        targets = tf.placeholder(tf.int32, shape=[batch_size, time_step], name="targets")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        return inputs, targets, keep_prob

    def fc(self, layer_name, x, in_channel, out_channel, is_train=True):
        with tf.variable_scope(layer_name):
            x = tf.reshape(x, shape=[-1, in_channel])
            weights = tf.get_variable(name="weights",
                                      shape=[in_channel, out_channel],
                                      trainable=is_train,
                                      initializer=tf.truncated_normal_initializer(stddev=0.05, dtype=tf.float32))

            biases = tf.get_variable(name="biases",
                                     shape=[out_channel],
                                     trainable=is_train,
                                     initializer=tf.constant_initializer(0.1))

            x = tf.matmul(x, weights)
            x = tf.nn.bias_add(x, biases)
        return x

    def inference(self, x, class_num, keep_prob):
        """
        :param x:
        :param class_num:
        :param keep_prob:
        :return:
        """
        # input data: [batch_size, time_step, input_size]
        with tf.variable_scope("lstm"):
            lstm_cell1 = tf.nn.rnn_cell.BasicLSTMCell(512)
            lstm_cell1 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell1, output_keep_prob=keep_prob)
            lstm_cell2 = tf.nn.rnn_cell.BasicLSTMCell(512)
            lstm_cell2 = tf.nn.rnn_cell.DropoutWrapper(lstm_cell2, output_keep_prob=keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell1, lstm_cell2])
            self.initial_state = cell.zero_state(x.get_shape()[0], tf.float32)
            outputs, new_state = tf.nn.dynamic_rnn(cell, x, initial_state=self.initial_state)
            output = tf.reshape(outputs, shape=[-1, 512])
        x = self.fc("fc1", output, 512, class_num, is_train=True)
        return x, new_state

    def losses(self, logits, targets):
        """
        :param logits: [batch_size, class_num]
        :param targets: [batch_size]
        :return:
        """
        with tf.name_scope("loss"):
            labels = tf.reshape(targets, [-1])
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                           labels=labels)
            loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar("loss", loss)
        return loss

    def evaluation(self, logits, targets):
        """
        :param logits: [batch_size, class_num]
        :param targets: [batch_size]
        :return:
        """
        with tf.name_scope("accuracy"):
            predictions = tf.nn.softmax(logits)
            labels = tf.reshape(targets, [-1])
            correct = tf.nn.in_top_k(predictions, labels, 1)
            correct = tf.cast(correct, tf.float32)
            accuracy = tf.reduce_mean(correct)
            tf.summary.scalar("accuracy", accuracy)
        return accuracy

    def training(self, loss, learning_rate, global_step):
        """
        :param loss:
        :param learning_rate:
        :param global_step:
        :return:
        """
        with tf.name_scope("optimizer"):
            decayed_lr = tf.train.exponential_decay(learning_rate=learning_rate,
                                                    global_step=global_step,
                                                    decay_steps=3000,
                                                    decay_rate=0.1,
                                                    staircase=True)

            optimizer = tf.train.AdamOptimizer(decayed_lr)
            tvars = tf.trainable_variables()
            grads = tf.gradients(loss, tvars)
            grads, _ = tf.clip_by_global_norm(grads, 5)
            grads = zip(grads, tvars)
            train_op = optimizer.apply_gradients(grads, global_step=global_step)

            # Add histograms for gradients trainable variables and gradients
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + "/gradients", grad)

        return train_op
