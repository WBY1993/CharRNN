import tensorflow as tf
import input_data
import rnn
import os

BATCH_SIZE = 32
TIME_STEP = 26
INPUT_SIZE = 128
CLASS_NUM = 5387
LEARNING_RATE = 0.001
EPOCHS = 2
SAVE_DIR = "/home/rvlab/program/Char_RNN/log/"

model = rnn.CharRNN(BATCH_SIZE, TIME_STEP, INPUT_SIZE, CLASS_NUM, LEARNING_RATE)
summary_op = tf.summary.merge_all()
saver = tf.train.Saver()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
tra_summary_writer = tf.summary.FileWriter(os.path.join(SAVE_DIR, "tra"), sess.graph)

step = -1
for e in range(EPOCHS):
    state = sess.run(model.initial_state)
    for x, y in input_data.get_batches("/home/rvlab/program/0data/poetry.txt", BATCH_SIZE, TIME_STEP):
        step += 1
        _, summary_str, state, tra_loss, tra_acc = sess.run([model.train_op, summary_op, model.new_state, model.loss, model.acc],
                                                            feed_dict={model.inputs: x,
                                                                       model.targets: y,
                                                                       model.keep_prob: 0.5,
                                                                       model.initial_state: state})
        if step % 100 == 0:
            tra_summary_writer.add_summary(summary_str, step)
            print("Epoch %d, Train Step %d, loss: %.4f, accuracy: %.4f" % (e, step, tra_loss, tra_acc))
    saver.save(sess, os.path.join(SAVE_DIR, "model.ckpt"), global_step=step)

sess.close()
