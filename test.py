import tensorflow as tf
import numpy as np
import rnn

BATCH_SIZE = 1
TIME_STEP = 1
INPUT_SIZE = 128
CLASS_NUM = 5387
LEARNING_RATE = 0.001
LOG_DIR = "/home/rvlab/program/Char_RNN/log/"


def pick_top_n(preds, class_num, top_n=5):
    """
    从预测结果中选取前top_n个最可能的字符
    preds: 预测结果
    vocab_size
    top_n
    """
    p = np.squeeze(preds)
    # 将除了top_n个预测值的位置都置为0
    p[np.argsort(p)[:-top_n]] = 0
    # 归一化概率
    p = p / np.sum(p)
    # 随机选取一个字符
    c = np.random.choice(class_num, 1, p=p)[0]

    return c


model = rnn.CharRNN(BATCH_SIZE, TIME_STEP, INPUT_SIZE, CLASS_NUM, LEARNING_RATE)
saver = tf.train.Saver()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print("Reading checkpoints...")
ckpt = tf.train.get_checkpoint_state(LOG_DIR)
if ckpt and ckpt.model_checkpoint_path:
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    saver.restore(sess, ckpt.model_checkpoint_path)
    print('Loading success, global_step is %s' % global_step)
else:
    print('No checkpoint file found')

dic_file1 = open("/home/rvlab/program/Char_RNN/char2int.txt", "r")
text_set_char2int = dic_file1.read()
text_set_char2int = eval(text_set_char2int)
dic_file1.close()
dic_file2 = open("/home/rvlab/program/Char_RNN/int2char.txt", "r")
text_set_int2char = dic_file2.read()
text_set_int2char = eval(text_set_int2char)
dic_file2.close()

prime = "何当"
result = [c for c in prime]
for c in prime:
    state = sess.run(model.initial_state)
    x = np.zeros((1, 1))
    x[0, 0] = text_set_char2int[c]
    pred, state = sess.run([model.pred, model.new_state],
                           feed_dict={model.inputs: x,
                                      model.keep_prob: 1,
                                      model.initial_state: state})

new_c = pick_top_n(pred, CLASS_NUM)
result.append(text_set_int2char[new_c])

for i in range(100):
    x[0, 0] = new_c
    pred, state = sess.run([model.pred, model.new_state],
                           feed_dict={model.inputs: x,
                                      model.keep_prob: 1,
                                      model.initial_state: state})

    new_c = pick_top_n(pred, CLASS_NUM)
    result.append(text_set_int2char[new_c])
print("".join(result))
sess.close()
