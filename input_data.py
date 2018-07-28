import numpy as np


def text_converter(file_name):
    infile = open(file_name, "r")
    text = infile.read()
    text_set = set(text)
    text_set_char2int = {c: i for i, c in enumerate(text_set)}
    text_set_int2char = dict(enumerate(text_set))
    outfile1 = open("char2int.txt", "w")
    outfile1.write(str(text_set_char2int))
    outfile1.close()
    outfile2 = open("int2char.txt", "w")
    outfile2.write(str(text_set_int2char))
    outfile2.close()
    infile.close()
    print(len(text_set))


def get_batches(file_name, batch_size, time_step):
    dic_file = open("/home/rvlab/program/Char_RNN/char2int.txt", "r")
    text_set_char2int = dic_file.read()
    text_set_char2int = eval(text_set_char2int)
    dic_file.close()
    text = open(file_name).read()
    text_encode = np.array([text_set_char2int[c] for c in text])
    n_batches = int(len(text_encode) / (batch_size * time_step))
    arr = text_encode[:batch_size * time_step * n_batches]
    arr = np.reshape(arr, [batch_size, -1])
    for n in range(0, arr.shape[1], time_step):
        x = arr[:, n:n+time_step]
        y = np.zeros_like(x)
        y[:, :-1] = x[:, 1:]
        y[:, -1] = x[:, 0]
        yield x, y


# text_converter("/home/rvlab/program/0data/poetry.txt")
