from subprocess import Popen
import itertools

n_gpus = 8
tag_embedding = [100, 150]
word_embedding = [100, 150]
char_embedding = [50, 100]
char_lstm_dim = [100, 150]
label_embedding = [100, 200]
lstm_dim = [250, 350, 450]
attention_dim = [200, 300]
label_dim = [100, 200]
dropouts = [(0.2, 0.2), (0.3, 0.3), (0.4, 0.4)]
for i, dims in enumerate(itertools.product(lstm_dim, attention_dim, label_dim)):
    device = 'GPU:{}'.format(i%n_gpus)
    command = ("python src/main.py train "
                "--use-char-lstm --parser-type my --model-path-base run_exp "
                "--dynet-mem 2048 --dynet-autobatch 1 "
                # "--tag-embedding-dim {} "
                # "--word-embedding-dim {} "
                # "--char-embedding-dim {} "
                # "--char-lstm-dim {} "
                # "--label-embedding-dim {} "
                "--lstm-dim {} "
                "--label-hidden-dim {} "
                "--attention-dim {} "
                # "--dropouts {} {} "
                "--dynet-devices {} "
                # "--dec-lstm-dim {} "
                 ).format(*dims, device)
    Popen(command.split())
