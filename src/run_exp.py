from subprocess import Popen, PIPE
import numpy as np
import itertools

proc = []
n_gpus = 8
lstm_drouput = np.arange(0.1, 0.5, 0.1)
embedding_dropout = np.arange(0, 0.5, 0.1)
for i, (ld, ed) in enumerate(itertools.product(lstm_drouput, embedding_dropout)):
    ld = round(ld, 1)
    ed = round(ed, 1)
    path = "model/char[+]_lstms[{}]_embeddings[{}]".format(ld, ed)
    device = 'GPU:{}'.format(i%n_gpus)
    command = ("python src/main train "
                "--use-char-lstm --epochs 10 "
                 "--dynet-mem 2048 --dynet-autobatch 1 "
                 "--model-path-base {} "
                 "--dropouts {} {} "
                "--dynet-devices {} "
                 ).format(path, ld, ed, device)
    proc.append(Popen(command, shell=True, stdout=PIPE))
    # poll = [p.poll() for p in proc]
    # if poll is None: still running
