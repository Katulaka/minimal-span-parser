from subprocess import Popen, PIPE, DEVNULL
import numpy as np
import itertools

proc = []
n_gpus = 8
embed_dim = [64, 128, 256, 512]
dropouts = [0.1, 0.2, 0.3, 0.4]
iter = itertools.product(dropouts, dropouts)
for i, drops in enumerate(iter):
    path = ("models_grid_search/char[+]_lstms[{}]_embeddings[{}]"
            "_temb[64]_cemb[64]_wemb[256]_lemb[256]").format(*drops)
    device = 'GPU:{}'.format(i%n_gpus)
    command = ("python src/main.py train "
                "--use-char-lstm --epochs 10 --parser-type my "
                "--dropouts {} {} "
                 "--dynet-mem 2048 --dynet-autobatch 1 "
                "--tag-embedding-dim 64 "
                "--char-embedding-dim 64 "
                "--word-embedding-dim 256 "
                "--label-embedding-dim 256 "
                # "--char-lstm-dim {100} "
                # "--lstm-dim {250} "
                # "--dec-lstm-dim {600} "
                # "--attention-dim {250} "
                # "--label-hidden-dim {250} "
                "--dynet-devices {} "
                 "--model-path-base {} "
                 ).format(*drops, device, path)
    proc.append(Popen(command.split()))
