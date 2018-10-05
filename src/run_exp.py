from subprocess import Popen, PIPE, DEVNULL
import numpy as np
import itertools

proc = []
n_gpus = 8
embed_dim = [64, 128, 256, 512]
embed_iter = itertools.product(embed_dim[:-1], embed_dim[:-1], embed_dim[1:], embed_dim[1:])
for i, dims in enumerate(embed_iter):
    path = ("models_grid_search/char[-]_lstms[0.1]_embeddings[0.4]"
            "_temb[{}]_cemb[{}]_wemb[{}]_lemb[{}]").format(*dims)
    device = 'GPU:{}'.format(i%n_gpus)
    command = ("python src/main.py train "
                "--epochs 10 --parser-type my "
                 "--dynet-mem 2048 --dynet-autobatch 1 "
                 "--model-path-base {} "
                 "--dropouts 0.1 0.4 "
                "--dynet-devices {} "
                "--tag-embedding-dim {} "
                "--char-embedding-dim {} "
                "--word-embedding-dim {} "
                "--label-embedding-dim {} "
                 ).format(path, device, *dims)
    proc.append(Popen(command.split()))

    #
    #
    #
    #
    #
    # "--char-lstm-dim 100"
    # "--lstm-dim 250"
    # "--dec-lstm-dim 600"
    # "--attention-dim 250"
    # "--label-hidden-dim 250"
