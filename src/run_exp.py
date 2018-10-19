from subprocess import Popen
import itertools

n_gpus = 8
for i, embeddings in enumerate(itertools.product([100, 150, 200], [250, 300, 350])):
    if embeddings != (100, 250):
        device = 'GPU:{}'.format(i%n_gpus)
        command = ("python src/main.py train "
                    "--use-char-lstm --parser-type my --model-path-base run_exp "
                    "--dynet-mem 2048 --dynet-autobatch 1 "
                    # "--dropouts {} {} "
                    # "--tag-embedding-dim {} "
                    # "--word-embedding-dim {} "
                    # "--char-embedding-dim {} "
                    # "--label-embedding-dim {} "
                    "--char-lstm-dim {} "
                    "--lstm-dim {} "
                    # "--dec-lstm-dim {} "
                    # "--attention-dim {} "
                    # "--label-hidden-dim {} "
                    "--dynet-devices {} "
                     ).format(*embeddings, device)
        Popen(command.split())
