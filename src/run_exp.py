from subprocess import Popen
import itertools

n_gpus = 8
for i, dropouts in  enumerate(itertools.product([0.1, 0.2, 0.3, 0.4, 0.5], [0, 0.1, 0.2, 0.3, 0.4, 0.5])):
    if dropouts!=(0.4, 0.4):
        device = 'GPU:{}'.format(i%n_gpus)
        command = ("python src/main.py train "
                    "--keep-valence-value "
                    "--use-char-lstm --parser-type my --model-path-base run_exp "
                    "--dynet-mem 2048 --dynet-autobatch 1 "
                    "--dropouts {} {} "
                    # "--tag-embedding-dim {} "
                    # "--word-embedding-dim {} "
                    # "--char-embedding-dim {} "
                    # "--label-embedding-dim {} "
                    # "--char-lstm-dim {} "
                    # "--lstm-dim {} "
                    # "--dec-lstm-dim {} "
                    # "--attention-dim {} "
                    # "--label-hidden-dim {} "
                    "--dynet-devices {} "
                     ).format(*dropouts, device)
    Popen(command.split())
