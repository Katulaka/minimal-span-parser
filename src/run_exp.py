from subprocess import Popen
import itertools

n_gpus = 8
# for i, embeddings in enumerate(itertools.product([200, 250, 300], [100, 150, 200])):
#     if embeddings != (250, 100):
# for i, embedding in enumerate([500,700,800,900,1000]):
for dropouts in  enumerate(itertools.product([0.4], [0, 0.1, 0.2, 0.3, 0.5])):
    device = 'GPU:{}'.format(i%n_gpus)
    command = ("python src/main.py train "
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
