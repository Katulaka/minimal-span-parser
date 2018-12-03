from subprocess import Popen
import itertools

n_gpus = 8
tag_embedding = [100, 150, 200]
word_embedding = [100, 150]
char_embedding = [50, 100]
label_embedding = [250, 300]
label_dim = [500, 600, 700]
for i, dims in enumerate(itertools.product(label_embedding, label_dim)):
    if dims != (100, 600):
        device = 'GPU:{}'.format(i%n_gpus)
        command = ("python src/main.py train "
                    "--keep-valence-value "
                    "--use-char-lstm --parser-type my --model-path-base run_exp "
                    "--dynet-mem 2048 --dynet-autobatch 1 "
                    "--dropouts 0.4 0.3 "
                    "--tag-embedding-dim 150 "
                    "--word-embedding-dim 150 "
                    "--char-embedding-dim 50 "
                    "--label-embedding-dim {} "
                    # "--char-lstm-dim {} "
                    # "--lstm-dim {} "
                    "--dec-lstm-dim {} "
                    # "--attention-dim {} "
                    # "--label-hidden-dim {} "
                    "--dynet-devices {} "
                     ).format(*dims, device)
        Popen(command.split())
