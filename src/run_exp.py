from subprocess import Popen
import itertools

n_gpus = 8
tag_embedding = [100, 150, 200]
word_embedding = [100, 150]
char_embedding = [50, 100]
# label_embedding = [100, 150, 200]
# char_lstm = [100, 150, 200]
for i, dims in enumerate(itertools.product(tag_embedding, word_embedding, char_embedding)):
    if dims != (150, 100, 50):
        device = 'GPU:{}'.format(i%n_gpus)
        command = ("python src/main.py train "
                    "--keep-valence-value "
                    "--use-char-lstm --parser-type my --model-path-base run_exp "
                    "--dynet-mem 2048 --dynet-autobatch 1 "
                    "--dropouts 0.4 0.3 "
                    "--tag-embedding-dim {} "
                    "--word-embedding-dim {} "
                    "--char-embedding-dim {} "
                    # "--label-embedding-dim {} "
                    # "--char-lstm-dim {} "
                    # "--lstm-dim {} "
                    # "--dec-lstm-dim {} "
                    # "--attention-dim {} "
                    # "--label-hidden-dim {} "
                    "--dynet-devices {} "
                     ).format(*dims, device)
        Popen(command.split())
