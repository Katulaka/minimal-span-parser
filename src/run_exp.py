from subprocess import Popen
import itertools

n_gpus = 8
tag_embedding = [150, 200]
word_embedding = [100, 150]
char_embedding = [50, 100]
for i, embeddings in enumerate(itertools.product(tag_embedding, word_embedding, char_embedding)):
    if embeddings != (150, 100, 50):
        device = 'GPU:{}'.format(i%n_gpus)
        command = ("python src/main.py train "
                    "--use-char-lstm --parser-type my --model-path-base run_exp "
                    "--dynet-mem 2048 --dynet-autobatch 1 "
                    # "--dropouts {} {} "
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
                     ).format(*embeddings, device)
    Popen(command.split())
