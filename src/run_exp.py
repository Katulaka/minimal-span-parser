from subprocess import Popen
import itertools

n_gpus = 8
label_embedding = [100, 150, 200]
char_lstm = [100, 150, 200]
for i, dims in enumerate(itertools.product(label_embedding, char_lstmg)):
    if dims != (100, 100):
        device = 'GPU:{}'.format(i%n_gpus)
        command = ("python src/main.py train "
                    "--use-char-lstm --parser-type my --model-path-base run_exp "
                    "--dynet-mem 2048 --dynet-autobatch 1 "
                    # "--dropouts {} {} "
                    "--tag-embedding-dim 200 "
                    "--word-embedding-dim 100 "
                    "--char-embedding-dim 50 "
                    "--label-embedding-dim {} "
                    "--char-lstm-dim {} "
                    # "--lstm-dim {} "
                    # "--dec-lstm-dim {} "
                    # "--attention-dim {} "
                    # "--label-hidden-dim {} "
                    "--dynet-devices {} "
                     ).format(*dims, device)
        Popen(command.split())
