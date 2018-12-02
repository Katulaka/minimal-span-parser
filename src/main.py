import argparse
import itertools
import math
import os.path
import pickle
import time
from subprocess import Popen, DEVNULL, PIPE
# from pycrayon import CrayonClient

import dynet as dy
import matplotlib.pyplot as plt
import numpy as np

import evaluate
import parse
import trees
import vocabulary

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

def get_dependancies(fin, path_penn="src/pennconverter.jar"):
    """ Creates dependancy dictionary for each intput file"""

    command = 'java -jar {} < {} -splitSlash=false'.format(path_penn, fin)
    proc = Popen(command, shell=True, stdout=PIPE)
    results = proc.stdout.readlines()
    dependancies = []
    dependancy = []
    for res in results:
        res = res.decode('utf8')
        if res == '\n':
            dependancies.append(dependancy)
            dependancy = []
        else:
            dependancy.append(int(res.split()[6]))
    return dependancies

def run_train(args):
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)

    print("Loading training trees from {}...".format(args.train_path))
    train_treebank = trees.load_trees(args.train_path)
    print("Loaded {:,} training examples.".format(len(train_treebank)))

    print("Loading development trees from {}...".format(args.dev_path))
    dev_treebank = trees.load_trees(args.dev_path)
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    print("Processing trees for training...")
    if args.parser_type != 'my':
        train_parse = [tree.convert() for tree in train_treebank]
    else:
        dependancies = get_dependancies(args.train_path)
        train_parse = [tree.myconvert(dep)(args.keep_valence_value)
                            for tree, dep in zip(train_treebank, dependancies)]
        print("Processing trees for development...")
        dependancies = get_dependancies(args.dev_path)
        dev_parse = [tree.myconvert(dep)(args.keep_valence_value)
                            for tree, dep in zip(dev_treebank, dependancies)]

    print("Constructing vocabularies...")

    tag_vocab = vocabulary.Vocabulary()
    tag_vocab.index(parse.START)
    tag_vocab.index(parse.STOP)

    word_vocab = vocabulary.Vocabulary()
    word_vocab.index(parse.START)
    word_vocab.index(parse.STOP)
    word_vocab.index(parse.UNK)

    # if args.parser_type == 'my':
    char_vocab = vocabulary.Vocabulary()
    char_vocab.index(parse.START)
    char_vocab.index(parse.STOP)
    for c in parse.START+parse.STOP+parse.UNK:
        char_vocab.index(c)

    label_vocab = vocabulary.Vocabulary()
    if args.parser_type != 'my':
        label_vocab.index(())
    else:
        label_vocab.index(parse.START)
        label_vocab.index(parse.STOP)
        if args.keep_valence_value:
            for tree in dev_parse:
                nodes = [tree]
                while nodes:
                    node = nodes.pop()
                    if isinstance(node, trees.InternalMyParseNode):
                        nodes.extend(reversed(node.children))
                    else:
                        for l in node.labels:
                            label_vocab.index(l)

    for tree in train_parse:
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalParseNode):
                label_vocab.index(node.label)
                nodes.extend(reversed(node.children))
            elif isinstance(node, trees.InternalMyParseNode):
                nodes.extend(reversed(node.children))
            else:
                tag_vocab.index(node.tag)
                word_vocab.index(node.word)
                for c in node.word:
                    char_vocab.index(c)
                if args.parser_type == 'my':
                    for l in node.labels:
                        label_vocab.index(l)

    tag_vocab.freeze()
    word_vocab.freeze()
    label_vocab.freeze()
    char_vocab.freeze()

    def print_vocabulary(name, vocab):
        special = {parse.START, parse.STOP, parse.UNK}
        print("{} ({:,}): {}".format(
            name, vocab.size,
            sorted(value for value in vocab.values if value in special) +
            sorted(value for value in vocab.values if value not in special)))

    if args.print_vocabs:
        print_vocabulary("Tag", tag_vocab)
        print_vocabulary("Word", word_vocab)
        print_vocabulary("Label", label_vocab)

    print("Initializing model...")
    model = dy.ParameterCollection()
    if args.parser_type == "my":
        if args.model_path_base == 'run_exp':
            args.model_path_base = ('models_grid_search/'
                'char-dim({})_'
                'tag-dim({})_'
                'word-dim({})_'
                'label-dim({})_'
                'char-h({})_'
                'word-h({})_'
                'label-h({})_'
                'attention-dim({})_'
                'projection-dim({})_'
                'dropouts({})_'
                'keep_valence_value_'
                ).format(
                    args.char_embedding_dim,
                    args.tag_embedding_dim,
                    args.word_embedding_dim,
                    args.label_embedding_dim,
                    args.char_lstm_dim,
                    args.lstm_dim,
                    args.dec_lstm_dim,
                    args.attention_dim,
                    args.label_hidden_dim,
                    args.dropouts
                    )
        parser = parse.MyParser(
            model,
            tag_vocab,
            word_vocab,
            char_vocab,
            label_vocab,
            args.use_char_lstm,
            args.tag_embedding_dim,
            args.word_embedding_dim,
            args.char_embedding_dim,
            args.label_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.char_lstm_dim,
            args.dec_lstm_dim,
            args.attention_dim,
            args.label_hidden_dim,
            args.keep_valence_value,
            args.dropouts
        )
    elif args.parser_type == "top-down":
        parser = parse.TopDownParser(
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            args.tag_embedding_dim,
            args.word_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.label_hidden_dim,
            args.split_hidden_dim,
            args.dropout,
        )
    else:
        parser = parse.ChartParser(
            model,
            tag_vocab,
            word_vocab,
            char_vocab,
            label_vocab,
            args.use_char_lstm,
            args.tag_embedding_dim,
            args.word_embedding_dim,
            args.char_embedding_dim,
            args.lstm_layers,
            args.lstm_dim,
            args.char_lstm_dim,
            args.label_hidden_dim,
            args.dropout,
        )
    trainer = dy.AdamTrainer(model)

    total_processed = 0
    current_processed = 0
    check_every = len(train_parse) / args.checks_per_epoch
    best_dev_fscore = -np.inf
    best_dev_model_path = None
    best_dev_loss = np.inf

    start_time = time.time()

    def my_check_dev():
        nonlocal best_dev_loss
        nonlocal best_dev_model_path

        dev_start_time = time.time()

        total_losses = []
        for start_index in range(0, len(dev_parse), args.batch_size):
            dy.renew_cg()
            batch_losses = []
            for tree in dev_parse[start_index:start_index + args.batch_size]:
                sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
                losses = parser.parse(sentence, tree, True)
                batch_losses.extend(losses)
            batch_loss = dy.average(batch_losses)
            total_losses.append(batch_loss.scalar_value())

            print(
                "batch {:,}/{:,} "
                "batch-loss {:.4f} "
                "dev-elapsed {} "
                "total-elapsed {}".format(
                    start_index // args.batch_size + 1,
                    int(np.ceil(len(dev_parse) / args.batch_size)),
                    total_losses[-1],
                    format_elapsed(dev_start_time),
                    format_elapsed(start_time),
                )
            )

        dev_loss = np.mean(total_losses)
        print(
            "dev-loss {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_loss,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        if dev_loss < best_dev_loss:
            if best_dev_model_path is not None:
                for ext in [".data", ".meta"]:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_loss = dev_loss
            best_dev_model_path = "{}_dev={:.4f}".format(
                args.model_path_base, dev_loss)
            print("Saving new best model to {}...".format(best_dev_model_path))
            dy.save(best_dev_model_path, [parser])

        return dev_loss


    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path

        dev_start_time = time.time()

        dev_predicted = []
        for tree in dev_treebank:
            dy.renew_cg()
            sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
            predicted, _ = parser.parse(sentence)
            dev_predicted.append(predicted.convert())

        dev_fscore = evaluate.evalb(args.evalb_dir, dev_treebank, dev_predicted)

        print(
            "dev-fscore {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_fscore,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )

        if dev_fscore.fscore > best_dev_fscore:
            if best_dev_model_path is not None:
                for ext in [".data", ".meta"]:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_fscore = dev_fscore.fscore
            best_dev_model_path = "{}_dev={:.2f}".format(
                args.model_path_base, dev_fscore.fscore)
            print("Saving new best model to {}...".format(best_dev_model_path))
            dy.save(best_dev_model_path, [parser])

    # Connect to the server
    # cc = CrayonClient()

    # cc.remove_all_experiments()
    # model_name = args.model_path_base.split('/')[-1]
    # try:
    #     for name in ['train-'+model_name, 'dev-'+model_name]:
    #         cc.remove_experiment(name)
    # except:
    #     print('No experiments to remove')
    # #Create a new experiment
    # train_exp = cc.create_experiment('train-'+model_name)
    # dev_exp = cc.create_experiment('dev-'+model_name)

    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break

        np.random.shuffle(train_parse)
        epoch_start_time = time.time()

        for start_index in range(0, len(train_parse), args.batch_size):
            dy.renew_cg()

            batch_losses = []
            for tree in train_parse[start_index:start_index + args.batch_size]:
                sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
                if args.parser_type == "my":
                    losses = parser.parse(sentence, tree)
                    batch_losses.extend(losses)
                elif args.parser_type == "top-down":
                    _, loss = parser.parse(sentence, tree, args.explore)
                    batch_losses.append(loss)
                else:
                    _, loss = parser.parse(sentence, tree)
                    batch_losses.append(loss)
                total_processed += 1
                current_processed += 1

            batch_loss = dy.average(batch_losses)
            batch_loss_value = batch_loss.scalar_value()
            # train_exp.add_scalar_value("loss", batch_loss_value)

            batch_loss.backward()
            trainer.update()

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {} ".format(
                    epoch,
                    start_index // args.batch_size + 1,
                    int(np.ceil(len(train_parse) / args.batch_size)),
                    total_processed,
                    batch_loss_value,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )

            if current_processed >= check_every:
                current_processed -= check_every
                if args.parser_type == "my":
                    dev_loss = my_check_dev()
                    # step = int(np.ceil(total_processed/args.batch_size))
                    # dev_exp.add_scalar_value("loss", dev_loss, step=step)
                else:
                    check_dev()

def run_test(args):
    print("Loading test trees from {}...".format(args.test_path))
    test_treebank = trees.load_trees(args.test_path)
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    print("Loading model from {}...".format(args.model_path_base))
    model = dy.ParameterCollection()
    [parser] = dy.load(args.model_path_base, model)

    print("Parsing test sentences...")

    start_time = time.time()
    test_predicted = []
    if args.parser_type == "my":
        test_rank = []
        astar_parms = [args.n_trees, args.time_out, args.n_discounts, args.discount_factor]
        predict_parms = {'astar_parms' : astar_parms, 'beam_parms' : args.beam_size}

    for i, tree in  enumerate(test_treebank):
        dy.renew_cg()
        sentence = [(leaf.tag, leaf.word) for leaf in tree.leaves()]
        prediction_start_time = time.time()
        if args.parser_type == "my":
            predicted, ranks = parser.parse(sentence, predict_parms=predict_parms)
        else:
            predicted, _ = parser.parse(sentence, k = args.n_trees)
        print(
            "processed {:,}/{:,} "
            "prediction-elapsed {} "
            "total-elapsed {}".format(
                i + 1,
                len(test_treebank),
                format_elapsed(prediction_start_time),
                format_elapsed(start_time),
            )
        )

        if isinstance(predicted, list):
            test_predicted.append([p.convert() for p in predicted])
        else:
            test_predicted.append(predicted.convert())

        if args.parser_type == "my":
            test_rank.append(ranks)

    if args.n_trees == 1:
        test_fscore = evaluate.evalb(args.evalb_dir, test_treebank, test_predicted)

    # if args.parser_type == "my":
    #     test_rank_fname = 'test_rank_{}'.format(args.n_trees)
    #     with open(test_rank_fname, 'wb') as f:
    #         pickle.dump(test_rank, f)
    #
    # test_predicted_fname = 'test_predicted_{}'.format(args.n_trees)
    # with open(test_predicted_fname ,'wb') as f:
    #     pickle.dump(test_predicted, f)

    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )
    import pdb; pdb.set_trace()

def run_print_results(args):
    print("Loading test trees from {}...".format(args.test_path))
    test_treebank = trees.load_trees(args.test_path)
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    with open('predict_5_top_20', 'rb') as f:
        predicted_5 = pickle.load(f)

    with open('predict_top_20_chart', 'rb') as f:
        predicted_chart = pickle.load(f)

    def helper(predict, k):
        if isinstance(predict, trees.InternalTreebankNode):
            return [predict] * k
        return predict + [predict[-1]] * (k - len(predict))

    predicted_5 = list(zip(*[helper(p, 20) for p in predicted_5]))
    beam_5_evalb = [evaluate.evalb_full(args.evalb_dir, test_treebank, predicted_5[k]) for k in range(20)]

    predicted_chart = list(zip(*predicted_chart))
    chart_evalb = [evaluate.evalb_full(args.evalb_dir, test_treebank, predicted_chart[k]) for k in range(20)]

    beam_5_fscores = list(zip(*[[y.Fscore.fscore for y in x] for x in beam_5_evalb]))
    beam_5_recall = [round(np.mean([int(100.0 in y[:(k+1)]) for y in beam_5_fscores]), 3)*100. for k in range(20)]
    # beam_5_recall = [evaluate.recall(test_treebank, predicted_5, k) for k in range(1, 21)]

    chart_fscores = list(zip(*[[y.Fscore.fscore for y in x] for x in chart_evalb]))
    chart_recall = [round(np.mean([int(100.0 in y[:(k+1)]) for y in chart_fscores]), 3)*100. for k in range(20)]
    # chart_recall = [evaluate.recall(test_treebank, predicted_chart, k) for k in range(1, 21)]

    # plt.style.use('seaborn-pastel')
    fig, ax = plt.subplots()
    delta = 2
    outline_b, = plt.plot(range(1, 21), beam_5_recall, ':', label='beam 5')
    outline_c, = plt.plot(range(1, 21), chart_recall, label='chart')
    min_y = min(chart_recall + beam_5_recall) - delta
    max_y = max(chart_recall + beam_5_recall) + delta
    plt.axis([0, 21, min_y, max_y])
    ax.set_yticklabels(['{:0.2f}%'.format(x) for x in ax.get_yticks()])
    plt.xlabel('Top-K')
    plt.ylabel('Percentage excat match')
    plt.title('Top-K by Percentage excat match')
    plt.legend(loc='upper left')
    plt.savefig('top_k_vs_match.png', bbox_inches='tight')

    fig, ax = plt.subplots()
    beam_5_max_fscore = [y[0] for y in beam_5_fscores]
    chart_max_fscore = [y[0] for y in chart_fscores]
    min_x = math.floor(min(beam_5_max_fscore+chart_max_fscore))
    bins = np.linspace(min_x, 100, 40)
    plt.hist([beam_5_max_fscore, chart_max_fscore],
            bins = bins,
            density = True,
            cumulative = -1,
            histtype = 'step',
            linestyle = ':',
            edgecolor=outline_b.get_color(),
            label = 'beam 5')
    plt.hist(chart_max_fscore,
            bins = bins,
            density = True,
            cumulative = -1,
            linestyle = 'solid',
            histtype = 'step',
            edgecolor=outline_c.get_color(),
            label = 'chart')
    # plt.xticks(bins, rotation='vertical', fontsize=6)
    plt.legend(loc='lower left')
    plt.xlabel('F-score')
    plt.ylabel('Density')
    plt.ylim((0,1.05))
    plt.title('F-Scores by Density')
    plt.savefig('fscore_vs_density.png', bbox_inches='tight')

    hist_5 = {}
    test_predicted_5 = sorted(beam_5_evalb[0], key = lambda x: x.length)
    for key, value in itertools.groupby(test_predicted_5, lambda x: int(x.length/10) * 10):
        value = list(value)
        match_bracket = sum((h.match_bracket for h in value))
        gold_bracket = sum((h.gold_bracket for h in value))
        test_bracket = sum((h.test_bracket for h in value))
        hist_5[key] = evaluate.Bracket(key, match_bracket, gold_bracket, test_bracket).Fscore.fscore

    hist_chart = {}
    test_predicted_chart = sorted(chart_evalb[0], key = lambda x: x.length)
    for key, value in itertools.groupby(test_predicted_chart, lambda x: int(x.length/10) * 10):
        value = list(value)
        match_bracket = sum((h.match_bracket for h in value))
        gold_bracket = sum((h.gold_bracket for h in value))
        test_bracket = sum((h.test_bracket for h in value))
        hist_chart[key] = evaluate.Bracket(key, match_bracket, gold_bracket, test_bracket).Fscore.fscore

    def autolabel(ax, rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.01 * h, '%.2f'%float(h),
                ha='center', va='bottom', fontsize=6)

    fig, ax = plt.subplots()

    xvals, yvals = zip(*sorted(hist_5.items()))
    _, zvals = zip(*sorted(hist_chart.items()))
    xvals = np.array(xvals) + 10

    width = 4.5
    opacity = 0.8

    rects1 = plt.bar(xvals,
                    yvals,
                    width,
                    alpha = opacity,
                    facecolor = 'none',
                    label = 'Beam 5',
                    hatch = '\\\\',
                    edgecolor = outline_b.get_color())
    rects2 = plt.bar(xvals + width,
                    zvals,
                    width,
                    alpha = opacity,
                    facecolor = 'none',
                    label = 'Chart',
                    hatch = '//',
                    edgecolor=outline_c.get_color())

    plt.xlabel('Sentence length')
    plt.ylabel('F-score')
    plt.title('F-Scores by Sentence length')
    plt.xticks(xvals + 1.5 * width, xvals)
    plt.legend(loc='lower left')

    autolabel(ax, rects1)
    autolabel(ax, rects2)

    plt.tight_layout()
    plt.savefig('fscore_vs_sentence_len.png', bbox_inches='tight')

    # with open('predict_5', 'rb') as f:
    #     test_predicted_5 = pickle.load(f)
    #
    # with open('predict_chart', 'rb') as f:
    #     test_predicted_chart = pickle.load(f)
    #
    # hist_5, hist_chart = {}, {}
    # ranges = [(l,u) for l, u in zip(range(0,70,10), range(10,80,10))]
    # for gold, tree_5, tree_chart in zip(test_treebank, test_predicted_5, test_predicted_chart):
    #     num_leaves = len(list(tree_5.leaves()))
    #     for key in ranges:
    #         if num_leaves in range(*key):
    #             hist_5.setdefault(key[1],[]).append((gold, tree_5))
    #             hist_chart.setdefault(key[1],[]).append((gold, tree_chart))
    #             break
    #
    # xvals = np.array(sorted(hist_5.keys()))
    # yvals = [evaluate.evalb(args.evalb_dir, *zip(*v)).fscore for k, v in sorted(hist_5.items())]
    # zvals = [evaluate.evalb(args.evalb_dir, *zip(*v)).fscore for k, v in sorted(hist_chart.items())]
    #
    # fig, ax = plt.subplots()
    #
    # rects1 = plt.bar(xvals, yvals, width, alpha=opacity, label='Beam 5')
    # rects2 = plt.bar(xvals + width, zvals, width, alpha=opacity, label='Chart')
    #
    # plt.xlabel('Sentence length')
    # plt.ylabel('F-score')
    # plt.title('F-Scores by Sentence length')
    # plt.xticks(xvals + 1.5 * width, xvals)
    # plt.legend(loc='lower left')
    #
    # autolabel(ax, rects1)
    # autolabel(ax, rects2)
    #
    # plt.tight_layout()
    # # plt.savefig('foo3.png', bbox_inches='tight')

    # plt.show()
    # import pdb; pdb.set_trace()


def main():
    dynet_args = [
        "--dynet-mem",
        "--dynet-weight-decay",
        "--dynet-autobatch",
        "--dynet-gpus",
        "--dynet-gpu",
        "--dynet-devices",
        "--dynet-seed",
    ]

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=run_train)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--parser-type", choices=["top-down", "chart", "my"], required=True)
    subparser.add_argument("--tag-embedding-dim", type=int, default=150)
    subparser.add_argument("--word-embedding-dim", type=int, default=100)
    subparser.add_argument("--char-embedding-dim", type=int, default=50)
    subparser.add_argument("--label-embedding-dim", type=int, default=100)
    subparser.add_argument("--char-lstm-dim", type=int, default=100)
    subparser.add_argument("--lstm-dim", type=int, default=350)
    subparser.add_argument("--dec-lstm-dim", type=int, default=600)
    subparser.add_argument("--label-hidden-dim", type=int, default=100)
    subparser.add_argument("--attention-dim", type=int, default=200)
    subparser.add_argument("--lstm-layers", type=int, default=2)
    subparser.add_argument("--split-hidden-dim", type=int, default=250)
    subparser.add_argument("--dropout", type=float, default=0.4)
    subparser.add_argument("--dropouts", nargs='+', type=float, default=[0.4, 0.4])
    subparser.add_argument("--explore", action="store_true")
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--train-path", default="data/02-21.10way.clean")
    subparser.add_argument("--dev-path", default="data/22.auto.clean")
    subparser.add_argument("--batch-size", type=int, default=10)
    subparser.add_argument("--epochs", type=int)
    subparser.add_argument("--checks-per-epoch", type=int, default=4)
    subparser.add_argument("--print-vocabs", action="store_true")
    subparser.add_argument("--keep-valence-value", action="store_true")
    subparser.add_argument("--use-char-lstm", action="store_true")

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    for arg in dynet_args:
        subparser.add_argument(arg)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--test-path", default="data/23.auto.clean")
    subparser.add_argument("--parser-type", choices=["top-down", "chart", "my"], required=True)
    subparser.add_argument("--n-trees", default=1, type=int)
    subparser.add_argument("--time-out", default=np.inf, type=float)
    subparser.add_argument("--n-discounts", default=1, type=int)
    subparser.add_argument("--discount-factor", default=0.2, type=float)
    subparser.add_argument("--beam-size", nargs='+', default=[5], type=int)

    subparser = subparsers.add_parser("print")
    subparser.set_defaults(callback=run_print_results)
    # subparser.add_argument("--predict-path", required=True)
    subparser.add_argument("--test-path", default="data/23.auto.clean")
    subparser.add_argument("--evalb-dir", default="EVALB/")

    args = parser.parse_args()
    args.callback(args)

if __name__ == "__main__":
    main()
