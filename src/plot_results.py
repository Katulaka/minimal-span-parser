import matplotlib.pyplot as plt
import pickle
import trees
import evaluate
import numpy as np
import math
import itertools


def plot_results(args):
    print("Loading test trees from {}...".format(args.test_path))
    test_treebank = trees.load_trees(args.test_path)
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    with open('results/predict_5_top_100', 'rb') as f:
        predicted_5_100 = pickle.load(f)

    with open('results/test_rank', 'rb') as f:
        test_rank = pickle.load(f)

    with open('results/test_predicted', 'rb') as f:
        predicted_5 = pickle.load(f)


    with open('results/predict_top_20_chart', 'rb') as f:
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

    chart_fscores = list(zip(*[[y.Fscore.fscore for y in x] for x in chart_evalb]))
    chart_recall = [round(np.mean([int(100.0 in y[:(k+1)]) for y in chart_fscores]), 3)*100. for k in range(20)]

    # plt.style.use('seaborn-pastel')
    fig, ax = plt.subplots()
    delta = 2
    outline_b, = plt.plot(range(1, 21), beam_5_recall, ':', label='This work')
    outline_c, = plt.plot(range(1, 21), chart_recall, label='Stern et al. \n (2017)')
    min_y = min(chart_recall + beam_5_recall) - delta
    max_y = max(chart_recall + beam_5_recall) + delta
    plt.axis([0, 21, min_y, max_y])
    ax.set_yticklabels(['{:d}%'.format(int(x)) for x in ax.get_yticks()])
    plt.xlabel('Top-K')
    plt.ylabel('Percentage excat match')
    plt.title('Top-K by Percentage excat match')
    plt.legend(loc='upper left')
    plt.savefig('top_k_vs_match.png', bbox_inches='tight')


    fig, ax = plt.subplots()
    test_rank_sum_0 = [sum(r[0]) for r in test_rank]
    beam_5_fscores_0 = [b[0] for b in beam_5_fscores]
    plt.axis([-1, 101, -1, 30])
    plt.scatter( beam_5_fscores_0, test_rank_sum_0, s=0.2)
    plt.xlabel('F1 score')
    plt.ylabel('Rank sum')
    plt.title('F1 score by Rank sum')
    plt.savefig('fscore_vs_rank_sum.png', bbox_inches='tight')

    fig, ax = plt.subplots()
    test_rank_avg_0 = [np.mean(r[0]) for r in test_rank]
    plt.axis([-1, 101, -0.02, 1])
    plt.scatter( beam_5_fscores_0, test_rank_avg_0, s=0.2)
    plt.xlabel('F1 score')
    plt.ylabel('Rank average')
    plt.title('F1 score by Rank average')
    plt.savefig('fscore_vs_rank_avg.png', bbox_inches='tight')

    fig, ax = plt.subplots()
    beam_5_0_fscore = [y[0] for y in beam_5_fscores]
    chart_0_fscore = [y[0] for y in chart_fscores]
    min_x = math.floor(min(beam_5_0_fscore+chart_0_fscore))
    bins = np.linspace(min_x, 100, 40)
    plt.hist(beam_5_0_fscore,
            bins = bins,
            density = True,
            cumulative = -1,
            histtype = 'step',
            linestyle = ':',
            edgecolor=outline_b.get_color(),
            label = 'This work')
    plt.hist(chart_0_fscore,
            bins = bins,
            density = True,
            cumulative = -1,
            linestyle = 'solid',
            histtype = 'step',
            edgecolor=outline_c.get_color(),
            label = 'Stern et al. \n (2017)')
    # plt.xticks(bins, rotation='vertical', fontsize=6)
    plt.legend(loc='lower left')
    plt.xlabel('F1 score')
    plt.ylabel('Density')
    plt.ylim((0,1.05))
    plt.title('F1 scores by Density')
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
                ha='center', va='bottom', fontsize=8)

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
                    label = 'This work',
                    hatch = '\\\\',
                    edgecolor = outline_b.get_color())
    rects2 = plt.bar(xvals + width,
                    zvals,
                    width,
                    alpha = opacity,
                    facecolor = 'none',
                    label = 'Stern et al. \n (2017)',
                    hatch = '//',
                    edgecolor=outline_c.get_color())

    plt.xlabel('Sentence length')
    plt.ylabel('F1 score')
    plt.title('F1 scores by Sentence length')
    plt.xticks(xvals + 1.5 * width, xvals)
    plt.legend(loc='lower left')

    autolabel(ax, rects1)
    autolabel(ax, rects2)

    plt.tight_layout()
    plt.savefig('fscore_vs_sentence_len.png', bbox_inches='tight')

    predicted_5_100 = list(zip(*[helper(p, 100) for p in predicted_5_100]))
    beam_5_100_evalb = [evaluate.evalb_full(args.evalb_dir, test_treebank, predicted_5_100[k]) for k in range(100)]
    beam_5_100_fscores = list(zip(*[[y.Fscore.fscore for y in x] for x in beam_5_100_evalb]))
    beam_5_100_recall = [round(np.mean([int(100.0 in y[:(k+1)]) for y in beam_5_100_fscores]), 3)*100. for k in range(100)]
    import pdb; pdb.set_trace()
