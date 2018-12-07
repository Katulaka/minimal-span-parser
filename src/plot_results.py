import matplotlib.pyplot as plt
import pickle
import trees
import evaluate
import numpy as np
import math
import itertools

def helper(predict, k):
    if isinstance(predict, trees.InternalTreebankNode):
        return [predict] * k
    return predict + [predict[-1]] * (k - len(predict))

# def plot_ranks(beam_5_fscores):
#
#     with open('results/test_rank', 'rb') as f:
#         test_rank = pickle.load(f)
#
#     fig, ax = plt.subplots()
#     test_rank_sum_0 = [sum(r[0]) for r in test_rank]
#     beam_5_fscores_0 = [b[0] for b in beam_5_fscores]
#     plt.axis([-1, 101, -1, 30])
#     plt.scatter( beam_5_fscores_0, test_rank_sum_0, s=0.2)
#     plt.xlabel('F1 score')
#     plt.ylabel('Rank sum')
#     plt.title('F1 score by Rank sum')
#     plt.savefig('fscore_vs_rank_sum.png', bbox_inches='tight')
#
#     fig, ax = plt.subplots()
#     test_rank_avg_0 = [np.mean(r[0]) for r in test_rank]
#     plt.axis([-1, 101, -0.02, 1])
#     plt.scatter( beam_5_fscores_0, test_rank_avg_0, s=0.2)
#     plt.xlabel('F1 score')
#     plt.ylabel('Rank average')
#     plt.title('F1 score by Rank average')
#     plt.savefig('fscore_vs_rank_avg.png', bbox_inches='tight')

def plot_density(our_fscore, chart_fscore, outline_b, outline_c):
    fig, ax = plt.subplots()

    min_x = math.floor(np.min([our_fscore, chart_fscore]))
    bins = np.linspace(min_x, 100, 40)
    plt.hist(our_fscore,
            bins = bins,
            density = True,
            cumulative = -1,
            histtype = 'step',
            linestyle = ':',
            edgecolor=outline_b.get_color(),
            label = 'This work')
    plt.hist(chart_fscore,
            bins = bins,
            density = True,
            cumulative = -1,
            linestyle = 'solid',
            histtype = 'step',
            edgecolor=outline_c.get_color(),
            label = 'Stern et al. \n (2017)')
    plt.legend(loc='lower left')
    plt.xlabel('F1 score')
    plt.ylabel('Density')
    plt.ylim((0,1.05))
    plt.title('F1 scores vs. Density')
    plt.savefig('plots/fscore_vs_density.png', bbox_inches='tight')

def plot_bars(hist_beam, hist_chart, outline_b, outline_c, y_label, f_name, loc='lower left'):

    def autolabel(ax, rects):
        for rect in rects:
            h = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.01 * h, '%.2f'%float(h),
                ha='center', va='bottom', fontsize=8)

    fig, ax = plt.subplots()

    xvals, yvals = zip(*sorted(hist_beam.items()))
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

    autolabel(ax, rects1)
    autolabel(ax, rects2)
    ax.set_yticklabels(['{:d}%'.format(int(x)) for x in ax.get_yticks()])

    plt.xlabel('Sentence length')
    plt.ylabel(y_label)
    plt.title('{} vs. Sentence length'.format(y_label))
    plt.xticks(xvals + 1.5 * width, xvals)
    plt.legend(loc=loc)

    plt.tight_layout()
    plt.savefig(f_name, bbox_inches='tight')

def plot_exact_match(beam_recall, chart_recall, n_trees):
    fig, ax = plt.subplots()
    delta = 2
    ax.plot(range(1, n_trees+1), beam_recall, '>:', label='This work')
    ax.plot(range(1, n_trees+1), chart_recall, '--d', label='Stern et al. \n (2017)')
    min_y = np.min([chart_recall, beam_recall]) - delta
    max_y = np.max([chart_recall, beam_recall]) + delta
    ax.axis([0, n_trees+1, min_y, max_y])
    ax.set_yticklabels(['{:d}%'.format(int(x)) for x in ax.get_yticks()])
    plt.xlabel('Top-K')
    plt.ylabel('Exact match')
    plt.title('Exact match vs. Top-K')
    plt.legend(loc='upper left')
    plt.savefig('plots/top_k_vs_match.png', bbox_inches='tight')
    return ax


def compute(args, test_treebank, predicted, n_trees, s_len=10):

    predict = np.array([helper(p, n_trees) for p in predicted])
    evalb = [evaluate.evalb_full(args.evalb_dir, test_treebank, predict[:,k]) \
                                    for k in range(n_trees)]

    fscores = np.array([[y.Fscore.fscore for y in x] for x in evalb]).transpose()

    match = np.array([[100.0 in y[:(k+1)] for y in fscores].count(True) \
                    for k in range(n_trees)])

    recall = match/len(test_treebank)*100

    hist_fscore = {}
    pair_sort = sorted(zip(test_treebank, predict[:,0]),\
                                    key = lambda x: len(list(x[1].leaves())))
    iter = itertools.groupby(pair_sort, \
                lambda x: int(len(list(x[1].leaves()))/s_len) * s_len)
    for key, value in iter:
        hist_fscore[key] = evaluate.evalb(args.evalb_dir, *zip(*list(value))).fscore

    hist_recall = {}
    evalb_sort = sorted(np.array(evalb).transpose() , key = lambda x: x[0].length)
    iter = itertools.groupby(evalb_sort, lambda x: int(x[0].length/s_len) * s_len)
    for key, value in iter:
        match = [100.0 == y[0].Fscore.fscore for y in value]
        hist_recall[key] = match.count(True)/len(match) * 100


    hist = {'fscore': hist_fscore,
            'recall': hist_recall,
            }
    return fscores, recall, hist

def plot_results(args):
    print("Loading test trees from {}...".format(args.test_path))
    test_treebank = trees.load_trees(args.test_path)
    print("Loaded {:,} test examples.".format(len(test_treebank)))

    # for i in ['beam_5','beam_10','beam_15','beam_20','beam_25','beam_32']:
    #     fname = 'results/lp_0_7_5/predicted_{}_lp_0.7_5_to_2'.format(i)
    #     with open(fname,  'rb') as f:
    #         predicted = pickle.load(f)
    #
    #     fscore = evaluate.evalb(args.evalb_dir, test_treebank, predicted)
    #     evalb = evaluate.evalb_full(args.evalb_dir, test_treebank, predicted)
    #     match = [100 == x.Fscore.fscore for x in evalb]
    #     recall = match.count(True)/len(match)*100
    #     print('Parser:{}, fscore:{}, recall:{}'.format(i,fscore,recall))
    #

    # with open('results/lp_0_7_5/predicted_beam_10_lp_0.7_5_top_20', 'rb') as f:
    #     our_predicted_lp = pickle.load(f)

    our_predicted = []
    for i in range(8):
        fname = 'results/predicted_beam_10/predicted_beam_10_top_20_{}_{}'.format(302*i, 302*(i+1))
        with open(fname,  'rb') as f:
            our_predicted.extend(pickle.load(f))

    with open('results/predict_top_20_chart', 'rb') as f:
        chart_predicted = pickle.load(f)

    n_trees = 20
    our_fscores, our_recall, hist_our = compute(args, test_treebank, our_predicted, n_trees)
    chart_fscores, chart_recall, hist_chart = compute(args, test_treebank, chart_predicted, n_trees)

    ax = plot_exact_match(our_recall, chart_recall, n_trees)
    f_name = 'plots/fscore_vs_sentence_len.png'
    y_label = 'F1 score'
    hist_fscore = [hist_our['fscore'], hist_chart['fscore']]
    plot_bars(*hist_fscore, *ax.lines, y_label, f_name)
    loc = 'upper right'
    f_name = 'plots/match_vs_sentence_len.png'
    y_label = 'Exact match Top-1'
    hist_recall = [hist_our['recall'], hist_chart['recall']]
    plot_bars(*hist_recall, *ax.lines, y_label, f_name, loc)

    import pdb; pdb.set_trace()
