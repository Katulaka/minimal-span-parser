import os, sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np

import parse_nk

def torch_load(load_path):
    if parse_nk.use_cuda:
        return torch.load(load_path)
    else:
        return torch.load(load_path, map_location=lambda storage, location: storage)

class Rescorer:
    def __init__(self, model_path_base):
        print("Loading model from {}...".format(model_path_base))
        assert model_path_base.endswith(".pt"), "Only pytorch savefiles supported"
        info = torch_load(model_path_base)
        assert 'hparams' in info['spec'], "Older savefiles not supported"
        self.parser = parse_nk.NKChartParser.from_spec(info['spec'], info['state_dict'])

    def label_index(self, label):
        """
        label: a tuple. For unary chains, the tuple ordering is:
            (TOPMOST, CHILD_OF_TOPMOST, ...), i.e. topmost label in the unary
            chain goes first
        """
        # Note that if a unary chain is not part of the label vocabulary, this
        # returns not-a-constituent
        return self.parser.label_vocab.index_or_unk(value, ())

    def precompute(self, sentence):
        """
        Precompute the parse chart for a single sentence

        sentence: list of words, or list of (tag, word) pairs
        """
        if not isinstance(sentence[0], tuple):
            # None of the parsers actually need tags, so plug in some dummy tags
            # for now
            sentence = list(zip(['NN'] * len(sentence), sentence))
        charts = self.parser.parse_batch([sentence], return_label_scores_charts=True)
        chart = charts[0]
        return Precomputed(chart)

class Precomputed:
    def __init__(self, chart):
        self.chart = chart
        # The root node is not allowed to have the null label
        chart[0, -1, 0] = -np.inf
        self.inside_scores, self.outside_scores = decode(self.chart)
        self.nolabel_chart = self.chart.max(-1)

    def constituent_score(self, left, right, label_idx):
        """
        The score for a constituent spanning fenceposts left...right, with label
        index label_idx. Use Rescorer.label_index to convert labels to label
        indices
        """
        return self.chart[left, right, label_idx]

    def heuristic_1(self, maximal_subtree_fenceposts):
        """
        A heuristic upper bound on how much the chart-parser score can increase
        when going from the current partial tree to a full parse tree.

        maximal_subtree_fenceposts is a list/tuple of integer fencepost
        positions [f_0, f_1, ...] where the spans f_0...f_1, f_1...f_2, etc. are
        maximum-size subtrees in the current derivation. These subtrees must be
        maximal, i.e the spans f_0...f_2, f_1...f_3, etc. must NOT be subtrees
        """
        assert len(maximal_subtree_fenceposts) >= 2, (
            "maximal_subtree_fenceposts must be at least length 2: "
            "1-word subtrees are allowed, so there should always be at least "
            "one subtree, which will have both a startpoint and an endpoint"
            )
        outside_score = self.outside_scores[
            maximal_subtree_fenceposts[0], maximal_subtree_fenceposts[-1]]

        if len(maximal_subtree_fenceposts) == 2:
            return outside_score

        first_fp = maximal_subtree_fenceposts[0]
        inner_fps = list(maximal_subtree_fenceposts[1:-1])
        last_fp = maximal_subtree_fenceposts[-1]

        return (
            outside_score
            + self.nolabel_chart[:first_fp, inner_fps].sum()
            + self.nolabel_chart[inner_fps, last_fp+1:].sum()
            )


def decode(label_scores_chart):
    sentence_len = label_scores_chart.shape[0]-1
    assert label_scores_chart[0, sentence_len, 0] < -1e8, (
        "This decoder assumes that a null label at the root has already been masked out"
        )

    value_chart = np.zeros((sentence_len+1, sentence_len+1), dtype=np.float32)
    split_idx_chart = np.zeros((sentence_len+1, sentence_len+1), dtype=np.int32)
    best_label_chart = np.zeros((sentence_len+1, sentence_len+1), dtype=np.int32)

    for length in range(1, sentence_len + 1):
        for left in range(0, sentence_len + 1 - length):
            right = left + length

            label_scores_for_span = label_scores_chart[left, right]
            argmax_label_index = label_scores_for_span.argmax()
            label_score = label_scores_for_span[argmax_label_index]
            best_label_chart[left, right] = argmax_label_index

            if length == 1:
                value_chart[left, right] = label_score
                continue

            best_split = left + 1
            split_val = -np.inf
            for split_idx in range(left + 1, right):
                max_split_val = value_chart[left, split_idx] + value_chart[split_idx, right]
                if max_split_val > split_val:
                    split_val = max_split_val
                    best_split = split_idx

            value_chart[left, right] = label_score + value_chart[left, best_split] + value_chart[best_split, right]
            split_idx_chart[left, right] = best_split

    outside_scores_chart = np.zeros((sentence_len+1, sentence_len+1), dtype=np.float32)
    for length in reversed(range(1, sentence_len)):
        # skip length == sentence_len: the outside score of the root is 0
        for left in range(0, sentence_len + 1 - length):
            right = left + length

            outside_score = -np.inf
            for merge_idx in range(0, left):
                outside_score = max(outside_score,
                    value_chart[merge_idx, left]
                    + label_scores_chart[merge_idx, right].max()
                    + outside_scores_chart[merge_idx, right]
                    )

            for merge_idx in range(right + 1, sentence_len + 1):
                outside_score = max(outside_score,
                    value_chart[right, merge_idx]
                    + label_scores_chart[left, merge_idx].max()
                    + outside_scores_chart[left, merge_idx])

            outside_scores_chart[left, right] = outside_score

    return value_chart, outside_scores_chart
