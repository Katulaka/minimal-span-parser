import functools
import collections

import dynet as dy
import numpy as np

import trees
from beam.search import BeamSearch
from astar.search import astar_search
from astar.search_chart import AstarNode, Solver

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"

def augment(scores, oracle_index):
    assert isinstance(scores, dy.Expression)
    shape = scores.dim()[0]
    assert len(shape) == 1
    increment = np.ones(shape)
    increment[oracle_index] = 0
    return scores + dy.inputVector(increment)

class Feedforward(object):
    def __init__(self, model, input_dim, hidden_dims, output_dim):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Feedforward")

        self.weights = []
        self.biases = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for prev_dim, next_dim in zip(dims, dims[1:]):
            self.weights.append(self.model.add_parameters((next_dim, prev_dim)))
            self.biases.append(self.model.add_parameters(next_dim))

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def __call__(self, x):
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            weight = dy.parameter(weight)
            bias = dy.parameter(bias)
            x = dy.affine_transform([bias, weight, x])
            if i < len(self.weights) - 1:
                x = dy.rectify(x)
        return x

class TopDownParser(object):
    def __init__(
            self,
            model,
            tag_vocab,
            word_vocab,
            label_vocab,
            tag_embedding_dim,
            word_embedding_dim,
            lstm_layers,
            lstm_dim,
            label_hidden_dim,
            split_hidden_dim,
            dropout,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Parser")
        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim

        self.tag_embeddings = self.model.add_lookup_parameters(
            (tag_vocab.size, tag_embedding_dim))
        self.word_embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))

        self.lstm = dy.BiRNNBuilder(
            lstm_layers,
            tag_embedding_dim + word_embedding_dim,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.f_label = Feedforward(
            self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size)
        self.f_split = Feedforward(
            self.model, 2 * lstm_dim, [split_hidden_dim], 1)

        self.dropout = dropout

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def parse(self, sentence, gold=None, explore=True):
        is_train = gold is not None

        if is_train:
            self.lstm.set_dropout(self.dropout)
        else:
            self.lstm.disable_dropout()

        embeddings = []
        for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
            tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                    word = UNK
            word_embedding = self.word_embeddings[self.word_vocab.index(word)]
            embeddings.append(dy.concatenate([tag_embedding, word_embedding]))

        lstm_outputs = self.lstm.transduce(embeddings)

        @functools.lru_cache(maxsize=None)
        def get_span_encoding(left, right):
            forward = (
                lstm_outputs[right][:self.lstm_dim] -
                lstm_outputs[left][:self.lstm_dim])
            backward = (
                lstm_outputs[left + 1][self.lstm_dim:] -
                lstm_outputs[right + 1][self.lstm_dim:])
            return dy.concatenate([forward, backward])

        def helper(left, right):
            assert 0 <= left < right <= len(sentence)

            label_scores = self.f_label(get_span_encoding(left, right))

            if is_train:
                oracle_label = gold.oracle_label(left, right)
                oracle_label_index = self.label_vocab.index(oracle_label)
                label_scores = augment(label_scores, oracle_label_index)

            label_scores_np = label_scores.npvalue()
            argmax_label_index = int(
                label_scores_np.argmax() if right - left < len(sentence) else
                label_scores_np[1:].argmax() + 1)
            argmax_label = self.label_vocab.value(argmax_label_index)

            if is_train:
                label = argmax_label if explore else oracle_label
                label_loss = (
                    label_scores[argmax_label_index] -
                    label_scores[oracle_label_index]
                    if argmax_label != oracle_label else dy.zeros(1))
            else:
                label = argmax_label
                label_loss = label_scores[argmax_label_index]

            if right - left == 1:
                tag, word = sentence[left]
                tree = trees.LeafParseNode(left, tag, word)
                if label:
                    tree = trees.InternalParseNode(label, [tree])
                return [tree], label_loss

            left_encodings = []
            right_encodings = []
            for split in range(left + 1, right):
                left_encodings.append(get_span_encoding(left, split))
                right_encodings.append(get_span_encoding(split, right))
            left_scores = self.f_split(dy.concatenate_to_batch(left_encodings))
            right_scores = self.f_split(dy.concatenate_to_batch(right_encodings))
            split_scores = left_scores + right_scores
            split_scores = dy.reshape(split_scores, (len(left_encodings),))

            if is_train:
                oracle_splits = gold.oracle_splits(left, right)
                oracle_split = min(oracle_splits)
                oracle_split_index = oracle_split - (left + 1)
                split_scores = augment(split_scores, oracle_split_index)

            split_scores_np = split_scores.npvalue()
            argmax_split_index = int(split_scores_np.argmax())
            argmax_split = argmax_split_index + (left + 1)

            if is_train:
                split = argmax_split if explore else oracle_split
                split_loss = (
                    split_scores[argmax_split_index] -
                    split_scores[oracle_split_index]
                    if argmax_split != oracle_split else dy.zeros(1))
            else:
                split = argmax_split
                split_loss = split_scores[argmax_split_index]

            left_trees, left_loss = helper(left, split)
            right_trees, right_loss = helper(split, right)

            children = left_trees + right_trees
            if label:
                children = [trees.InternalParseNode(label, children)]

            return children, label_loss + split_loss + left_loss + right_loss

        children, loss = helper(0, len(sentence))
        assert len(children) == 1
        tree = children[0]
        if is_train and not explore:
            assert gold.convert().linearize() == tree.convert().linearize()
        return tree, loss

class ChartParser(object):
    def __init__(
            self,
            model,
            tag_vocab,
            word_vocab,
            char_vocab,
            label_vocab,
            use_char_lstm,
            tag_embedding_dim,
            word_embedding_dim,
            char_embedding_dim,
            lstm_layers,
            lstm_dim,
            char_lstm_dim,
            label_hidden_dim,
            dropout,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")

        self.model = model.add_subcollection("Parser")
        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim

        self.tag_embeddings = self.model.add_lookup_parameters(
            (tag_vocab.size, tag_embedding_dim))
        self.word_embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))

        embedding_dim = tag_embedding_dim + word_embedding_dim
        if use_char_lstm:
            self.char_vocab = char_vocab
            self.char_embeddings = self.model.add_lookup_parameters(
                (char_vocab.size, char_embedding_dim))
            self.char_lstm = dy.LSTMBuilder(
                1,
                char_embedding_dim,
                char_lstm_dim,
                self.model)
            embedding_dim += char_lstm_dim

        self.lstm = dy.BiRNNBuilder(
            lstm_layers,
            # tag_embedding_dim + word_embedding_dim,
            embedding_dim,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.f_label = Feedforward(
            self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size - 1)

        self.dropout = dropout
        self.use_char_lstm = use_char_lstm


    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def parse(self, sentence, gold=None, k=1):
        is_train = gold is not None

        if self.use_char_lstm:
            char_lstm = self.char_lstm.initial_state()
            if is_train:
                self.char_lstm.set_dropout(self.dropout)
            else:
                self.char_lstm.disable_dropout()

        if is_train:
            embedding_dropout = self.dropout
            self.lstm.set_dropout(self.dropout)
        else:
            embedding_dropout = 0
            self.lstm.disable_dropout()

        embeddings = []
        for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
            tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
            tag_embedding = dy.dropout(tag_embedding, embedding_dropout)
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                    word = UNK
            word_embedding = self.word_embeddings[self.word_vocab.index(word)]
            word_embedding = dy.dropout(word_embedding, embedding_dropout)
            if self.use_char_lstm:
                chars_embedding = []
                for c in [START] + list(word) + [STOP]:
                    char_embedding = self.char_embeddings[self.char_vocab.index(c)]
                    char_embedding = dy.dropout(char_embedding, embedding_dropout)
                    chars_embedding.append(char_embedding)
                word_char_embedding = char_lstm.transduce(chars_embedding)[-1]
                embeddings.append(dy.concatenate([tag_embedding, word_embedding, word_char_embedding]))
            else:
                embeddings.append(dy.concatenate([tag_embedding, word_embedding]))
            # embeddings.append(dy.concatenate([tag_embedding, word_embedding]))

        lstm_outputs = self.lstm.transduce(embeddings)

        @functools.lru_cache(maxsize=None)
        def get_span_encoding(left, right):
            forward = (
                lstm_outputs[right][:self.lstm_dim] -
                lstm_outputs[left][:self.lstm_dim])
            backward = (
                lstm_outputs[left + 1][self.lstm_dim:] -
                lstm_outputs[right + 1][self.lstm_dim:])
            return dy.concatenate([forward, backward])

        @functools.lru_cache(maxsize=None)
        def get_label_scores(left, right):
            non_empty_label_scores = self.f_label(get_span_encoding(left, right))
            return dy.concatenate([dy.zeros(1), non_empty_label_scores])

        def helper(force_gold):
            if force_gold:
                assert is_train

            chart = {}

            for length in range(1, len(sentence) + 1):
                for left in range(0, len(sentence) + 1 - length):
                    right = left + length

                    label_scores = get_label_scores(left, right)

                    if is_train:
                        oracle_label = gold.oracle_label(left, right)
                        oracle_label_index = self.label_vocab.index(oracle_label)

                    if force_gold:
                        label = oracle_label
                        label_score = label_scores[oracle_label_index]
                    else:
                        if is_train:
                            label_scores = augment(label_scores, oracle_label_index)
                        label_scores_np = label_scores.npvalue()
                        argmax_label_index = int(
                            label_scores_np.argmax() if length < len(sentence) else
                            label_scores_np[1:].argmax() + 1)
                        argmax_label = self.label_vocab.value(argmax_label_index)
                        label = argmax_label
                        label_score = label_scores[argmax_label_index]

                    if length == 1:
                        tag, word = sentence[left]
                        tree = trees.LeafParseNode(left, tag, word)
                        if label:
                            tree = trees.InternalParseNode(label, [tree])
                        chart[left, right] = [tree], label_score
                        continue

                    if force_gold:
                        oracle_splits = gold.oracle_splits(left, right)
                        oracle_split = min(oracle_splits)
                        best_split = oracle_split
                    else:
                        best_split = max(
                            range(left + 1, right),
                            key=lambda split:
                                chart[left, split][1].value() +
                                chart[split, right][1].value())

                    left_trees, left_score = chart[left, best_split]
                    right_trees, right_score = chart[best_split, right]

                    children = left_trees + right_trees
                    if label:
                        children = [trees.InternalParseNode(label, children)]

                    chart[left, right] = (
                        children, label_score + left_score + right_score)

            children, score = chart[0, len(sentence)]
            assert len(children) == 1
            return children[0], score

        if not is_train and k > 1:
            grid, chart = {}, {}
            for length in range(1, len(sentence) + 1):
                for left in range(0, len(sentence) + 1 - length):
                    right = left + length
                    np_label_scores = get_label_scores(left, right).npvalue()
                    label_indecies = (np_label_scores.argsort()[-k:][::-1]
                                        if length < len(sentence) else
                                        np_label_scores[1:].argsort()[-k:][::-1] + 1)
                    grid[left, right] = [(self.label_vocab.value(idx), np_label_scores[idx])
                                                for idx in label_indecies]
                    if length == 1:
                        for label, score in grid[left, right]:
                            tag, word = sentence[left]
                            tree = trees.LeafParseNode(left, tag, word)
                            if label:
                                tree = trees.InternalParseNode(label, [tree])
                            chart.setdefault((left, right), []).append(([tree], score))
                    else:
                        start = [AstarNode(left, right, split, [0,0,0], chart, grid) for split in range(left + 1, right)]
                        goal = (left, right)
                        for node in Solver(grid, chart).astar(start, goal, k):
                            chart.setdefault((left, right), []).append((node.children, node.score))
            return [children[0] for children, _ in chart[0, len(sentence)]], None
        else:
            tree, score = helper(False)
            if is_train:
                oracle_tree, oracle_score = helper(True)
                assert oracle_tree.convert().linearize() == gold.convert().linearize()
                correct = tree.convert().linearize() == gold.convert().linearize()
                loss = dy.zeros(1) if correct else score - oracle_score
                return tree, loss
            else:
                return tree, score

class MyParser(object):
    def __init__(
            self,
            model,
            tag_vocab,
            word_vocab,
            char_vocab,
            label_vocab,
            use_char_lstm,
            tag_embedding_dim,
            word_embedding_dim,
            char_embedding_dim,
            label_embedding_dim,
            lstm_layers,
            lstm_dim,
            char_lstm_dim,
            dec_lstm_dim,
            attention_dim,
            label_hidden_dim,
            keep_valence_value,
            dropouts,
    ):
        self.spec = locals()
        self.spec.pop("self")
        self.spec.pop("model")
        self.model = model.add_subcollection("Parser")
        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.keep_valence_value = keep_valence_value
        self.lstm_dim = lstm_dim

        self.use_char_lstm = use_char_lstm

        self.tag_embeddings = self.model.add_lookup_parameters(
            (tag_vocab.size, tag_embedding_dim))
        self.word_embeddings = self.model.add_lookup_parameters(
            (word_vocab.size, word_embedding_dim))

        embedding_dim = tag_embedding_dim + word_embedding_dim

        if use_char_lstm:
            self.char_vocab = char_vocab
            self.char_embeddings = self.model.add_lookup_parameters(
                (char_vocab.size, char_embedding_dim))
            self.char_lstm = dy.LSTMBuilder(
                1,
                char_embedding_dim,
                char_lstm_dim,
                self.model)
            embedding_dim += char_lstm_dim

        self.enc_lstm = dy.BiRNNBuilder(
            lstm_layers,
            embedding_dim,
            2 * lstm_dim,
            self.model,
            dy.VanillaLSTMBuilder)

        self.label_embeddings = self.model.add_lookup_parameters(
            (label_vocab.size, label_embedding_dim))

        self.dec_lstm = dy.VanillaLSTMBuilder(
            1,
            label_embedding_dim,
            dec_lstm_dim,
            self.model)

        enc_out_dim = embedding_dim + 2 * lstm_dim
        # dec_attend_dim = 2 * dec_lstm_dim
        # Weights = collections.namedtuple('Weights', 'name prev_dim next_dim')
        # ws = []
        # ws.append(Weights(name='c_dec', prev_dim=enc_out_dim, next_dim=dec_lstm_dim))
        # ws.append(Weights(name='key', prev_dim=dec_lstm_dim, next_dim=attention_dim))
        # ws.append(Weights(name='query', prev_dim=dec_lstm_dim, next_dim=attention_dim))
        # ws.append(Weights(name='attention', prev_dim=dec_attend_dim, next_dim=label_hidden_dim))
        # ws.append(Weights(name='probs', prev_dim=label_hidden_dim, next_dim=label_vocab.size))
        # self.ws = {}
        # for w in ws:
        #     weight = self.model.add_parameters((w.next_dim, w.prev_dim))
        #     bias = self.model.add_parameters((w.next_dim))
        #     self.ws[w.name] = (bias, weight)

        self.ws = {}
        W_cdec = self.model.add_parameters((dec_lstm_dim, enc_out_dim))
        b_cdec = self.model.add_parameters((dec_lstm_dim))
        self.ws['c_dec'] = (b_cdec, W_cdec)
        self.ws['alpha'] = self.model.add_parameters((1, attention_dim))
        self.ws['context'] = self.model.add_parameters((label_embedding_dim, dec_lstm_dim))
        self.ws['e'] = self.model.add_parameters((attention_dim, dec_lstm_dim))
        self.ws['d'] = self.model.add_parameters((attention_dim, dec_lstm_dim))
        W_x = self.model.add_parameters((label_hidden_dim, dec_lstm_dim))
        b_x = self.model.add_parameters((label_hidden_dim))
        self.ws['x'] = (b_x, W_x)
        self.ws['probs'] = self.model.add_parameters((label_vocab.size, label_hidden_dim))

        Dropouts = collections.namedtuple('Dropouts', 'lstm embedding')
        self.dropouts = Dropouts(lstm=dropouts[0], embedding=dropouts[1])

    def param_collection(self):
        return self.model

    @classmethod
    def from_spec(cls, spec, model):
        return cls(model, **spec)

    def parse(self, sentence, gold=None, is_dev=False, predict_parms=None):
        is_train = gold is not None
        use_dropout = is_train and not is_dev

        if self.use_char_lstm:
            char_lstm = self.char_lstm.initial_state()
            if use_dropout:
                self.char_lstm.set_dropout(self.dropouts.lstm)
            else:
                self.char_lstm.disable_dropout()

        if use_dropout:
            dropouts = self.dropouts.embedding
            self.enc_lstm.set_dropout(self.dropouts.lstm)
            self.dec_lstm.set_dropout(self.dropouts.lstm)
        else:
            dropouts = 0
            self.enc_lstm.disable_dropout()
            self.dec_lstm.disable_dropout()

        embeddings = []
        for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
            tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
            tag_embedding = dy.dropout(tag_embedding, dropouts)
            if word not in (START, STOP):
                count = self.word_vocab.count(word)
                if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                    word = UNK
            word_embedding = self.word_embeddings[self.word_vocab.index(word)]
            word_embedding = dy.dropout(word_embedding, dropouts)
            if self.use_char_lstm:
                chars_embedding = []
                for c in [START] + list(word) + [STOP]:
                    char_embedding = self.char_embeddings[self.char_vocab.index(c)]
                    char_embedding = dy.dropout(char_embedding, dropouts)
                    chars_embedding.append(char_embedding)
                word_char_embedding = char_lstm.transduce(chars_embedding)[-1]
                embeddings.append(dy.concatenate([tag_embedding, word_embedding, word_char_embedding]))
            else:
                embeddings.append(dy.concatenate([tag_embedding, word_embedding]))
        lstm_outputs = self.enc_lstm.transduce(embeddings)

        encode_outputs_list = [dy.affine_transform([*self.ws['c_dec'], dy.concatenate([e, l])])
                                    for e, l in zip(embeddings[1:-1], lstm_outputs[1:-1])]

        if is_train:
            decode_inputs = [(START,) + tuple(leaf.labels) + (STOP,) for leaf in gold.leaves()]
            losses = []
            encode_outputs = dy.concatenate_cols(encode_outputs_list)

            # k = dy.affine_transform([*self.ws['key'], encode_outputs])
            # key = dy.transpose(dy.rectify(k))

            e = self.ws['e'] * encode_outputs

            for encode_output, decode_input in zip(encode_outputs_list, decode_inputs):

                # label_embedding = []

                context = dy.zeros(self.ws['context'].dim()[0][0])
                h_dec = encode_output
                c_dec = dy.zeros(h_dec.dim()[0])
                dec_s = self.dec_lstm.initial_state([c_dec, h_dec])
                for label, next_label in zip(decode_input[:-1], decode_input[1:]):

                    label_id = self.label_vocab.index(label)
                    label_embedding = self.label_embeddings[label_id]
                    label_embedding = dy.dropout(label_embedding, dropouts)

                    dec_s = dec_s.add_input(label_embedding + context)
                    decode_out = dec_s.output()

                    d = self.ws['d'] * decode_out
                    alpha = dy.softmax(dy.transpose(self.ws['alpha'] * dy.rectify(d + e)))

                    context = self.ws['context'] * encode_outputs * alpha

                    x = dy.affine_transform([*self.ws['x'], decode_out])
                    x = dy.rectify(x)
                    probs = dy.softmax(self.ws['probs'] * x)
                    next_label_id = self.label_vocab.index(next_label)
                    losses.append(-dy.log(dy.pick(probs, next_label_id)))

                    # label_embedding.append(dy.dropout(l, dropouts))

                # import pdb; pdb.set_trace()

                # h_dec = encode_output
                # c_dec = dy.zeros(h_dec.dim()[0])
                #
                # decode_init = self.dec_lstm.initial_state([c_dec, h_dec])
                #
                # decode_output_list = decode_init.transduce(label_embedding)
                # decode_output = dy.concatenate_cols(decode_output_list)
                #
                # q = dy.affine_transform([*self.ws['query'], decode_output])
                #
                # query = dy.rectify(q)
                # alpha = dy.softmax(key * query)
                # context = encode_outputs * alpha
                # x = dy.concatenate([decode_output, context])
                # a = dy.affine_transform([*self.ws['attention'], x])
                # attention = dy.rectify(a)
                #
                # p = dy.affine_transform([*self.ws['probs'], attention])
                # probs = dy.softmax(p)
                #
                # log_prob = []
                # for i, label in enumerate(decode_input[1:]):
                #     id = self.label_vocab.index(label)
                #     log_prob.append(-dy.log(dy.pick(dy.pick(probs, id), i)))
                # losses.extend(log_prob)

            return losses

        else:
            Cell = collections.namedtuple('Cell', 'tree score')

            start = self.label_vocab.index(START)
            stop = self.label_vocab.index(STOP)
            astar_parms = predict_parms['astar_parms']
            for beam_size in predict_parms['beam_parms']:
                hyps = BeamSearch(start, stop, beam_size).beam_search(
                                                            encode_outputs_list,
                                                            self.label_embeddings,
                                                            self.dec_lstm,
                                                            self.ws)

                grid = {}
                for left, (leaf_hyps, leaf) in enumerate(zip(hyps, sentence)):
                    for rank, hyp in enumerate(leaf_hyps):
                        labels = [self.label_vocab.value(h) for h in hyp[0]]
                        partial_tree = trees.LeafMyParseNode(left, *leaf).deserialize(labels)
                        if partial_tree is not None:
                            grid[left, rank] = Cell(tree = partial_tree, score = hyp[1])

                nodes, seen = astar_search(grid, self.keep_valence_value, astar_parms)
                if nodes != []:
                    if astar_parms[0] == 1:
                        return nodes[0].tree
                    else:
                        return [node.tree for node in nodes]

            # if astar_parms[0] == 1:
            nodes = sorted(filter(lambda x: x.left == 0 and x.right == len(sentence), seen), key = lambda x: x.score)
            if nodes != []:
                node = nodes[-1]
                for l in node.tree.missing_leaves():
                    l.parent.children = list(filter(lambda x: x != l, l.parent.children))
            else:
                nodes = sorted(seen, key = lambda x: abs(len(list(x.tree.leaves())) - len(sentence)), reverse = True)
                node = nodes[-1]
                for l in node.tree.missing_leaves():
                     l.parent.children = list(filter(lambda x: x != l, l.parent.children))
                left_leaves = [trees.LeafMyParseNode(i, *leaf) for i, leaf in
                                zip(range(node.left), sentence[:node.left])]
                right_leaves = [trees.LeafMyParseNode(i, *leaf) for i, leaf in
                                zip(range(node.right, len(sentence)), sentence[node.right:])]
                children =  left_leaves + list(node.tree.children) + right_leaves
                node.tree = trees.InternalMyParseNode(node.tree.label, children)

            if node.tree.label in [trees.CL, trees.CR]:
                node.tree.label = 'S'
            return node.tree
            # else:
            #     return None
