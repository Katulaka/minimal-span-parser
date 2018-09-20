from .astar import AStar
import trees
import numpy as np

class NodeT(object):

    def __init__(self, left, right, rank, trees = []):

        assert isinstance(left, int)
        self.left = left

        assert isinstance(right, int)
        self.right = right

        assert isinstance(rank, list)
        self.rank = rank

        assert isinstance(trees, list)
        self.trees = trees

    def __eq__(self, other):
        return self.rank == other.rank and (self.left, self.right) == (other.left, other.right)

    def __hash__(self):
        return id(self)

    def is_valid(self, miss_tag_any):
    def format_print(self, label):
        pair = '({},{})'.format(self.left, self.right)

        ranks_split = np.split(np.array(self.rank), np.where(np.diff(self.rank))[0] + 1)
        ranks = ', '.join(['{{{}}}{}'.format(r[0], len(r)) for r in ranks_split])

        MY_LENGTH_CONSTRAINT = len(ranks_split) * 7
        node_string = '[{}:] node: {: <8} rank: [{: <{mlc}}]'.format(label, pair, ranks,
                                                        mlc = MY_LENGTH_CONSTRAINT)

        for i, tree in enumerate(self.trees):
            pair = '({},{})'.format(tree.left, tree.right)
            # ptb = tree.convert().linearize()
            node_string = '{} tree[{}]: {: <8}'.format(node_string, i, pair)

        return node_string

        assert isinstance(self.trees, list)
        assert len(self.trees) in [1,2]

        def helper(c_trees, comb_side, miss_side, miss_tag_any):

            assert isinstance(c_trees[0], trees.InternalMyParseNode)
            assert (c_trees[0].label in [trees.CR, trees.CL])
            assert len(c_trees[0].children) == 1

            for leaf in list(c_trees[1].leaves())[::-1]:
                # check that destination tree has missing leaves and
                # they combine to the proper side
                if isinstance(leaf, trees.MissMyParseNode) and leaf.label.startswith(miss_side):
                    if miss_tag_any:
                        return leaf
                    label = leaf.label.split(miss_side)[-1]
                    if isinstance(c_trees[0].children[-1], trees.InternalMyParseNode):
                        src_label = c_trees[0].children[-1].label
                    else:
                        src_label = c_trees[0].children[-1].tag
                    if src_label == label:
                        return leaf
            return None

        if len(self.trees) == 1:
            return True

        if all(isinstance(tree, trees.InternalMyParseNode) for tree in self.trees):
            #try combining left tree into right tree
            if self.trees[0].label == trees.CR and self.trees[0].is_no_missing_leaves():
                miss_node = helper(self.trees, trees.CR, trees.L, miss_tag_any)
                if miss_node is not None:
                    self.trees = [self.trees[1].combine_tree(self.trees[0], miss_node)]
                    return True

            #try combining right tree into left tree
            if isinstance(self.trees[1], trees.InternalMyParseNode):
                if self.trees[1].label == trees.CL and self.trees[1].is_no_missing_leaves():
                    miss_node = helper(self.trees[::-1], trees.CL, trees.R, miss_tag_any)
                    if miss_node is not None:
                        self.trees = [self.trees[0].combine_tree(self.trees[1], miss_node)]
                        return True

        return False


class ClosedList(object):

    def __init__(self):
        self.lindex = {}
        self.rindex = {}

    def put(self, node):
        if node.left in self.lindex:
            if node not in self.lindex[node.left]:
                self.lindex[node.left].append(node)
        else:
            self.lindex[node.left] = [node]

        if node.right in self.rindex:
            if node not in self.rindex[node.right]:
                self.rindex[node.right].append(node)
        else:
            self.rindex[node.right] = [node]

    def getr(self, idx):
        return self.rindex.get(idx, [])

    def getl(self, idx):
        return self.lindex.get(idx, [])


class Solver(AStar):

    def __init__(self, ts_mat, no_val_gap):
        self.ts_mat = ts_mat
        self.miss_tag_any = no_val_gap
        self.cl = ClosedList()
        self.seen = []

    def heuristic_cost(self, node, goal, cost_coeff):
        left = list(range(node.left))
        right = list(range(node.right, goal.right))
        return cost_coeff * sum([self.ts_mat[i][0][1] for i in chain(left, right)])

    def real_cost(self, node):
        position = zip(range(node.left, node.right), node.rank)
        return sum([self.ts_mat[i][rank][1] for i, rank in position])

    def fscore(self, node, goal, cost_coeff):
        real_cost = self.real_cost(node)
        heuristic_cost = self.heuristic_cost(node, goal, cost_coeff)
        return real_cost + heuristic_cost

    def move_to_closed(self, current):
        self.cl.put(current)

    def neighbors(self, node):
        neighbors = []
        for nb in self.cl.getl(node.right):
            nb_node = NodeT(node.left, nb.right, node.rank + nb.rank, node.trees + nb.trees)
            if nb_node not in self.seen and nb_node.is_valid(self.miss_tag_any):
                self.seen.append(nb_node)
                neighbors.append(nb_node)
        for nb in self.cl.getr(node.left):
            nb_node = NodeT(nb.left, node.right, nb.rank + node.rank, nb.trees + node.trees)
            if nb_node.is_valid(self.miss_tag_any) and nb_node not in self.seen:
                self.seen.append(nb_node)
                neighbors.append(nb_node)
        if len(node.rank) == 1 and node.rank[0] + 1 < len(self.ts_mat[node.left]):
            rank = node.rank[0] + 1
            trees = [self.ts_mat[node.left][rank][0]]
            nb_node = NodeT(node.left, node.right, [rank], trees)
            if nb_node.is_valid(self.miss_tag_any) and nb_node not in self.seen:
                self.seen.append(nb_node)
                neighbors.append(nb_node)
        return neighbors

    def is_goal_reached(self, current, goal):
        if (current.left, current.right) == (goal.left, goal.right):
            if len(current.trees) == 1:
                return current.trees[0].is_no_missing_leaves()
        return False

def astar_search(beams, keep_valence_value, astar_parms, verbose=1):

    n_words = len(beams)
    start = [NodeT(idx, idx+1, [0], [beams[idx][0][0]]) for idx in range(n_words)]
    goal = NodeT(0, n_words, [])
    # let's solve it
    nodes = Solver(beams, keep_valence_value).astar(start, goal, *astar_parms, verbose)

    if nodes == []:
         return trees.LeafMyParseNode(0, '', '')
    return nodes[0].trees[0]
