from itertools import chain
# import functools
import numpy as np

from .astar import AStar
import trees


class AstarNode(object):

    def __init__(self, left, right, rank=[], tree=None):

        assert isinstance(left, int)
        self.left = left

        assert isinstance(right, int)
        self.right = right

        assert isinstance(rank, list)
        self.rank = rank

        self.tree = tree
        # assert isinstance(trees, list)
        # self.trees = trees

    def __eq__(self, other):
        return self.rank == other.rank and (self.left, self.right) == (other.left, other.right)

    def __hash__(self):
        return id(self)

    def format_print(self, label):
        pair = '({},{})'.format(self.left, self.right)

        ranks_split = np.split(np.array(self.rank), np.where(np.diff(self.rank))[0] + 1)
        ranks = ', '.join(['{{{}}}{}'.format(r[0], len(r)) for r in ranks_split])

        MY_LENGTH_CONSTRAINT = len(ranks_split) * 7
        node_string = '[{}:] node: {: <8} rank: [{: <{mlc}}]'.format(label, pair, ranks,
                                                        mlc = MY_LENGTH_CONSTRAINT)

        for i, tree in enumerate(self.trees):
            pair = '({},{})'.format(tree.left, tree.right)
            node_string = '{} tree[{}]: {: <8}'.format(node_string, i, pair)

        return node_string

    def is_valid(self, keep_valence_value, left_tree, right_tree):
        assert isinstance(left_tree, trees.InternalMyParseNode)
        assert isinstance(right_tree, trees.InternalMyParseNode)

        # @functools.lru_cache(maxsize=None)
        def helper(_trees, miss_side):

            assert (_trees[0].label in [trees.CR, trees.CL])
            assert len(_trees[0].children) == 1

            leaves = list(_trees[1].missing_leaves(miss_side))
            if leaves != []:
                leaves = leaves[::-1] if miss_side == trees.L else leaves
                if not keep_valence_value:
                    return _trees[1].combine(_trees[0].children[0], leaves[0])

                label = _trees[0].children[-1].bracket_label()
                for leaf in leaves:
                    if label == leaf.label.split(miss_side)[-1]:
                        return _trees[1].combine(_trees[0].children[0], leaf)
            return None

        if not len(list(right_tree.missing_leaves())) and \
                not len(list(left_tree.missing_leaves())):
            return False

        #Trying to combine Left Tree --> Right Tree
        if left_tree.label == trees.CR and not len(list(left_tree.missing_leaves())):
            tree = helper([left_tree, right_tree], trees.L)
            if tree is not None:
                self.tree = tree
                return True

        #Trying to combine Right Tree --> Left Tree
        if right_tree.label == trees.CL and not len(list(right_tree.missing_leaves())):
            tree = helper([right_tree, left_tree], trees.R)
            if tree is not None:
                self.tree = tree
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

    def __init__(self, grid, keep_valence_value):
        self.grid = grid
        self.keep_valence_value = keep_valence_value
        self.cl = ClosedList()
        self.seen = []

    def heuristic_cost(self, node, goal, cost_coefficient):
        left = list(range(node.left))
        right = list(range(node.right, goal.right))
        return cost_coefficient * sum([self.grid[i,0].score for i in chain(left, right)])

    def real_cost(self, node):
        position = zip(range(node.left, node.right), node.rank)
        return sum([self.grid[i,rank].score for i, rank in position])

    def fscore(self, node, goal, cost_coefficient):
        real_cost = self.real_cost(node)
        heuristic_cost = self.heuristic_cost(node, goal, cost_coefficient)
        node.score = real_cost + heuristic_cost
        # return real_cost + heuristic_cost
        return node.score

    def move_to_closed(self, node):
        self.cl.put(node)

    def neighbors(self, node):
        neighbors = []
        for nb in self.cl.getl(node.right):
            nb_node = AstarNode(node.left, nb.right, node.rank + nb.rank)
            if nb_node not in self.seen and nb_node.is_valid(self.keep_valence_value, node.tree, nb.tree):
                self.seen.append(nb_node)
                neighbors.append(nb_node)
        for nb in self.cl.getr(node.left):
            # nb_node = AstarNode(nb.left, node.right, nb.rank + node.rank, nb.trees + node.trees)
            nb_node = AstarNode(nb.left, node.right, nb.rank + node.rank)
            if nb_node not in self.seen and nb_node.is_valid(self.keep_valence_value, nb.tree,  node.tree):
                self.seen.append(nb_node)
                neighbors.append(nb_node)
        rank = node.rank[0] + 1
        if len(node.rank) == 1 and (node.left, rank) in self.grid:
            nb_node = AstarNode(node.left, node.right, [rank], self.grid[node.left, rank].tree)
            if nb_node not in self.seen:
                self.seen.append(nb_node)
                neighbors.append(nb_node)
        return neighbors

    def is_goal_reached(self, node, goal):
        # if (node.left, node.right) == (goal.left, goal.right):
        #     return not len(list(node.tree.missing_leaves()))
        if (node.left, node.right) == (goal.left, goal.right) \
            and not len(list(node.tree.missing_leaves())):
            node_leaves = list(node.tree.leaves())
            goal_leaves = list(goal.tree.leaves())
            return all(
                (goal_leaf.tag, goal_leaf.word) == (node_leaf.tag, node_leaf.word)
                for goal_leaf, node_leaf in zip(goal_leaves, node_leaves))
        return False

def astar_search(grid, sentence, keep_valence_value, astar_parms):

    n_words = max(grid.keys(), key = lambda x : x[0])[0] + 1
    start = [AstarNode(left, left + 1, [0], grid[left, 0].tree) for left in range(n_words)]
    # goal = AstarNode(0, n_words)
    children = [trees.LeafMyParseNode(left, *leaf) for left, leaf in enumerate(sentence)]
    goal_tree = trees.InternalMyParseNode('.', children)
    goal = AstarNode(0, len(sentence), tree = goal_tree)
    # let's solve it
    solver = Solver(grid, keep_valence_value)
    nodes = solver.astar(start, goal, *astar_parms)

    if len(nodes):
        print([s[1] for s in sentence] == [l.word for l in nodes[0].tree.leaves()])
    else:
        nodes = filter(lambda x: x.left == 0 and x.right == len(sentence), solver.seen)
        nodes = sorted(nodes, key = lambda x: x.score, reverse = True)
        if len(nodes):
            import pdb; pdb.set_trace()
            for node in nodes:
                node.tree = node.tree.filter_missing()
         else:
             import pdb; pdb.set_trace()
             nodes = sorted(solver.seen, key = lambda x: x.right - x.left, reverse = True)
             for node in nodes:
                 node.tree = node.tree.filter_missing()

        print("No nodes")
    return nodes, solver.seen


    # nodes = filter(lambda x: x.left == 0 and x.right == len(sentence), seen)
    # nodes = sorted(nodes, key = lambda x: x.score, reverse = True)
    # nodes = nodes[:astar_parms[0]]
    # if nodes != []:
    #     for node in nodes:
    #         for l in node.tree.missing_leaves():
    #             l.parent.children = list(filter(lambda x: x != l, l.parent.children))
    # else:
    #     nodes = sorted(seen, key = lambda x: x.right - x.left, reverse = True)
    #     nodes = nodes[:astar_parms[0]]
    #     for node in nodes:
    #         for l in node.tree.missing_leaves():
    #             l.parent.children = list(filter(lambda x: x != l, l.parent.children))
    #         left_leaves = [trees.LeafMyParseNode(i, *leaf) for i, leaf in
    #                     zip(range(node.left), sentence[:node.left])]
    #         right_leaves = [trees.LeafMyParseNode(i, *leaf) for i, leaf in
    #                     zip(range(node.right, len(sentence)), sentence[node.right:])]
    #         children =  left_leaves + list(node.tree.children) + right_leaves
    #         node.tree = trees.InternalMyParseNode(node.tree.label, children)
    #
    #         left_rank = [predict_parms['beam_parms'][-1]] * node.left
    #         right_rank = [predict_parms['beam_parms'][-1]]*(len(sentence)-node.right)
    #         node.rank = left_rank + node.rank + right_rank
    #
    # for node in nodes:
    #     if node.tree.label in [trees.CL, trees.CR]:
    #         node.tree.label = 'S'
