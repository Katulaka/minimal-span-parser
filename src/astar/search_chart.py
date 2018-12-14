import numpy as np

from .astar import AStar
import trees


class AstarNode(object):

    def __init__(self, left, right, split, rank, chart, grid):

        assert isinstance(left, int)
        self.left = left

        assert isinstance(right, int)
        self.right = right

        assert isinstance(split, int)
        self.split = split

        assert isinstance(rank, tuple)
        self.rank = rank

        left_rank, right_rank, label_rank = rank
        left_trees, left_score = chart[left, split][left_rank]
        right_trees, right_score = chart[split, right][right_rank]
        children = left_trees + right_trees
        label, label_score = grid[left, right][label_rank]
        if label:
            children = [trees.InternalParseNode(label, children)]

        self.children = children
        self.score = (left_score + right_score + label_score)

    def __eq__(self, other):
        # return self.children == other.children
        return (str([child.convert().linearize() for child in self.children])
                == str([child.convert().linearize() for child in other.children]))

    def __hash__(self):
        # return hash(tuple(self.children))
        return hash(str([child.convert().linearize() for child in self.children]))

    def format_print(self):
        return 'left: {}, right: {}, split: {}'.format(self.left,
                                                        self.right,
                                                        self.split
                                                    )

class Solver(AStar):

    def __init__(self, grid, chart):
        self.grid = grid
        self.chart = chart
        self.cl = {}

    def heuristic_cost(self, node, goal, cost_coefficient):
        return 0

    def real_cost(self, node):
        return node.score

    def fscore(self, node, goal, cost_coefficient):
        real_cost = self.real_cost(node)
        heuristic_cost = self.heuristic_cost(node, goal, cost_coefficient)
        return real_cost + heuristic_cost

    def move_to_closed(self, node):
        # self.cl.append(node)

        self.cl[node.left, node.right, node.split, node.rank] = node
        return True

    def neighbors(self, node):

        neighbors = []
        for i, delta in enumerate(np.eye(3)):
            rank = tuple(np.array(node.rank) + delta)
            if rank[i] < len(self.chart[node.left, node.split]):
                nb_key = (node.left, node.right, node.split, rank)
                neighbor = AstarNode(*nb_key, self.chart, self.grid)
                if nb_key not in self.cl:
                    neighbors.append(neighbor)
        return neighbors

        # rank = tuple(np.array(node.rank) + np.array([1, 0, 0]))
        # if rank[0] < len(self.chart[node.left, node.split]):
        #     nb_key = (node.left, node.right, node.split, rank)
        #     neighbor = AstarNode(*nb_key, self.chart, self.grid)
        #     if nb_key not in self.cl:
        #         neighbors.append(neighbor)

        # rank = tuple(np.array(node.rank) + np.array([0, 1, 0]))
        # if rank[1] < len(self.chart[node.split, node.right]):
        #     nb_key = (node.left, node.right, node.split, rank)
        #     neighbor = AstarNode(*nb_key, self.chart, self.grid)
        #     if nb_key not in self.cl:
        #         neighbors.append(neighbor)
        #
        # rank = tuple(np.array(node.rank) + np.array([0, 0, 1]))
        # if rank[2] < len(self.grid[node.left, node.right]):
        #     nb_key = (node.left, node.right, node.split, rank)
        #     neighbor = AstarNode(*nb_key, self.chart, self.grid)
        #     if nb_key not in self.cl:
        #         neighbors.append(neighbor)


    def is_goal_reached(self, node, goal):
        return (node.left, node.right) == goal
