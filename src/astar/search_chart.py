import numpy as np

from .astar import AStar
# import trees


class AstarNode(object):

    def __init__(self, left, right, split=0, rank=[0,0,0]):

        assert isinstance(left, int)
        self.left = left

        assert isinstance(right, int)
        self.right = right

        assert isinstance(split, int)
        self.split = split

        assert isinstance(rank, list)
        self.rank = rank

    def __eq__(self, other):
        return all([self.rank == other.rank,
                    self.left == other.left,
                    self.right == other.right,
                    self.split == other.split])

    def __hash__(self):
        return hash((self.left, self.right, self.split, tuple(self.rank)))

    def format_print(self):
        return 'left: {}, right: {}, split: {}'.format(self.left,
                                                        self.right,
                                                        self.split
                                                    )

class Solver(AStar):

    def __init__(self, grid, chart):
        self.grid = grid
        self.chart = chart
        self.cl = []

    def heuristic_cost(self, node, goal, cost_coefficient):
        return 0

    def real_cost(self, node):
        rank_left, rank_right, rank_label = node.rank
        _, left_score = self.chart[node.left, node.split][rank_left]
        _, right_score = self.chart[node.split, node.right][rank_right]
        label_score = self.grid[node.left, node.right][rank_label][1]
        node.score = (left_score + right_score + label_score)
        return node.score

    def fscore(self, node, goal, cost_coefficient):
        real_cost = self.real_cost(node)
        heuristic_cost = self.heuristic_cost(node, goal, cost_coefficient)
        return real_cost + heuristic_cost

    def move_to_closed(self, node):
        self.cl.append(node)
        return True

    def neighbors(self, node):

        neighbors = []

        rank = list(np.array(node.rank) + np.array([1, 0, 0]))
        if rank[0] < len(self.chart[node.left, node.split]):
            neighbor = AstarNode(node.left, node.right, node.split, rank)
            if neighbor not in self.cl:
                neighbors.append(neighbor)

        rank = list(np.array(node.rank) + np.array([0, 1, 0]))
        if rank[1] < len(self.chart[node.split, node.right]):
            neighbor = AstarNode(node.left, node.right, node.split, rank)
            if neighbor not in self.cl:
                neighbors.append(neighbor)

        rank = list(np.array(node.rank) + np.array([0, 0, 1]))
        if rank[2] < len(self.grid[node.left, node.right]):
            neighbor = AstarNode(node.left, node.right, node.split, rank)
            if neighbor not in self.cl:
                neighbors.append(neighbor)

        return neighbors

    def is_goal_reached(self, node, goal):
        return (node.left, node.right) == (goal.left, goal.right)
