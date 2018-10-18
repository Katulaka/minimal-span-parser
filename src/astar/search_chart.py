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
        return id(self)

    def format_print(self):
        return 'left: {}, right: {}, split: {}'.format(self.left,
                                                        self.right,
                                                        self.split
                                                    )

# class ClosedList(object):
#
#     def __init__(self):
#         self.lindex = {}
#         self.rindex = {}
#
#     def put(self, node):
#         if node.left in self.lindex:
#             if node not in self.lindex[node.left]:
#                 self.lindex[node.left].append(node)
#         else:
#             self.lindex[node.left] = [node]
#
#         if node.right in self.rindex:
#             if node not in self.rindex[node.right]:
#                 self.rindex[node.right].append(node)
#         else:
#             self.rindex[node.right] = [node]
#
#     def getr(self, idx):
#         return self.rindex.get(idx, [])
#
#     def getl(self, idx):
#         return self.lindex.get(idx, [])

class Solver(AStar):

    def __init__(self, grid, chart):
        self.chart = chart
        self.grid = grid
        # self.cl = ClosedList()
        # self.seen = []

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
        # self.cl.put(node)
        return True

    def neighbors(self, node):
        def helper(lst):
            return np.array(lst) + np.eye(len(lst), dtype=int)
        res = [AstarNode(node.left, node.right, node.split, list(rank))
                    for rank in helper(node.rank)]
        import pdb; pdb.set_trace()
        return res

    def is_goal_reached(self, node, goal):
        return (node.left, node.right) == (goal.left, goal.right)
