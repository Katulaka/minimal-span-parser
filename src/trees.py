import collections.abc
import math
R = '}'
L = '{'
CR = '>'
CL = '<'
ANY = '*'

class TreebankNode(object):
    pass

class InternalTreebankNode(TreebankNode):
    def __init__(self, label, children):
        assert isinstance(label, str)
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, TreebankNode) for child in children)
        assert children
        self.children = tuple(children)

        for child in self.children:
            child.parent = self

        self.parent = None

    def linearize(self):
        return "({} {})".format(
            self.label, " ".join(child.linearize() for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self, index=0):
        tree = self
        sublabels = [self.label]

        while len(tree.children) == 1 and isinstance(
                tree.children[0], InternalTreebankNode):
            tree = tree.children[0]
            sublabels.append(tree.label)

        children = []
        for child in tree.children:
            children.append(child.convert(index=index))
            index = children[-1].right

        return InternalParseNode(tuple(sublabels), children)

    def myconvert(self, dependancy, index=0):
        tree = self
        children = []
        for child in tree.children:
            children.append(child.myconvert(dependancy, index=index))
            index = children[-1].right

        return InternalPathParseNode(tree.label, children)

    def normalize(self):
        tree = self
        children = []
        for child in tree.children:
            children.append(child.normalize())
        return InternalTreebankNode('XX', children)


class LeafTreebankNode(TreebankNode):
    def __init__(self, tag, word):
        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

        self.parent = None

    def linearize(self):
        return "({} {})".format(self.tag, self.word)

    def leaves(self):
        yield self

    def convert(self, index=0):
        return LeafParseNode(index, self.tag, self.word)

    def myconvert(self, dependancy, index=0):
        return LeafPathParseNode(index, self.tag, self.word)(dependancy[index] - 1)

    def normalize(self):
        return LeafTreebankNode('XX', self.word)


class ParseNode(object):
    pass

class InternalParseNode(ParseNode):
    def __init__(self, label, children):
        assert isinstance(label, tuple)
        assert all(isinstance(sublabel, str) for sublabel in label)
        assert label
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, ParseNode) for child in children)
        assert children
        assert len(children) > 1 or isinstance(children[0], LeafParseNode)
        assert all(
            left.right == right.left
            for left, right in zip(children, children[1:]))
        self.children = tuple(children)

        self.left = children[0].left
        self.right = children[-1].right

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self):
        children = [child.convert() for child in self.children]
        tree = InternalTreebankNode(self.label[-1], children)
        for sublabel in reversed(self.label[:-1]):
            tree = InternalTreebankNode(sublabel, [tree])
        return tree

    def enclosing(self, left, right):
        assert self.left <= left < right <= self.right
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right:
                return child.enclosing(left, right)
        return self

    def oracle_label(self, left, right):
        enclosing = self.enclosing(left, right)
        if enclosing.left == left and enclosing.right == right:
            return enclosing.label
        return ()

    def oracle_splits(self, left, right):
        return [
            child.left
            for child in self.enclosing(left, right).children
            if left < child.left < right
        ]

    def __eq__(self, other):
        return isinstance(other, InternalParseNode) and self.label == other.label and self.children == other.children

    def __hash__(self):
        return hash((self.label, tuple(self.children)))

class LeafParseNode(ParseNode):
    def __init__(self, index, tag, word):
        assert isinstance(index, int)
        assert index >= 0
        self.left = index
        self.right = index + 1

        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def leaves(self):
        yield self

    def convert(self):
        return LeafTreebankNode(self.tag, self.word)

    def __eq__(self, other):
        return isinstance(other, LeafParseNode) and self.tag == other.tag and self.word == other.word

    def __hash__(self):
        return hash((self.tag, self.word))


class PathParseNode(object):
    pass

class InternalPathParseNode(PathParseNode):
    def __init__(self, label, children):
        assert isinstance(label, str)
        assert label
        self.label = label

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, PathParseNode) for child in children)
        assert children

        assert all(
            left.right == right.left
            for left, right in zip(children, children[1:])
            if not (isinstance(left, MissPathParseNode)
                    or isinstance(right, MissPathParseNode)))
        self.children = tuple(children)

        for child in self.children:
            child.parent = self

        # self.left = children[0].left
        # self.right = children[-1].right
        self.left = min([child.left for child in children if child.left>-1])
        self.right = max([child.right for child in children if child.right < math.inf])

        self.parent = None

    # def __call__(self, keep_valence_value):
    def __call__(self):
        # self.serialize(keep_valence_value)
        self.serialize()
        return self

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def missing_leaves(self, side=None):
        for child in self.children:
            yield from child.missing_leaves(side)

    def convert(self):
        children = [child.convert() for child in self.children]
        tree = InternalTreebankNode(self.label, children)
        return tree

    # def serialize(self, keep_valence_value):
    def serialize(self):

        def helper(current, sibling):
            side = L if current.left > sibling.left else R
            # if not keep_valence_value:
            return side+ANY
            # return  side+sibling.bracket_label()

        # Recursion
        flag = CR
        for child in self.children:
            # winner_child_leaf = child.serialize(keep_valence_value)
            winner_child_leaf = child.serialize()

            # Reached end of path can add flag
            if winner_child_leaf.dependancy in range(self.left, self.right) or (flag == CL):
                winner_child_leaf.labels.append(flag)
            else:
                # only single child will move to parent
                # and its value will be the one that is returned
                # to the parent
                winner_child_leaf.labels.append(self.label)
                winner_child_leaf.labels.extend([helper(child, sibling) for sibling in child.siblings()])
                ret_leaf_node = winner_child_leaf

                # once we reached here, it means that
                # this path includes the parent and thus flag
                # direction should flip
                flag = CL

        return ret_leaf_node

    def siblings(self):
        for child in self.parent.children:
            if child != self:
                yield child

    def bracket_label(self):
        return self.label

    def combine(self, node_to_merge, node_to_remove):
        tree = self
        children = []
        for child in tree.children:
            children.append(child.combine(node_to_merge, node_to_remove))
        children = sorted(children, key= lambda x: x.left)
        return InternalPathParseNode(tree.label, children)

    def filter_missing(self):
        tree = self
        children = []
        for child in tree.children:
            children.append(child.filter_missing())
        children = list(filter(lambda x: isinstance(x, PathParseNode), children))
        return InternalPathParseNode(tree.label, children)

    def subtrees(self):
        tree = self
        if not list(tree.missing_leaves()):
            return [tree.left, tree.right]
        subtrees = []
        for child in tree.children:
            subtrees.extend(child.subtrees())
        return list(set(subtrees))


class LeafPathParseNode(PathParseNode):
    def __init__(self, index, tag, word):
        assert isinstance(index, int)
        assert index >= 0
        self.left = index
        self.right = index + 1

        assert isinstance(tag, str)
        self.tag = tag

        assert isinstance(word, str)
        self.word = word

    def __call__(self, dependency):
        assert isinstance(dependency, int)
        self.dependancy = dependency
        return self

    def bracket_label(self):
        return self.tag

    def leaves(self):
        yield self

    def missing_leaves(self, side):
        yield from ()

    def convert(self):
        return LeafTreebankNode(self.tag, self.word)

    def combine(self, node_to_merge, node_to_remove):
        return LeafPathParseNode(self.left, self.tag, self.word)

    def filter_missing(self):
        return LeafPathParseNode(self.left, self.tag, self.word)

    def siblings(self):
        for child in self.parent.children:
            if child != self:
                yield child

    # def serialize(self, keep_valence_value):
    def serialize(self):
        self.labels = []
        return self

    def deserialize(self, labels):
        # invalid labels: labels should start with token not missing
        if (labels[0].startswith(R) or labels[0].startswith(L)):
            return None
        index = self.left
        children = [self]
        while labels:
            p_label = labels[0]
            labels = labels[1:]
            while labels and (labels[0].startswith(R) or labels[0].startswith(L)):
                if labels[0].startswith(R):
                    # index = children[-1].right
                    # children += [MissPathParseNode(labels[0], index)]
                    children += [MissPathParseNode(labels[0])]
                else:
                    # index = children[0].left - 1
                    # children = [MissPathParseNode(labels[0], index)] + children
                    children = [MissPathParseNode(labels[0])] + children
                labels = labels[1:]
            children = [InternalPathParseNode(p_label, children)]
        return children[-1]

    def subtrees(self):
        return [self.left, self.right]

class MissPathParseNode(PathParseNode):
    # def __init__(self, label, index = 0):
    def __init__(self, label):
        self.label = label
        self.left = -1 if label.startswith(L) else math.inf
        self.right = -1 if label.startswith(L) else math.inf
        # self.left = index
        # self.right = index + 1

    def leaves(self):
        yield self

    def missing_leaves(self, side):
        if side is None or self.label.startswith(side):
            yield self
        else:
            yield from ()

    def siblings(self):
        for child in self.parent.children:
            if child != self:
                yield child

    def convert(self):
        return LeafTreebankNode(self.label, self.label)

    def bracket_label(self):
        return self.label

    def combine(self, node_to_merge, node_to_remove):

        tree = self
        if tree == node_to_remove:
            return node_to_merge.combine(node_to_merge, node_to_remove)
        # return MissPathParseNode(tree.label, tree.left)
        return MissPathParseNode(tree.label)

    def filter_missing(self):
        yield from ()

    def subtrees(self):
        return []

def load_trees(path, strip_top=True):
    with open(path) as infile:
        tokens = infile.read().replace("(", " ( ").replace(")", " ) ").split()

    def helper(index):
        trees = []

        while index < len(tokens) and tokens[index] == "(":

            paren_count = 0
            while tokens[index] == "(":
                index += 1
                paren_count += 1

            label = tokens[index]
            index += 1

            if tokens[index] == "(":
                children, index = helper(index)
                trees.append(InternalTreebankNode(label, children))
            else:
                word = tokens[index]
                index += 1
                trees.append(LeafTreebankNode(label, word))

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1
        return trees, index

    trees, index = helper(0)
    assert index == len(tokens)

    if strip_top:
        for i, tree in enumerate(trees):
            if tree.label == "TOP":
                assert len(tree.children) == 1
                trees[i] = tree.children[0]
                trees[i].parent = None

    return trees
