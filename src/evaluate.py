import math
import os.path
import re
import subprocess
import tempfile
import numpy as np

import trees

class FScore(object):
    def __init__(self, recall, precision, fscore):
        self.recall = recall
        self.precision = precision
        self.fscore = fscore

    def __str__(self):
        return "(Recall={:.2f}, Precision={:.2f}, FScore={:.2f})".format(
            self.recall, self.precision, self.fscore)

class Bracket(object):
    def __init__(self, length, match_bracket, gold_bracket, test_bracket):
        self.length = length
        self.match_bracket = match_bracket
        self.gold_bracket = gold_bracket
        self.test_bracket = test_bracket
        recall = min(self.match_bracket/self.gold_bracket, 1.) * 100
        precision = min(self.match_bracket/self.test_bracket, 1.) * 100
        fscore = 2./(1/recall + 1/precision) if (recall > 0 and precision > 0) else 0.0
        self.Fscore = FScore(round(recall, 2), round(precision, 2), round(fscore, 2))

def evalb(evalb_dir, gold_trees, predicted_trees):
    import pdb; pdb.set_trace()
    assert os.path.exists(evalb_dir)
    evalb_program_path = os.path.join(evalb_dir, "evalb")
    evalb_param_path = os.path.join(evalb_dir, "COLLINS.prm")
    assert os.path.exists(evalb_program_path)
    assert os.path.exists(evalb_param_path)

    assert len(gold_trees) == len(predicted_trees)
    for gold_tree, predicted_tree in zip(gold_trees, predicted_trees):
        assert isinstance(gold_tree, trees.TreebankNode)
        assert isinstance(predicted_tree, trees.TreebankNode)
        gold_leaves = list(gold_tree.leaves())
        predicted_leaves = list(predicted_tree.leaves())
        assert len(gold_leaves) == len(predicted_leaves)
        assert all(
            gold_leaf.word == predicted_leaf.word
            for gold_leaf, predicted_leaf in zip(gold_leaves, predicted_leaves))

    temp_dir = tempfile.TemporaryDirectory(prefix="evalb-")
    gold_path = os.path.join(temp_dir.name, "gold.txt")
    predicted_path = os.path.join(temp_dir.name, "predicted.txt")
    output_path = os.path.join(temp_dir.name, "output.txt")

    with open(gold_path, "w") as outfile:
        for tree in gold_trees:
            outfile.write("{}\n".format(tree.linearize()))

    with open(predicted_path, "w") as outfile:
        for tree in predicted_trees:
            outfile.write("{}\n".format(tree.linearize()))

    command = "{} -p {} {} {} > {}".format(
        evalb_program_path,
        evalb_param_path,
        gold_path,
        predicted_path,
        output_path,
    )
    subprocess.run(command, shell=True)

    fscore = FScore(math.nan, math.nan, math.nan)
    with open(output_path) as infile:
        for line in infile:
            match = re.match(r"Bracketing Recall\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.recall = float(match.group(1))
            match = re.match(r"Bracketing Precision\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.precision = float(match.group(1))
            match = re.match(r"Bracketing FMeasure\s+=\s+(\d+\.\d+)", line)
            if match:
                fscore.fscore = float(match.group(1))
                break
    success = (
        not math.isnan(fscore.fscore) or
        fscore.recall == 0.0 or
        fscore.precision == 0.0)

    if success:
        temp_dir.cleanup()
    else:
        print("Error reading EVALB results.")
        print("Gold path: {}".format(gold_path))
        print("Predicted path: {}".format(predicted_path))
        print("Output path: {}".format(output_path))

    return fscore


def evalb_full(evalb_dir, gold_trees, predicted_trees):
    assert os.path.exists(evalb_dir)
    evalb_program_path = os.path.join(evalb_dir, "evalb")
    evalb_param_path = os.path.join(evalb_dir, "COLLINS.prm")
    assert os.path.exists(evalb_program_path)
    assert os.path.exists(evalb_param_path)

    assert len(gold_trees) == len(predicted_trees)
    for gold_tree, predicted_tree in zip(gold_trees, predicted_trees):
        assert isinstance(gold_tree, trees.TreebankNode)
        assert isinstance(predicted_tree, trees.TreebankNode)
        gold_leaves = list(gold_tree.leaves())
        predicted_leaves = list(predicted_tree.leaves())
        assert len(gold_leaves) == len(predicted_leaves)
        assert all(
            gold_leaf.word == predicted_leaf.word
            for gold_leaf, predicted_leaf in zip(gold_leaves, predicted_leaves))

    temp_dir = tempfile.TemporaryDirectory(prefix="evalb-")
    gold_path = os.path.join(temp_dir.name, "gold.txt")
    predicted_path = os.path.join(temp_dir.name, "predicted.txt")
    output_path = os.path.join(temp_dir.name, "output.txt")

    with open(gold_path, "w") as outfile:
        for tree in gold_trees:
            outfile.write("{}\n".format(tree.linearize()))

    with open(predicted_path, "w") as outfile:
        for tree in predicted_trees:
            outfile.write("{}\n".format(tree.linearize()))

    command = "{} -p {} {} {} > {}".format(
        evalb_program_path,
        evalb_param_path,
        gold_path,
        predicted_path,
        output_path,
    )
    subprocess.run(command, shell=True)

    # fscore = FScore(math.nan, math.nan, math.nan)
    start =  False
    results = []
    with open(output_path) as infile:
        for line in infile:
            match = re.match(r"====", line)
            if not start and match:
                start = True
            elif start and match:
                break
            elif start:
                # length, match_bracket, gold_bracket, test_bracket = np.array(line.split())[[1,5,6,7]]
                line = line.split()
                length = int(line[1])
                match_bracket, gold_bracket, test_bracket = line[5:8]
                result = Bracket(length, float(match_bracket), float(gold_bracket), float(test_bracket))
                results.append(result)
    temp_dir.cleanup()
    return results


def recall(gold_trees, predicted_trees, k):

    def helper(predict):
        return [p.linearize() for p in predict]

    return np.mean([g.linearize() in helper(p[:k]) for g, p in zip(gold_trees, predicted_trees)])
