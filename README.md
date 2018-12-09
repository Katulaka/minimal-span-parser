# Path-Based Neural Constituency Parsing

This is a reference Python implementation of the path-based constituency parsers. 

## Requirements and Setup

* Python 3.5 or higher.
* [DyNet](https://github.com/clab/dynet). We recommend installing DyNet from source with MKL support for significantly faster run time.
* [EVALB](http://nlp.cs.nyu.edu/evalb/). Before starting, run `make` inside the `EVALB/` directory to compile an `evalb` executable. This will be called from Python for evaluation.

## Training

A new model can be trained using the command `python3 src/main.py train ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--numpy-seed` | NumPy random seed | Random
`--parser-type` | `path` or `chart` | N/A
`--tag-embedding-dim` | Tag embedding dimension | 50
`--word-embedding-dim` | Word embedding dimension | 100
`--lstm-layers` | Number of bidirectional LSTM layers | 2
`--lstm-dim` | Hidden dimension of each LSTM within each layer | 250
`--label-hidden-dim` | Hidden dimension of label-scoring feedforward network | 250
`--dropout` | Dropout rate for LSTMs | 0.4
`--model-path-base` | Path base to use for saving models | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--train-path` | Path to training trees | `data/02-21.10way.clean`
`--dev-path` | Path to development trees | `data/22.auto.clean`
`--batch-size` | Number of examples per training update | 10
`--epochs` | Number of training epochs | No limit
`--checks-per-epoch` | Number of development evaluations per epoch | 4
`--print-vocabs` | Print the vocabularies before training | Do not print the vocabularies

Any of the DyNet command line options can also be specified.

The training and development trees are assumed to have predicted part-of-speech tags.

For each development evaluation, the loss on the development set is computed and compared to the previous best. If the current model is better, the previous model will be deleted and the current model will be saved. The new filename will be derived from the provided model path base and the development loss.

As an example, to train a path-based parser using the default hyperparameters, you can use the command:

```
python3 src/main.py train --parser-type my --explore --model-path-base models/path-model
```

Alternatively, to train a chart parser using the default hyperparameters, you can use the command:

```
python3 src/main.py train --parser-type chart --model-path-base models/chart-model
```

Compressed pre-trained models with these settings are provided in the `models/zipped/` directory. See the section above for extraction instructions.

## Evaluation

A saved model can be evaluated on a test corpus using the command `python3 src/main.py test ...` with the following arguments:

Argument | Description | Default
--- | --- | ---
`--model-path-base` | Path base of saved model | N/A
`--evalb-dir` |  Path to EVALB directory | `EVALB/`
`--test-path` | Path to test trees | `data/23.auto.clean`

As above, any of the DyNet command line options can also be specified.

The test trees are assumed to have predicted part-of-speech tags.

As an example, after extracting the pre-trained top-down model, you can evaluate it on the test set using the following command:

```
python3 src/main.py test --model-path-base models/top-down-model_dev=92.34
```

The pre-trained top-down model obtains F-scores of 92.34 on the development set and 91.80 on the test set. The pre-trained chart model obtains F-scores of 92.24 on the development set and 91.86 on the test set.

## Parsing New Sentences

The `parse` method of a parser can be used to parse new sentences. In particular, `parser.parse(sentence)` will return a tuple containing the predicted tree and a DyNet expression for the score of the tree under the model. The input sentence should be pre-tagged and represented as a list of (tag, word) pairs.

See the `run_test` function in `src/main.py` for an example of how a parser can be loaded from disk and used to parse sentences.

## Citation

If you use this software for research, please cite our paper as follows:

```

```
