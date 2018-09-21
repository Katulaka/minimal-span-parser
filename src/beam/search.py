"""Beam search module.

Beam search takes the top K results from the model, predicts the K results for
each of the previous K result, getting K*K results. Pick the top K results from
K*K results, and start over again until certain number of results are fully
decoded.
"""

from six.moves import xrange
import numpy as np
import dynet as dy
import math


class Hypothesis(object):
    """Defines a hypothesis during beam search."""

    def __init__(self, tokens, prob, state, score=None):
        """Hypothesis constructor.

        Args:
          tokens: start tokens for decoding.
          prob: prob of the start tokens, usually 1.
          state: decoder initial states.
          score: decoder intial score.
        """
        self.tokens = tokens
        self.prob = prob
        self.state = state
        self.score = math.log(prob[-1]) if score is None else score

    def extend_(self, token, prob, new_state):
        """Extend the hypothesis with result from latest step.

        Args:
          token: latest token from decoding.
          prob: prob of the latest decoded tokens.
          new_state: decoder output state. Fed to the decoder for next step.
        Returns:
          New Hypothesis with the results from latest step.
        """
        tokens = self.tokens + [token]
        probs = self.prob + [prob]
        score = self.score + math.log(prob)
        return Hypothesis(tokens, probs, new_state, score)

    @property
    def latest_token(self):
        return self.tokens[-1]

    def __str__(self):
        return ('Hypothesis(prob = {:4f}, tokens = {})'.format(
                        self.prob, self.tokens)
                )

class BeamSearch(object):
    """Beam search."""

    def __init__(self, start_token, end_token, beam_size, max_steps):
        """Creates BeamSearch object.

        Args:
          beam_size: int.
          start_token: int, id of the token to start decoding with
          end_token: int, id of the token that completes an hypothesis
          max_steps: int, upper limit on the size of the hypothesis
        """
        self._beam_size = beam_size
        self._start_token = start_token
        self._end_token = end_token
        self._max_steps = max_steps

    def beam_search(self, encode_outputs, label_embeddings, dec_lstm, ws):
        """Performs beam search for decoding.

         Args:
            encode_outputs:
            label_embeddings:
            dec_lstm:
            ws:
         Returns:
            hyps: list of Hypothesis, the best hypotheses found by beam search,
                    ordered by score
         """

        hyps_per_sentence = []
        #iterate over words in seq
        _encode_outputs = dy.concatenate_cols(encode_outputs)
        query = dy.transpose(dy.rectify(dy.affine_transform([*ws[0], _encode_outputs])))
        for encode_output in encode_outputs:
            encode_output_zeros = dy.zeros(encode_output.dim()[0])
            intial_state = dec_lstm.initial_state(
                                [encode_output, encode_output_zeros])
            complete_hyps = []
            hyps = [Hypothesis([self._start_token], [1.0], intial_state)]
            for steps in xrange(self._max_steps):
                if hyps != []:
                    # Extend each hypothesis.
                    # The first step takes the best K results from first hyps.
                    # Following steps take the best K results from K*K hyps.
                    all_hyps = []
                    for hyp in hyps:
                        label_embedding = label_embeddings[hyp.latest_token]
                        new_state = hyp.state.add_input(label_embedding)
                        decode_output = new_state.output()
                        key = dy.rectify(dy.affine_transform([*ws[1], decode_output]))
                        alpha = dy.softmax(query * key)
                        context = _encode_outputs * alpha
                        x = dy.concatenate([decode_output, context])
                        _probs = dy.softmax(dy.affine_transform([*ws[2], x]))
                        probs = _probs.npvalue()
                        top_ids = np.argsort(probs)[-self._beam_size:]
                        top_probs = probs[top_ids]
                        all_hyps.extend([hyp.extend_(idx, prob, new_state)
                                    for idx, prob in zip(top_ids, top_probs)])
                    hyps = []

                    for h in self.best_hyps(all_hyps):
                        # Filter and collect any hypotheses that have the end token.
                        if h.latest_token == self._end_token and len(h.tokens)>2:
                            # Pull the hypothesis off the beam
                            #if the end token is reached.
                            complete_hyps.append(h)
                        elif h.latest_token == self._end_token:
                            pass
                        elif len(complete_hyps) >= self._beam_size \
                            and h.score < min(complete_hyps, key=lambda h: h.score).score:
                            pass
                        else:
                            # Otherwise continue to the extend the hypothesis.
                            hyps.append(h)
            hyps_per_word = self.best_hyps(complete_hyps)
            hyps_per_sentence.append([(h.tokens[1:-1], h.score) for h in hyps_per_word])
        return hyps_per_sentence

    def best_hyps(self, hyps):
        """return top <beam_size> hyps.

        Args:
          hyps: A list of hypothesis.
        Returns:
          hyps: A sub list of top <beam_size> hyps.
        """
        return sorted(hyps, key=lambda h: h.score, reverse=True)[:self._beam_size]
