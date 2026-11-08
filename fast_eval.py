"""
fast_eval.py - vectorized (numpy) Omaha hand scoring and Monte Carlo equity.

Why: evaluator.omaha_score takes ~30 us per call in pure Python, which only
allows a few dozen Monte Carlo samples per decision. This module scores
thousands of hands per millisecond by turning every 5-card hand into two
array indices that can be summed:

    rank key = sum(5 ** rank)   (unique per rank multiset: each rank appears <= 4 times)
    suit key = sum(6 ** suit)   (all five cards share a suit  <=>  key == 5 * 6**s)

Because both keys are sums, the 100 Omaha combos (10 hole pairs x 10 board
triples) are just a broadcasted add of pair keys and triple keys, followed by
table lookups. Scores are identical to evaluator.score_five / omaha_score.
"""

from itertools import combinations, combinations_with_replacement

import numpy as np

from evaluator import score_five

NUM_CARDS = 36

_RANK_W = np.array([5 ** r for r in range(9)], dtype=np.int64)
_SUIT_W = np.array([6 ** s for s in range(4)], dtype=np.int64)
CARD_RKEY = np.array([_RANK_W[c >> 2] for c in range(NUM_CARDS)], dtype=np.int32)
CARD_SKEY = np.array([_SUIT_W[c & 3] for c in range(NUM_CARDS)], dtype=np.int16)

_PAIRS = np.array(list(combinations(range(5), 2)), dtype=np.intp)     # (10, 2)
_TRIPLES = np.array(list(combinations(range(5), 3)), dtype=np.intp)   # (10, 3)

_NONFLUSH: np.ndarray | None = None   # rank key -> score when not all one suit
_FLUSH: np.ndarray | None = None      # rank key -> score when all one suit
_IS_FLUSH = np.zeros(6 ** 4, dtype=bool)
for _s in range(4):
    _IS_FLUSH[5 * 6 ** _s] = True


def build_tables() -> None:
    """Build the lookup tables (~0.1 s). Call once at startup."""
    global _NONFLUSH, _FLUSH
    if _NONFLUSH is not None:
        return
    size = 5 ** 9
    nonflush = np.zeros(size, dtype=np.int32)
    flush = np.zeros(size, dtype=np.int32)
    for ranks in combinations_with_replacement(range(9), 5):   # sorted ascending
        if max(ranks.count(r) for r in set(ranks)) > 4:
            continue
        key = sum(5 ** r for r in ranks)
        # Suits cycle 0,1,2,3,0: duplicate ranks are adjacent so they get
        # different suits, and five cards never share one suit.
        cards = [r * 4 + (i % 4) for i, r in enumerate(ranks)]
        nonflush[key] = score_five(cards)
        if len(set(ranks)) == 5:
            flush[key] = score_five([r * 4 for r in ranks])
    _NONFLUSH, _FLUSH = nonflush, flush


def _pair_keys(holes: np.ndarray):
    """holes (N, 5) -> rank and suit keys of the 10 two-card subsets, (N, 10) each."""
    r = CARD_RKEY[holes]
    s = CARD_SKEY[holes]
    return (r[:, _PAIRS[:, 0]] + r[:, _PAIRS[:, 1]],
            s[:, _PAIRS[:, 0]] + s[:, _PAIRS[:, 1]])


def _triple_keys(boards: np.ndarray, triples: np.ndarray = _TRIPLES):
    """boards (N, k) -> rank and suit keys of the given 3-card subsets."""
    r = CARD_RKEY[boards]
    s = CARD_SKEY[boards]
    return (r[:, triples[:, 0]] + r[:, triples[:, 1]] + r[:, triples[:, 2]],
            s[:, triples[:, 0]] + s[:, triples[:, 1]] + s[:, triples[:, 2]])


def _best(pr, ps, tr, ts) -> np.ndarray:
    """Best score over every (pair, triple) combination, per row."""
    rk = pr[:, :, None] + tr[:, None, :]
    sk = ps[:, :, None] + ts[:, None, :]
    scores = np.where(_IS_FLUSH[sk], _FLUSH[rk], _NONFLUSH[rk])
    return scores.reshape(scores.shape[0], -1).max(axis=1)


def omaha_scores(holes: np.ndarray, boards: np.ndarray) -> np.ndarray:
    """Vectorized omaha_score: holes (N, 5), boards (N, 5) -> scores (N,)."""
    build_tables()
    pr, ps = _pair_keys(holes)
    tr, ts = _triple_keys(boards)
    return _best(pr, ps, tr, ts)


def partial_scores(holes: np.ndarray, board: tuple) -> np.ndarray:
    """Made-hand strength of each hole hand on a 3- or 4-card board (no draws).
    Uses exactly 2 hole + 3 board cards like the real rule."""
    build_tables()
    b = np.asarray(board, dtype=np.intp)[None, :]
    triples = np.array(list(combinations(range(len(board)), 3)), dtype=np.intp)
    tr, ts = _triple_keys(b, triples)
    pr, ps = _pair_keys(holes)
    n = holes.shape[0]
    return _best(pr, ps, np.broadcast_to(tr, (n, tr.shape[1])), np.broadcast_to(ts, (n, ts.shape[1])))


def equity(hole, board, n: int, rng: np.random.Generator, range_k: float = 0.0) -> float:
    """Monte Carlo equity of `hole` against one unknown opponent hand.

    Deals the opponent 5 cards and the rest of the board `n` times.
    Ties count as half a win.

    range_k > 0 narrows the opponent's range toward strong hands (used when
    the opponent has bet): on the flop/turn each sampled opponent hand is
    weighted by (percentile of its current made-hand strength) ** range_k.
    Preflop it is ignored.
    """
    build_tables()
    hole = np.asarray(hole, dtype=np.intp)
    board = tuple(int(c) for c in board)
    dead = np.zeros(NUM_CARDS, dtype=bool)
    dead[hole] = True
    if board:
        dead[list(board)] = True
    deck = np.flatnonzero(~dead)
    need = 5 + (5 - len(board))

    # n random draws of `need` distinct cards from the deck.
    draws = deck[np.argpartition(rng.random((n, deck.size)), need, axis=1)[:, :need]]
    opp = draws[:, :5]
    if board:
        full = np.concatenate([np.broadcast_to(np.asarray(board, dtype=np.intp), (n, len(board))),
                               draws[:, 5:]], axis=1)
    else:
        full = draws[:, 5:]

    tr, ts = _triple_keys(full)
    pr_me, ps_me = _pair_keys(hole[None, :])
    me = _best(np.broadcast_to(pr_me, (n, 10)), np.broadcast_to(ps_me, (n, 10)), tr, ts)
    pr_op, ps_op = _pair_keys(opp)
    op = _best(pr_op, ps_op, tr, ts)
    result = (me > op).astype(np.float64) + 0.5 * (me == op)

    if range_k > 0 and len(board) >= 3:
        strength = partial_scores(opp, board)
        pct = (np.argsort(np.argsort(strength, kind="stable"), kind="stable") + 1.0) / n
        w = pct ** range_k
        return float((result * w).sum() / w.sum())
    return float(result.mean())
