"""
features.py - turns one decision point into a fixed-length feature vector.

Used by BOTH the offline data pipeline (tools/gen_data.py) and, later, by the
bot at play time, so a trained model always sees features computed exactly
the way they were during training. If you change anything here, regenerate
the data (old shards will no longer match) and bump FEATURE_VERSION.

extract(obs, eq_fn, opp) -> np.ndarray[float32] of len(FEATURE_NAMES)
    obs    the Observation
    eq_fn  eq_fn(range_k) -> equity vs an opponent range weighted by
           strength**range_k (0 = random hand). The agent passes its cached
           Monte Carlo function so nothing is computed twice.
    opp    dict of opponent-model stats (see OPP_KEYS); missing keys -> priors
"""

import numpy as np

import fast_eval
from game import Action, Street

FEATURE_VERSION = 1

# Range weightings used for the equity features (postflop). Preflop only the
# random-hand equity exists, and it is copied into the other slot.
EQ_KS = (0.0, 1.5)

OPP_KEYS = ("aggression", "fold_rate", "range_factor", "showdown_rate", "experience")
OPP_PRIORS = {"aggression": 1 / 3, "fold_rate": 0.4, "range_factor": 1.05,
              "showdown_rate": 0.3, "experience": 0.0}

FEATURE_NAMES = [
    # --- street / position
    "street_preflop", "street_flop", "street_turn", "street_river", "is_button",
    # --- money (stacks are 80 chips; amounts normalised by 80 or by the pot)
    "pot_frac", "to_call_frac", "pot_odds", "my_stack_frac", "opp_stack_frac", "spr",
    "half_raise_cost", "pot_raise_cost",      # extra chips a raise costs, / pot (0 if illegal)
    "can_fold", "can_raise",
    # --- betting history
    "raises_this_street", "opp_raises_street", "opp_raises_hand", "my_raises_hand",
    "opp_checked_street", "opp_last_aggressor", "opp_raised_prev_streets",
    # --- equity
    "eq_random", "eq_strong",
    # --- my hole cards
    "hole_pairs", "hole_trips", "hole_max_suit", "hole_suited_pairs", "hole_distinct",
    "hole_conn", "hole_aces", "hole_high", "hole_mean_rank",
    # --- board
    "board_n", "board_paired", "board_max_suit", "board_conn",
    # --- my made hand (category 0..8 / 8, -1 preflop)
    "made_category",
    # --- opponent model
    *("opp_" + k for k in OPP_KEYS),
]
N_FEATURES = len(FEATURE_NAMES)

_RAISES = (Action.RAISE_HALF, Action.RAISE_POT)
# 5-rank windows for straights; ranks 0 (six) .. 8 (ace); A-6-7-8-9 wheel too.
_WINDOWS = [set(range(lo, lo + 5)) for lo in range(5)] + [{8, 0, 1, 2, 3}]


def _conn(ranks) -> int:
    """Most distinct ranks that fit inside one straight window."""
    s = set(ranks)
    return max(len(s & w) for w in _WINDOWS) if s else 0


def _hole_features(hole):
    ranks = [c >> 2 for c in hole]
    suits = [c & 3 for c in hole]
    rc = [ranks.count(r) for r in set(ranks)]
    sc = [suits.count(s) for s in range(4)]
    return [
        sum(1 for c in rc if c >= 2),           # pairs (or better)
        1.0 if max(rc) >= 3 else 0.0,           # trips in hand (bad in Omaha)
        max(sc),                                # most cards of one suit
        sum(1 for c in sc if c >= 2),           # suits with 2+ cards (double-suited = 2)
        len(set(ranks)),                        # distinct ranks
        _conn(ranks),                           # straight connectedness
        ranks.count(8),                         # aces
        sum(1 for r in ranks if r >= 5),        # J, Q, K, A
        sum(ranks) / (8 * 5),                   # mean rank 0..1
    ]


def _board_features(board):
    if not board:
        return [0, 0, 0, 0]
    ranks = [c >> 2 for c in board]
    suits = [c & 3 for c in board]
    return [
        len(board),
        1.0 if len(set(ranks)) < len(ranks) else 0.0,
        max(suits.count(s) for s in range(4)),
        _conn(ranks),
    ]


def _made_category(hole, board) -> float:
    if len(board) < 3:
        return -1.0
    score = int(fast_eval.partial_scores(np.asarray([hole], dtype=np.intp), tuple(board))[0])
    return (score >> 20) / 8.0


def extract(obs, eq_fn, opp: dict | None = None) -> np.ndarray:
    opp = {**OPP_PRIORS, **(opp or {})}
    hist = obs.action_history
    street = obs.street
    pot = max(obs.pot, 1)
    stack0 = 80.0
    legal = obs.legal_actions

    def raise_cost(a):
        if a not in legal:
            return 0.0
        return (obs.raise_to[a] - obs.my_bet) / pot

    opp_raises_street = sum(1 for a in hist if a.street == street and not a.is_me and a.action in _RAISES)
    opp_raises_hand = sum(1 for a in hist if not a.is_me and a.action in _RAISES)
    my_raises_hand = sum(1 for a in hist if a.is_me and a.action in _RAISES)
    opp_checked = any(a.street == street and not a.is_me and a.action == Action.CHECK_CALL
                      and a.amount == 0 for a in hist)
    last_aggr = 0.0
    for a in reversed(hist):
        if a.action in _RAISES:
            last_aggr = 0.0 if a.is_me else 1.0
            break
    opp_raised_prev = 1.0 if any(a.street < street and not a.is_me and a.action in _RAISES
                                 for a in hist) else 0.0

    eq0 = eq_fn(EQ_KS[0])
    eq1 = eq_fn(EQ_KS[1]) if street != Street.PREFLOP else eq0

    f = [
        *(1.0 if street == s else 0.0 for s in (Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER)),
        1.0 if obs.is_button else 0.0,
        obs.pot / (2 * stack0),
        obs.to_call / pot,
        obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0,
        obs.my_stack / stack0,
        obs.opp_stack / stack0,
        min(obs.my_stack / pot, 10.0) / 10.0,
        raise_cost(Action.RAISE_HALF),
        raise_cost(Action.RAISE_POT),
        1.0 if Action.FOLD in legal else 0.0,
        1.0 if any(a in legal for a in _RAISES) else 0.0,
        obs.raises_this_street / 4.0,
        opp_raises_street,
        opp_raises_hand,
        my_raises_hand,
        1.0 if opp_checked else 0.0,
        last_aggr,
        opp_raised_prev,
        eq0,
        eq1,
        *_hole_features(obs.hole_cards),
        *_board_features(obs.board),
        _made_category(obs.hole_cards, obs.board),
        *(float(opp[k]) for k in OPP_KEYS),
    ]
    return np.asarray(f, dtype=np.float32)
