"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import sys
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import time

import numpy as np

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street
import fast_eval
import features

# Build lookup tables now, while the server starts, instead of during the
# first decision (where it would eat into the time bank). The server must
# answer within 15 seconds of starting.
warmup()
fast_eval.build_tables()
fast_eval.equity((0, 1, 2, 3, 4), (), 50, np.random.default_rng(0))  # warm numpy code paths


def _load_one_model(path):
    import json
    from tree_model import TreeModel
    with open(os.path.join(os.path.dirname(path), "q_xgb_meta.json")) as f:
        meta = json.load(f)
    version = features.version_of(meta["feature_names"])
    if version is None:
        raise ValueError(f"{path}: model features don't match any version in features.py")
    model = TreeModel.load(path)
    # Residual models also ship split parts: V(s) (state value) and
    # A(s, a) (per-action part). Q = V + A, but V only matters when FOLD
    # is legal, so evaluating them separately is much cheaper.
    mdir = os.path.dirname(path)
    if os.path.exists(os.path.join(mdir, "a_trees.npz")) and os.path.exists(os.path.join(mdir, "v_trees.npz")):
        model.a_part = TreeModel.load(os.path.join(mdir, "a_trees.npz"))
        model.v_part = TreeModel.load(os.path.join(mdir, "v_trees.npz"))
    else:
        model.a_part = model.v_part = None
    model.feature_version = version
    model.n_features = len(meta["feature_names"])
    return model


def _load_q_model():
    """Load the learned action-value model (models/q_trees.npz, the numpy
    export of the XGBoost model) if it exists and was trained on exactly the
    features features.py computes. Any problem -> None, and the bot plays its
    hand-written rules instead. Set USE_MODEL=0 to force the rules (testing).
    MODEL_PATH may list several models separated by commas: their Q values
    are averaged (merged into one forest, so it costs one prediction)."""
    from tree_model import TreeModel
    here = os.path.dirname(os.path.abspath(__file__))
    paths = [os.path.join(here, p.strip())
             for p in os.environ.get("MODEL_PATH", "models/q_trees.npz").split(",") if p.strip()]
    if os.environ.get("USE_MODEL", "1") == "0" or not paths or not all(os.path.exists(p) for p in paths):
        return None
    try:
        ms = [_load_one_model(p) for p in paths]
        if len({(m.feature_version, m.n_features) for m in ms}) != 1:
            raise ValueError("ensemble members use different feature sets")
        if len(ms) == 1:
            model = ms[0]
        else:
            model = TreeModel.blend(ms)
            if all(m.a_part is not None for m in ms):
                model.a_part = TreeModel.blend([m.a_part for m in ms])
                model.v_part = TreeModel.blend([m.v_part for m in ms])
            else:
                model.a_part = model.v_part = None
            model.feature_version, model.n_features = ms[0].feature_version, ms[0].n_features
        model.predict(np.zeros((3, model.n_features + 3), dtype=np.float32))  # warm up
        return model
    except Exception:
        print(f"[agent] could not load the model:\n{traceback.format_exc()}", file=sys.stderr)
        return None


Q_MODEL = _load_q_model()


def _load_preflop_table():
    """Precomputed preflop equities (models/preflop_eq.npz); None -> simulate."""
    if os.environ.get("USE_PREFLOP_TABLE", "1") == "0":
        return None
    try:
        from preflop_table import PreflopTable
        table = PreflopTable.load()
        table.equity((0, 5, 10, 15, 20))
        return table
    except Exception:
        print(f"[agent] no preflop table, simulating instead:\n{traceback.format_exc()}", file=sys.stderr)
        return None


PREFLOP_TABLE = _load_preflop_table()

# rename as your team/bot's name
PARTICIPANT = "Calypso"
AGENT_NAME = "Coderot Verity"


# ===== YOUR CODE HERE =====

# ---- Tunable parameters ------------------------------------------------------
# Every parameter can be overridden by an environment variable of the same
# name, for local tuning experiments only (e.g. tools/selfplay.py). The
# tournament doesn't set them, so the defaults below are what plays.

def _env(name: str, default):
    try:
        return type(default)(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Monte Carlo samples per equity estimate (~0.6 ms per 300 samples).
SAMPLES_PREFLOP = _env("SAMPLES_PREFLOP", 600)
SAMPLES_POSTFLOP = _env("SAMPLES_POSTFLOP", 400)
# Time management. memory["time_used"] estimates what the judge charges
# (see _account_time); _speed_level() economises when we run ahead of a pace
# of TIME_TARGET seconds per half, and hard caps kick in past HARD_LIMIT_1/2.
TIME_TARGET = _env("TIME_TARGET", 4.5)
CALL_OVERHEAD = _env("CALL_OVERHEAD", 0.0002)   # network/serialisation we can't see
HARD_LIMIT_1 = _env("HARD_LIMIT_1", 6.0)   # past this many seconds: rules only
HARD_LIMIT_2 = _env("HARD_LIMIT_2", 7.0)   # past this: rules with minimal samples
DEBUG_TIME = _env("DEBUG_TIME", 0)        # print time accounting to stderr

# Preflop thresholds, in equity vs a random hand. Reference percentiles:
# 20% -> 0.445, 50% -> 0.50, 70% -> 0.538, 80% -> 0.56, 90% -> 0.585, 95% -> 0.60
PRE_OPEN_RAISE = _env("PRE_OPEN_RAISE", 0.535)    # button: raise top ~30%
PRE_OPEN_CALL = _env("PRE_OPEN_CALL", 0.430)      # button: complete the small blind with top ~85%
PRE_BB_RAISE = _env("PRE_BB_RAISE", 0.565)        # big blind after a limp: raise top ~18%
PRE_3BET = _env("PRE_3BET", 0.585)                # facing a raise: re-raise top ~10% (+0.02 per extra raise)
PRE_CALL_MARGIN = _env("PRE_CALL_MARGIN", 0.10)   # facing a raise: call if equity >= pot odds + this

# Postflop thresholds, in equity vs the opponent's (weighted) range.
POST_BET_POT = _env("POST_BET_POT", 0.72)
POST_BET_HALF = _env("POST_BET_HALF", 0.60)
POST_RAISE = _env("POST_RAISE", 0.72)             # raise when facing a bet (+0.05 per raise already made)
POST_CALL_MARGIN = _env("POST_CALL_MARGIN", 0.03)
BLUFF_FREQ = _env("BLUFF_FREQ", 0.25)             # base frequency of stabs when checked to
RIVER_BLUFF_MAX_EQ = _env("RIVER_BLUFF_MAX_EQ", 0.25)  # only bluff the river with hands this weak

# Feature flags for the quick fixes (1 = on), so each can be A/B tested.
FIX_RIVER_BLUFF = _env("FIX_RIVER_BLUFF", 1)      # medium-strength river hands check instead of bluffing
FIX_CHECK_RANGE = _env("FIX_CHECK_RANGE", 1)      # don't assume strength after the opponent checks

# Learned model (models/q_xgb.ubj). It predicts the bb won from here for each
# action; the bot overrides its hand-written rules only when the model thinks
# another action is better by more than MODEL_MARGIN bb.
MODEL_MARGIN = _env("MODEL_MARGIN", 0.0)
MODEL_STREETS = _env("MODEL_STREETS", "0123")     # streets the model may act on (0 = preflop)


def new_half(config: AgentConfig, memory: dict) -> None:
    """Called at the start of each half-match (1000 hands)."""
    memory["rng"] = np.random.default_rng(20261107)
    memory["time_used"] = 0.0
    memory["hand"] = -1          # hand index the per-hand state belongs to
    memory["seen_hist"] = 0      # how much of action_history we've already counted
    memory["cache"] = {}         # per-hand equity cache
    memory["i_bet"] = False      # did we bet/raise this hand?
    # Opponent model (counts over the half).
    memory["opp_actions"] = 0
    memory["opp_raises"] = 0
    memory["our_bet_hands"] = 0  # hands where we bet and the hand ended
    memory["opp_folds"] = 0      # ...and the opponent folded
    memory["showdowns"] = 0
    memory["exceptions"] = 0     # strategy errors caught by choose_action()
    # Situation-specific opponent stats (feature version 2).
    for k in ("pf_actions", "pf_raises", "post_unopened", "post_bets", "post_facing",
              "post_raise_vs", "post_folds", "aggr_sd_n", "pas_sd_n"):
        memory[k] = 0
    memory["aggr_sd_sum"] = 0.0
    memory["pas_sd_sum"] = 0.0


# ---- Helpers ------------------------------------------------------------------

def _start_hand_if_new(obs: Observation, memory: dict) -> None:
    if memory["hand"] != obs.hand_index:
        memory["hand"] = obs.hand_index
        memory["seen_hist"] = 0
        memory["cache"] = {}
        memory["i_bet"] = False
        # chips each player has in on each street: [me, opp]; blinds preflop
        memory["tot"] = {0: [1, 2] if obs.is_button else [2, 1]}
        memory["opp_late_aggr"] = False   # opponent bet/raised on the turn or river


def _update_opp_model(obs: Observation, memory: dict) -> None:
    """Count opponent actions we haven't seen yet in this hand's history."""
    hist = obs.action_history
    tot = memory.setdefault("tot", {0: [0, 0]})
    for a in hist[memory["seen_hist"]:]:
        t = tot.setdefault(int(a.street), [0, 0])
        if a.is_me:
            t[0] = a.street_total
            continue
        raised = a.action in (Action.RAISE_HALF, Action.RAISE_POT)
        facing = t[0] > t[1]
        memory["opp_actions"] += 1
        memory["opp_raises"] += raised
        if a.street == Street.PREFLOP:
            memory["pf_actions"] += 1
            memory["pf_raises"] += raised
        else:
            if facing:
                memory["post_facing"] += 1
                memory["post_raise_vs"] += raised
            else:
                memory["post_unopened"] += 1
                memory["post_bets"] += raised
            if raised and a.street >= Street.TURN:
                memory["opp_late_aggr"] = True
        t[1] = a.street_total
    memory["seen_hist"] = len(hist)


def _aggression(memory: dict) -> float:
    """Opponent's raise frequency, with a prior of ~1/3."""
    return (memory["opp_raises"] + 2) / (memory["opp_actions"] + 6)


def _fold_rate(memory: dict) -> float:
    """How often the opponent folds when we bet, with a prior of ~40%."""
    return (memory["opp_folds"] + 2) / (memory["our_bet_hands"] + 5)


def _range_factor(memory: dict) -> float:
    """How much to believe the opponent's bets: ~1 for a normal opponent,
    lower for maniacs (they bet everything), higher for passive players."""
    return float(np.clip((0.75 - _aggression(memory)) / 0.4, 0.15, 1.3))


def opp_stats(memory: dict) -> dict:
    """The opponent model in the form features.extract() expects."""
    hands = max(memory.get("hand", 0) + 1, 1)
    return {
        "aggression": _aggression(memory),
        "fold_rate": _fold_rate(memory),
        "range_factor": _range_factor(memory),
        "showdown_rate": (memory["showdowns"] + 3) / (hands + 10),
        "experience": min(hands / 200.0, 1.0),
        "pf_raise_rate": (memory["pf_raises"] + 1) / (memory["pf_actions"] + 3),
        "post_bet_rate": (memory["post_bets"] + 1) / (memory["post_unopened"] + 3),
        "post_fold_to_bet": (memory["post_folds"] + 1) / (memory["post_facing"] + 3),
        "post_raise_to_bet": (memory["post_raise_vs"] + 0.5) / (memory["post_facing"] + 3),
        "aggr_sd_strength": (memory["aggr_sd_sum"] + 0.75 * 3) / (memory["aggr_sd_n"] + 3),
        "passive_sd_strength": (memory["pas_sd_sum"] + 0.55 * 3) / (memory["pas_sd_n"] + 3),
        "sd_seen": min((memory["aggr_sd_n"] + memory["pas_sd_n"]) / 50.0, 1.0),
    }


def _speed_level(obs: Observation, memory: dict) -> int:
    """How much to economise, from the estimated judge-charged time so far
    (memory["time_used"], see choose_action) vs a steady pace through the half.
        0 full strength           1 half the samples
        2 120 samples             3 60 samples
        4 rules only, minimal samples (emergency)
    The model stays on through level 3: with only 60 samples it still beats
    the full-strength rules by ~3.6 bb/deal, and the prediction itself is
    cheap next to the Monte Carlo equity."""
    used = memory.get("time_used", 0.0)
    if used > HARD_LIMIT_2:
        return 4
    if used > HARD_LIMIT_1:
        return 3
    allowed = TIME_TARGET * (obs.hand_index + 1) / 1000 + 0.25
    ratio = used / allowed
    if ratio < 1.0:
        return 0
    if ratio < 1.15:
        return 1
    if ratio < 1.3:
        return 2
    return 3


def _n_samples(base: int, obs: Observation, memory: dict) -> int:
    """Monte Carlo samples for the current speed level."""
    level = memory.get("level", 0)
    if level == 0:
        return base
    if level == 1:
        return base // 2
    if level == 2:
        return min(base, 120)
    if level == 3:
        return 60
    return 25


def _equity(obs: Observation, memory: dict, range_k: float, base: int) -> float:
    key = (len(obs.board), round(range_k, 2))
    cache = memory["cache"]
    if key not in cache:
        if not obs.board and PREFLOP_TABLE is not None:
            # Preflop equity vs a random hand is precomputed (range_k is
            # ignored preflop anyway).
            cache[key] = PREFLOP_TABLE.equity(obs.hole_cards)
            return cache[key]
        n = _n_samples(base, obs, memory)
        if round(range_k, 2) in _FEATURE_KS:
            # The model's two equity features come from one shared simulation.
            vals = fast_eval.equity(obs.hole_cards, obs.board, n, memory["rng"], _FEATURE_KS)
            for k, v in zip(_FEATURE_KS, vals):
                cache[(len(obs.board), k)] = v
        else:
            cache[key] = fast_eval.equity(obs.hole_cards, obs.board, n, memory["rng"], range_k)
    return cache[key]


_FEATURE_KS = tuple(round(k, 2) for k in features.EQ_KS)


def _opp_street_raises(obs: Observation) -> int:
    return sum(1 for a in obs.action_history
               if a.street == obs.street and not a.is_me
               and a.action in (Action.RAISE_HALF, Action.RAISE_POT))


def _opp_checked_this_street(obs: Observation) -> bool:
    """True if the opponent has already checked on this street."""
    return any(a.street == obs.street and not a.is_me
               and a.action == Action.CHECK_CALL and a.amount == 0
               for a in obs.action_history)


def _opp_was_last_aggressor(obs: Observation) -> bool:
    """True if the most recent raise in this hand was the opponent's."""
    for a in reversed(obs.action_history):
        if a.action in (Action.RAISE_HALF, Action.RAISE_POT):
            return not a.is_me
    return False


def _raise(obs: Observation, memory: dict, big: bool) -> Action:
    """Pick a raise size, falling back to whatever raise (or call) is legal."""
    order = [Action.RAISE_POT, Action.RAISE_HALF] if big else [Action.RAISE_HALF, Action.RAISE_POT]
    for a in order:
        if a in obs.legal_actions:
            return a
    return Action.CHECK_CALL


def _fold_or_check(obs: Observation) -> Action:
    if obs.to_call == 0 or Action.FOLD not in obs.legal_actions:
        return Action.CHECK_CALL
    return Action.FOLD


def _can_raise(obs: Observation) -> bool:
    return Action.RAISE_HALF in obs.legal_actions or Action.RAISE_POT in obs.legal_actions


# ---- Strategy -----------------------------------------------------------------

def _preflop(obs: Observation, memory: dict) -> Action:
    eq = _equity(obs, memory, 0.0, SAMPLES_PREFLOP)
    pot_odds = obs.to_call / (obs.pot + obs.to_call)
    raises = obs.raises_this_street

    if raises == 0:
        if obs.to_call == 0:                        # big blind, opponent limped
            if eq >= PRE_BB_RAISE and _can_raise(obs):
                return _raise(obs, memory, big=True)
            return Action.CHECK_CALL
        # button, first to act (to_call == small blind gap)
        if eq >= PRE_OPEN_RAISE and _can_raise(obs):
            return _raise(obs, memory, big=True)
        if eq >= PRE_OPEN_CALL:
            return Action.CHECK_CALL
        return _fold_or_check(obs)

    # Facing a raise. Discount our equity for the opponent's stronger range,
    # less so against opponents who raise a lot.
    adj = eq - 0.025 * _opp_street_raises(obs) * _range_factor(memory)
    if adj >= PRE_3BET + 0.02 * (raises - 1) and _can_raise(obs):
        return _raise(obs, memory, big=True)
    if adj >= pot_odds + PRE_CALL_MARGIN:
        return Action.CHECK_CALL
    return _fold_or_check(obs)


def _postflop(obs: Observation, memory: dict) -> Action:
    rf = _range_factor(memory)
    opp_raises = _opp_street_raises(obs)
    pot_odds = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0

    if obs.to_call == 0:
        # Nobody has bet this street. If the opponent checked, their range is
        # NOT stronger than random (if anything weaker): k = 0. If we act
        # first and they were the last aggressor, weigh their range lightly.
        if FIX_CHECK_RANGE:
            if _opp_checked_this_street(obs):
                k = 0.0
            elif _opp_was_last_aggressor(obs):
                k = 0.3 * rf
            else:
                k = 0.0
        else:
            k = 0.3 * rf
        eq = _equity(obs, memory, k, SAMPLES_POSTFLOP)
        if _can_raise(obs):
            if eq >= POST_BET_POT:
                return _raise(obs, memory, big=True)
            if eq >= POST_BET_HALF:
                return _raise(obs, memory, big=False)
            # Bluff / semi-bluff, mostly in position, more often against
            # opponents who fold a lot. Before the river only with a real
            # draw (eq >= 0.40); on the river only with hands too weak to win
            # at showdown, since medium hands gain nothing by betting.
            in_position = obs.is_button
            stab = BLUFF_FREQ * (_fold_rate(memory) / 0.4) * (1.0 if in_position else 0.5)
            if obs.street != Street.RIVER:
                eligible = eq >= 0.40
            elif FIX_RIVER_BLUFF:
                eligible = eq <= RIVER_BLUFF_MAX_EQ
            else:
                eligible = True
            if eligible and memory["rng"].random() < min(stab, 0.6):
                return _raise(obs, memory, big=False)
        return Action.CHECK_CALL

    # Facing a bet: the opponent's range is stronger. Bigger bets and more
    # raises -> stronger range (scaled by how believable this opponent is).
    big_bet = obs.to_call >= 0.75 * (obs.pot - obs.to_call)
    k = (1.0 + 0.6 * max(0, opp_raises - 1) + (0.4 if big_bet else 0.0)) * rf
    eq = _equity(obs, memory, k, SAMPLES_POSTFLOP)

    if eq >= max(POST_RAISE + 0.05 * obs.raises_this_street, pot_odds + 0.25) and _can_raise(obs):
        return _raise(obs, memory, big=True)
    if eq >= pot_odds + POST_CALL_MARGIN:
        return Action.CHECK_CALL
    return _fold_or_check(obs)


_MODEL_ACTIONS = (Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT)


def _model_values(obs: Observation, memory: dict) -> dict:
    """Predicted bb won from here for each legal action (FOLD is exactly 0)."""
    base = SAMPLES_PREFLOP if obs.street == Street.PREFLOP else SAMPLES_POSTFLOP
    nf = Q_MODEL.n_features
    x = features.extract(obs, lambda k: _equity(obs, memory, k, base), opp_stats(memory),
                         version=Q_MODEL.feature_version)
    cands = [a for a in _MODEL_ACTIONS if a in obs.legal_actions]
    rows = np.zeros((len(cands), nf + 3), dtype=np.float32)
    rows[:, :nf] = x
    for i, a in enumerate(cands):
        rows[i, nf + int(a) - 1] = 1.0
    fold_legal = Action.FOLD in obs.legal_actions
    if Q_MODEL.a_part is not None:
        q = Q_MODEL.a_part.predict(rows)
        if fold_legal:      # need absolute values to compare with FOLD = 0
            q = q + Q_MODEL.v_part.predict(rows[:1])[0]
    else:
        q = Q_MODEL.predict(rows)
    values = {a: float(q[i]) for i, a in enumerate(cands)}
    if fold_legal:
        values[Action.FOLD] = 0.0
    return values


def _rules(obs: Observation, memory: dict) -> Action:
    if obs.street == Street.PREFLOP:
        return _preflop(obs, memory)
    return _postflop(obs, memory)


def _decide(obs: Observation, memory: dict) -> Action:
    _start_hand_if_new(obs, memory)
    _update_opp_model(obs, memory)
    use_model = (Q_MODEL is not None and memory.get("level", 0) <= 3
                 and str(int(obs.street)) in MODEL_STREETS)
    if not use_model:
        return _rules(obs, memory)
    if MODEL_MARGIN <= 0:
        # The model always wins ties-or-better, so the rule action is only
        # needed as a fallback: skip computing it (saves an equity estimate).
        try:
            values = _model_values(obs, memory)
            return max(values, key=values.get)
        except Exception:
            _report_exception(memory, "model")
            return _rules(obs, memory)
    rule_action = _rules(obs, memory)
    try:
        values = _model_values(obs, memory)
        best = max(values, key=values.get)
        if best != rule_action and values[best] - values.get(rule_action, 0.0) > MODEL_MARGIN:
            memory["model_overrides"] = memory.get("model_overrides", 0) + 1
            return best
    except Exception:
        _report_exception(memory, "model")
    return rule_action


def choose_action(obs: Observation, memory: dict) -> Action:
    """Called every time it is your turn. Return one of obs.legal_actions.
    (Observation fields are documented in game.py / the participant guide.)"""
    t0 = time.perf_counter()
    c0 = time.process_time()
    try:
        if "rng" not in memory:          # defensive: new_half not run
            new_half(AgentConfig(), memory)
        memory["level"] = _speed_level(obs, memory)
        action = _decide(obs, memory)
        if action not in obs.legal_actions:
            action = Action.CHECK_CALL
        if action in (Action.RAISE_HALF, Action.RAISE_POT):
            memory["i_bet"] = True       # for the opponent fold-rate stat
    except Exception:
        # Never crash: an error costs far more than a slightly bad action.
        # But don't hide it either: print it (visible in the bot's console /
        # tools/arena.sh logs) and count it.
        action = Action.CHECK_CALL
        _report_exception(memory, "choose_action")
    try:
        _account_time(memory, t0, c0)
        if DEBUG_TIME and memory["calls"] % 500 == 0:
            print(f"[agent] hand {obs.hand_index} calls {memory['calls']} decide {memory['raw_time']:.2f}s "
                  f"est charged {memory['time_used']:.2f}s level {memory['level']} ratio {memory.get('ratio', 0):.2f}",
                  file=sys.stderr, flush=True)
    except Exception:
        pass
    return action


def _account_time(memory: dict, t0: float, c0: float) -> None:
    """Estimate what the judge charges for this request.

    The judge times the whole request and charges everything after the first
    1 ms. Our own clock (t0..now) only covers choose_action; the server work
    around it (parsing the request, on_hand_end, building the reply) we
    measure as CPU time this process used since the previous decision ended
    (the bot is idle otherwise). If the CPU is shared, wall time runs ahead
    of CPU time, so that server work is scaled by the wall/CPU ratio we see
    while deciding."""
    t1, c1 = time.perf_counter(), time.process_time()
    wall, cpu = t1 - t0, c1 - c0
    last_c = memory.get("cpu_end")
    server_cpu = max(0.0, c0 - last_c) if last_c is not None else 0.0
    server_cpu = min(server_cpu, 0.05)                 # ignore idle-time artefacts
    memory["wall_sum"] = memory.get("wall_sum", 0.0) + wall
    memory["cpu_sum"] = memory.get("cpu_sum", 0.0) + cpu
    ratio = min(max(memory["wall_sum"] / max(memory["cpu_sum"], 1e-6), 1.0), 8.0)
    memory["ratio"] = ratio
    charged = max(0.0, wall + server_cpu * ratio + CALL_OVERHEAD - 0.001)
    memory["time_used"] = memory.get("time_used", 0.0) + charged
    memory["raw_time"] = memory.get("raw_time", 0.0) + wall
    memory["calls"] = memory.get("calls", 0) + 1
    memory["cpu_end"] = c1


def _report_exception(memory: dict, where: str) -> None:
    try:
        memory["exceptions"] = memory.get("exceptions", 0) + 1
        if memory["exceptions"] <= 20:   # don't flood the log
            print(f"[agent] exception #{memory['exceptions']} in {where}:\n"
                  f"{traceback.format_exc()}", file=sys.stderr, flush=True)
    except Exception:
        pass


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Called once for every finished hand: update the opponent model."""
    try:
        if "rng" not in memory:
            return
        opp_folded = not result.went_to_showdown and result.my_net > 0
        if result.went_to_showdown:
            memory["showdowns"] += 1
        if opp_folded:
            # The opponent's fold ends the hand, so choose_action() never sees
            # it in action_history. Count it here so _aggression() isn't
            # biased toward raises.
            memory["opp_actions"] += 1
            if len(result.board) >= 3:          # folded to a bet after the flop
                memory["post_facing"] += 1
                memory["post_folds"] += 1
            else:
                memory["pf_actions"] += 1
        same_hand = memory.get("hand") == result.hand_index
        if (result.went_to_showdown and result.opp_hole is not None and len(result.board) == 5
                and same_hand and memory.get("level", 0) <= 2):
            # How strong was what they showed down (equity vs a random hand on
            # this board), split by whether they bet the turn/river: a
            # bluff detector.
            strength = fast_eval.equity(result.opp_hole, result.board, 150, memory["rng"])
            if memory.get("opp_late_aggr"):
                memory["aggr_sd_n"] += 1
                memory["aggr_sd_sum"] += strength
            else:
                memory["pas_sd_n"] += 1
                memory["pas_sd_sum"] += strength
        if same_hand and memory.get("i_bet"):
            memory["our_bet_hands"] += 1
            if opp_folded:
                memory["opp_folds"] += 1
    except Exception:
        _report_exception(memory, "on_hand_end")

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
