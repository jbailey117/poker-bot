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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for game.py etc.
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import time

import numpy as np

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street
import fast_eval

# Build lookup tables now, while the server starts, instead of during the
# first decision (where it would eat into the time bank). The server must
# answer within 15 seconds of starting.
warmup()
fast_eval.build_tables()
fast_eval.equity((0, 1, 2, 3, 4), (), 50, np.random.default_rng(0))  # warm numpy code paths

# rename as your team/bot's name
PARTICIPANT = "TotallyAwesomeDatathonParticipant"
AGENT_NAME = "EquityBot-v1-frozen"


# ===== YOUR CODE HERE =====

# ---- Tunable parameters ------------------------------------------------------
# Monte Carlo samples per equity estimate (~0.6 ms per 300 samples).
# (Environment variables override these for local tuning experiments only;
# the tournament doesn't set them, so the defaults are what plays.)
SAMPLES_PREFLOP = int(os.environ.get("SAMPLES_PREFLOP", 600))
SAMPLES_POSTFLOP = int(os.environ.get("SAMPLES_POSTFLOP", 400))
# Time management: aim to use at most this much of the 8 s bank per half.
TIME_TARGET = float(os.environ.get("TIME_TARGET", 4.0))

# Preflop thresholds, in equity vs a random hand. Reference percentiles:
# 20% -> 0.445, 50% -> 0.50, 70% -> 0.538, 80% -> 0.56, 90% -> 0.585, 95% -> 0.60
PRE_OPEN_RAISE = 0.535   # button: raise top ~30%
PRE_OPEN_CALL = 0.430    # button: complete the small blind with top ~85%
PRE_BB_RAISE = 0.565     # big blind after a limp: raise top ~18%
PRE_3BET = 0.585         # facing a raise: re-raise top ~10% (+0.02 per extra raise)
PRE_CALL_MARGIN = 0.10   # facing a raise: call if equity >= pot odds + this

# Postflop thresholds, in equity vs the opponent's (weighted) range.
POST_BET_POT = 0.72
POST_BET_HALF = 0.60
POST_RAISE = 0.72        # raise when facing a bet (+0.05 per raise already made)
POST_CALL_MARGIN = 0.03
BLUFF_FREQ = 0.25        # base frequency of in-position stabs when checked to


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


# ---- Helpers ------------------------------------------------------------------

def _start_hand_if_new(obs: Observation, memory: dict) -> None:
    if memory["hand"] != obs.hand_index:
        memory["hand"] = obs.hand_index
        memory["seen_hist"] = 0
        memory["cache"] = {}
        memory["i_bet"] = False


def _update_opp_model(obs: Observation, memory: dict) -> None:
    """Count opponent actions we haven't seen yet in this hand's history."""
    hist = obs.action_history
    for a in hist[memory["seen_hist"]:]:
        if not a.is_me:
            memory["opp_actions"] += 1
            if a.action in (Action.RAISE_HALF, Action.RAISE_POT):
                memory["opp_raises"] += 1
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


def _n_samples(base: int, obs: Observation, memory: dict) -> int:
    """Scale Monte Carlo work down if we're using the time bank too fast."""
    used = memory["time_used"]
    if used > 7.0:
        return 30
    if used > 6.0:
        return max(60, base // 6)
    allowed = TIME_TARGET * (obs.hand_index + 1) / 1000 + 0.3
    if used > allowed:
        return max(100, base // 3)
    return base


def _equity(obs: Observation, memory: dict, range_k: float, base: int) -> float:
    key = (len(obs.board), round(range_k, 2))
    cache = memory["cache"]
    if key not in cache:
        n = _n_samples(base, obs, memory)
        cache[key] = fast_eval.equity(obs.hole_cards, obs.board, n, memory["rng"], range_k)
    return cache[key]


def _opp_street_raises(obs: Observation) -> int:
    return sum(1 for a in obs.action_history
               if a.street == obs.street and not a.is_me
               and a.action in (Action.RAISE_HALF, Action.RAISE_POT))


def _raise(obs: Observation, memory: dict, big: bool) -> Action:
    """Pick a raise size, falling back to whatever raise (or call) is legal."""
    order = [Action.RAISE_POT, Action.RAISE_HALF] if big else [Action.RAISE_HALF, Action.RAISE_POT]
    for a in order:
        if a in obs.legal_actions:
            memory["i_bet"] = True
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
        # Nobody has bet this street: weigh the opponent's range only lightly.
        eq = _equity(obs, memory, 0.3 * rf, SAMPLES_POSTFLOP)
        if _can_raise(obs):
            if eq >= POST_BET_POT:
                return _raise(obs, memory, big=True)
            if eq >= POST_BET_HALF:
                return _raise(obs, memory, big=False)
            # Bluff / semi-bluff in position (or as the preflop aggressor),
            # more often against opponents who fold a lot.
            in_position = obs.is_button
            stab = BLUFF_FREQ * (_fold_rate(memory) / 0.4) * (1.0 if in_position else 0.5)
            min_eq = 0.40 if obs.street != Street.RIVER else 0.0
            if eq >= min_eq and memory["rng"].random() < min(stab, 0.6):
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


def _decide(obs: Observation, memory: dict) -> Action:
    _start_hand_if_new(obs, memory)
    _update_opp_model(obs, memory)
    if obs.street == Street.PREFLOP:
        return _preflop(obs, memory)
    return _postflop(obs, memory)


def choose_action(obs: Observation, memory: dict) -> Action:
    """Called every time it is your turn. Return one of obs.legal_actions.
    (Observation fields are documented in game.py / the participant guide.)"""
    t0 = time.perf_counter()
    try:
        if "rng" not in memory:          # defensive: new_half not run
            new_half(AgentConfig(), memory)
        action = _decide(obs, memory)
        if action not in obs.legal_actions:
            action = Action.CHECK_CALL
    except Exception:
        # Never crash: an error costs far more than a slightly bad action.
        action = Action.CHECK_CALL
    try:
        memory["time_used"] = memory.get("time_used", 0.0) + (time.perf_counter() - t0)
    except Exception:
        pass
    return action


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Called once for every finished hand: update the opponent model."""
    try:
        if "rng" not in memory:
            return
        if result.went_to_showdown:
            memory["showdowns"] += 1
        if memory.get("hand") == result.hand_index and memory.get("i_bet"):
            memory["our_bet_hands"] += 1
            if not result.went_to_showdown and result.my_net > 0:
                memory["opp_folds"] += 1
    except Exception:
        pass

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
