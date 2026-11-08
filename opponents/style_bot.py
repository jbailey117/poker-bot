"""
style_bot.py - held-out opponent styles for testing (see STYLE below).

    STYLE=nit PORT=5009 python opponents/style_bot.py
    python tools/selfplay.py --a-file opponents/style_bot.py --a "STYLE=balanced" --b ""
"""
import os
import sys
import traceback

# This file lives in baseline-bots/, so look one folder up for game.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from game import Action, AgentConfig, HandResult, Observation, Street

PARTICIPANT = "Challenges Team"
AGENT_NAME = "RandomBot"


# ===== YOUR CODE HERE =====
# Opponent styles that are NOT in v4's training pool, written to resemble
# plausible bots from other teams. Pick one with the STYLE env variable:
#   nit       very tight: plays few hands, bets only strong hands, never bluffs
#   potodds   straightforward "equity vs a random hand compared to pot odds"
#   balanced  tight-aggressive with a measured share of bluffs and bluff-raises
#   trapper   slow-plays strong hands (check/call), springs a pot raise on the river
#   guide     the minimal example bot from the participant guide (section 4.1)
import numpy as np

import fast_eval
from evaluator import FLUSH, omaha_score

STYLE = os.environ.get("STYLE", "potodds")
AGENT_NAME = f"Style-{STYLE}"
fast_eval.build_tables()
RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


def new_half(config: AgentConfig, memory: dict) -> None:
    memory["rng"] = np.random.default_rng(sum(map(ord, STYLE)))
    memory["cache"] = {}
    memory["hand"] = -1


def _eq(obs, memory, k=0.0, n=400):
    if memory["hand"] != obs.hand_index:
        memory["hand"] = obs.hand_index
        memory["cache"] = {}
    key = (len(obs.board), k)
    if key not in memory["cache"]:
        memory["cache"][key] = fast_eval.equity(obs.hole_cards, obs.board, n, memory["rng"], k)
    return memory["cache"][key]


def _do(obs, *prefs):
    """First legal action among prefs; FOLD only when it is legal, else check."""
    for a in prefs:
        if a in obs.legal_actions:
            return a
    return Action.CHECK_CALL


def _fold(obs):
    return _do(obs, Action.FOLD) if obs.to_call > 0 else Action.CHECK_CALL


def _pot_odds(obs):
    return obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0


def _nit(obs, memory):
    eq = _eq(obs, memory)
    if obs.street == Street.PREFLOP:
        if obs.raises_this_street == 0:
            if obs.to_call == 0:
                return _do(obs, Action.RAISE_POT) if eq >= 0.58 else Action.CHECK_CALL
            if eq >= 0.56:
                return _do(obs, Action.RAISE_POT)
            return Action.CHECK_CALL if eq >= 0.50 else _fold(obs)
        if eq >= 0.60:
            return _do(obs, Action.RAISE_POT)
        return Action.CHECK_CALL if eq >= 0.53 and _pot_odds(obs) < 0.36 else _fold(obs)
    if obs.to_call == 0:
        if eq >= 0.80:
            return _do(obs, Action.RAISE_POT)
        return _do(obs, Action.RAISE_HALF) if eq >= 0.70 else Action.CHECK_CALL
    if eq >= 0.85:
        return _do(obs, Action.RAISE_POT)
    return Action.CHECK_CALL if eq >= max(0.60, _pot_odds(obs) + 0.25) else _fold(obs)


def _potodds(obs, memory):
    eq = _eq(obs, memory)
    po = _pot_odds(obs)
    if obs.street == Street.PREFLOP:
        if eq >= 0.55 and obs.raises_this_street < 2:
            return _do(obs, Action.RAISE_HALF)
        return Action.CHECK_CALL if eq >= po or obs.to_call <= 1 else _fold(obs)
    if obs.to_call == 0:
        if eq >= 0.75:
            return _do(obs, Action.RAISE_POT)
        return _do(obs, Action.RAISE_HALF) if eq >= 0.60 else Action.CHECK_CALL
    if eq >= 0.80:
        return _do(obs, Action.RAISE_POT)
    return Action.CHECK_CALL if eq >= po else _fold(obs)


def _balanced(obs, memory):
    rng = memory["rng"]
    po = _pot_odds(obs)
    if obs.street == Street.PREFLOP:
        eq = _eq(obs, memory)
        if obs.raises_this_street == 0:
            if eq >= 0.54 or (eq < 0.45 and rng.random() < 0.10):     # value + bluff opens
                return _do(obs, Action.RAISE_POT)
            return Action.CHECK_CALL if eq >= 0.44 or obs.to_call == 0 else _fold(obs)
        if eq >= 0.59 or (eq < 0.47 and rng.random() < 0.06):
            return _do(obs, Action.RAISE_POT)
        return Action.CHECK_CALL if eq >= po + 0.08 else _fold(obs)
    if obs.to_call == 0:
        eq = _eq(obs, memory, 0.0)
        if eq >= 0.80:
            return _do(obs, Action.RAISE_POT)
        if eq >= 0.65:
            return _do(obs, Action.RAISE_HALF)
        bluff_p = 0.40 if obs.is_button else 0.20
        if eq < 0.30 and rng.random() < bluff_p:
            return _do(obs, Action.RAISE_HALF if obs.street != Street.RIVER else Action.RAISE_POT)
        return Action.CHECK_CALL
    eq = _eq(obs, memory, 1.0)            # facing a bet: assume a stronger range
    if eq >= 0.78 or (eq < 0.2 and obs.street != Street.RIVER and rng.random() < 0.05):
        return _do(obs, Action.RAISE_POT)
    return Action.CHECK_CALL if eq >= po + 0.05 else _fold(obs)


def _trapper(obs, memory):
    eq = _eq(obs, memory)
    po = _pot_odds(obs)
    if obs.street == Street.PREFLOP:
        if eq >= 0.62:
            return _do(obs, Action.RAISE_POT)
        return Action.CHECK_CALL if eq >= 0.45 or obs.to_call == 0 or eq >= po + 0.1 else _fold(obs)
    if eq >= 0.78:                              # monster: slow-play until the river
        if obs.street == Street.RIVER:
            return _do(obs, Action.RAISE_POT)
        return Action.CHECK_CALL
    if obs.to_call == 0:
        return _do(obs, Action.RAISE_HALF) if 0.55 <= eq else Action.CHECK_CALL
    return Action.CHECK_CALL if eq >= po + 0.08 else _fold(obs)


def _guide(obs, memory):
    if obs.street != Street.RIVER:
        if obs.to_call <= obs.big_blind:
            return Action.CHECK_CALL
        return _fold(obs)
    score = omaha_score(obs.hole_cards, obs.board)
    if score >= (FLUSH << 20) and Action.RAISE_POT in obs.legal_actions:
        return Action.RAISE_POT
    return Action.CHECK_CALL


_STYLES = {"nit": _nit, "potodds": _potodds, "balanced": _balanced, "trapper": _trapper, "guide": _guide}


def choose_action(obs: Observation, memory: dict) -> Action:
    try:
        a = _STYLES[STYLE](obs, memory)
        return a if a in obs.legal_actions else Action.CHECK_CALL
    except Exception:
        traceback.print_exc()
        return Action.CHECK_CALL


def on_hand_end(result: HandResult, memory: dict) -> None:
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
# Keep your per-half state in `memory`: other global variables are NOT reset
# between halves.

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
    # Locally it defaults to 5009 for your own testing convenience
    port = int(os.environ.get("PORT", "5009"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
