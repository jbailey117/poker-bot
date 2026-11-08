"""
preflop_table.py - precomputed preflop equity (vs a random hand) for every
5-card starting hand, so the bot never simulates preflop.

Hands that differ only by relabelling suits have the same equity, so the
table stores one entry per suit-isomorphism class (canonical form = the
smallest sorted encoding over all 24 suit permutations).

    python preflop_table.py --build            # writes models/preflop_eq.npz (~1 min, all cores)
    from preflop_table import PreflopTable     # play time: PreflopTable.load().equity(hole)
"""

import os
from itertools import combinations, permutations

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "models", "preflop_eq.npz")
_PERMS = [tuple(p) for p in permutations(range(4))]


def canonical_key(cards) -> int:
    """Suit-isomorphism-invariant integer key for a 5-card hand."""
    best = None
    ranks = [c >> 2 for c in cards]
    suits = [c & 3 for c in cards]
    for p in _PERMS:
        mapped = sorted(r * 4 + p[s] for r, s in zip(ranks, suits))
        key = (((mapped[0] * 36 + mapped[1]) * 36 + mapped[2]) * 36 + mapped[3]) * 36 + mapped[4]
        if best is None or key < best:
            best = key
    return best


def decode_key(key: int):
    cards = []
    for _ in range(5):
        cards.append(key % 36)
        key //= 36
    return cards[::-1]


class PreflopTable:
    def __init__(self, keys, eqs):
        self.table = dict(zip(keys.tolist(), eqs.tolist()))
        self._cache = {}

    @classmethod
    def load(cls, path: str = DEFAULT_PATH) -> "PreflopTable":
        with np.load(path) as z:
            return cls(z["keys"], z["eq"])

    def equity(self, hole) -> float:
        hole = tuple(hole)
        v = self._cache.get(hole)
        if v is None:
            v = self.table[canonical_key(hole)]
            if len(self._cache) < 200_000:
                self._cache[hole] = v
        return v


def _work(args):
    keys, n, seed = args
    import fast_eval
    rng = np.random.default_rng(seed)
    return [fast_eval.equity(decode_key(k), (), n, rng) for k in keys]


def build(path: str = DEFAULT_PATH, samples: int = 6000, workers: int = None) -> None:
    import time
    from multiprocessing import Pool
    t0 = time.time()
    keys = sorted({canonical_key(h) for h in combinations(range(36), 5)})
    print(f"{len(keys)} canonical preflop hands; {samples} samples each")
    chunks = [keys[i::200] for i in range(200)]
    with Pool(workers or max(1, (os.cpu_count() or 2) - 2)) as pool:
        results = pool.map(_work, [(c, samples, i) for i, c in enumerate(chunks)])
    eq = {}
    for c, r in zip(chunks, results):
        eq.update(zip(c, r))
    k = np.asarray(keys, dtype=np.int64)
    e = np.asarray([eq[x] for x in keys], dtype=np.float32)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, keys=k, eq=e, samples=np.int32(samples))
    print(f"-> {path} in {time.time() - t0:.0f}s; equity range {e.min():.3f}..{e.max():.3f}")


if __name__ == "__main__":
    import sys
    if "--build" in sys.argv:
        build()
    else:
        print(__doc__)
