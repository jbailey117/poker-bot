"""
tree_model.py - run a trained XGBoost model with numpy only.

xgboost's own predict call costs ~0.4 ms per decision, mostly Python-side
overhead. Here every tree is flattened into arrays and all trees are walked
together, level by level, with numpy: ~0.05-0.1 ms for a few rows. It also
means the tournament image doesn't need xgboost installed at all.

Training side (needs xgboost):   export(booster, "models/q_trees.npz")
Play side (numpy only):          m = TreeModel.load("models/q_trees.npz"); m.predict(X)

XGBoost's rule at a split: go left if x < threshold (comparison in float32),
missing values follow `default_left`. Prediction = base_score + sum of leaves
(identity link, as used by reg:squarederror).
"""

import json

import numpy as np


def _booster_trees(booster):
    model = json.loads(booster.save_raw("json"))
    learner = model["learner"]
    objective = learner["objective"]["name"]
    if objective not in ("reg:squarederror", "reg:pseudohubererror", "reg:absoluteerror"):
        raise ValueError(f"unsupported objective {objective}: only identity-link regression")
    base = learner["learner_model_param"]["base_score"]
    base_score = float(json.loads(base)[0]) if base.startswith("[") else float(base)
    return base_score, learner["gradient_booster"]["model"]["trees"]


def export(booster, path: str, base_score: float | None = None) -> None:
    """Flatten an xgboost Booster (reg:squarederror, numeric splits) to .npz.

    `booster` may also be a list [first, second, ...]: their trees are summed
    into one forest. Only the FIRST booster's base_score is used; the others
    must have been trained with base_margin = the previous ones' prediction
    (xgboost ignores base_score when a base_margin is given). Used for the
    residual model: Q(s, a) = V(s) + A(s, a)."""
    boosters = booster if isinstance(booster, (list, tuple)) else [booster]
    first_base, trees = _booster_trees(boosters[0])
    if base_score is None:
        base_score = first_base
    trees = list(trees)
    for b in boosters[1:]:
        trees += _booster_trees(b)[1]

    left, right, feat, cond, dleft, roots = [], [], [], [], [], []
    offset = 0
    max_depth = 0
    for t in trees:
        l = np.asarray(t["left_children"], dtype=np.int64)
        r = np.asarray(t["right_children"], dtype=np.int64)
        n = len(l)
        leaf = l == -1
        # Leaves point to themselves, so extra traversal steps are no-ops.
        idx = np.arange(n)
        left.append(np.where(leaf, idx, l) + offset)
        right.append(np.where(leaf, idx, r) + offset)
        feat.append(np.where(leaf, 0, np.asarray(t["split_indices"], dtype=np.int64)))
        cond.append(np.asarray(t["split_conditions"], dtype=np.float32))
        dleft.append(np.asarray(t["default_left"], dtype=bool))
        roots.append(offset)
        # depth of this tree
        depth = np.zeros(n, dtype=np.int64)
        for i in range(n):
            if not leaf[i]:
                depth[l[i]] = depth[r[i]] = depth[i] + 1
        max_depth = max(max_depth, int(depth.max()))
        offset += n
    is_leaf = np.concatenate([np.asarray(t["left_children"]) == -1 for t in trees])
    np.savez_compressed(
        path,
        left=np.concatenate(left).astype(np.int32), right=np.concatenate(right).astype(np.int32),
        feat=np.concatenate(feat).astype(np.int32), cond=np.concatenate(cond),
        default_left=np.concatenate(dleft), is_leaf=is_leaf,
        roots=np.asarray(roots, dtype=np.int32), base_score=np.float32(base_score),
        max_depth=np.int32(max_depth),
    )


class TreeModel:
    def __init__(self, arrays):
        self.left = arrays["left"]
        self.right = arrays["right"]
        self.feat = arrays["feat"]
        self.cond = arrays["cond"]
        self.default_left = arrays["default_left"]
        self.is_leaf = arrays["is_leaf"]
        self.roots = arrays["roots"]
        self.base = float(arrays["base_score"])
        self.depth = int(arrays["max_depth"])
        # Leaf values live in `cond` at leaf nodes.
        self.leaf_value = np.where(self.is_leaf, self.cond, 0.0).astype(np.float64)
        # Fast path (no NaN in X): child = children[2 * node + go_left]. Leaves
        # get +inf thresholds so they always "go left" to themselves.
        self.children = np.stack([self.right, self.left], axis=1).reshape(-1).astype(np.intp)
        self.cond_fast = np.where(self.is_leaf, np.inf, self.cond).astype(np.float32)
        self.feat_ip = self.feat.astype(np.intp)
        self.roots_ip = self.roots.astype(np.intp)

    @classmethod
    def load(cls, path: str) -> "TreeModel":
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files})

    def save(self, path: str) -> None:
        np.savez_compressed(path, left=self.left, right=self.right, feat=self.feat, cond=self.cond,
                            default_left=self.default_left, is_leaf=self.is_leaf, roots=self.roots,
                            base_score=np.float32(self.base), max_depth=np.int32(self.depth))

    @classmethod
    def blend(cls, models, weights=None) -> "TreeModel":
        """One forest predicting sum(w_i * model_i(x)) (default: the mean)."""
        w = list(weights) if weights is not None else [1.0 / len(models)] * len(models)
        offs = np.cumsum([0] + [len(m.left) for m in models[:-1]])
        cat = lambda name, shift=False: np.concatenate(
            [getattr(m, name) + (o if shift else 0) for m, o in zip(models, offs)])
        cond = np.concatenate([np.where(m.is_leaf, m.cond * wi, m.cond).astype(np.float32)
                               for m, wi in zip(models, w)])
        return cls({"left": cat("left", True).astype(np.int32), "right": cat("right", True).astype(np.int32),
                    "feat": cat("feat"), "cond": cond, "default_left": cat("default_left"),
                    "is_leaf": cat("is_leaf"), "roots": cat("roots", True).astype(np.int32),
                    "base_score": np.float32(sum(wi * m.base for m, wi in zip(models, w))),
                    "max_depth": np.int32(max(m.depth for m in models))})

    @property
    def n_trees(self) -> int:
        return len(self.roots)

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if not np.isnan(X).any():
            return self._predict_fast(X)
        rows = np.arange(X.shape[0])[:, None]
        node = np.broadcast_to(self.roots, (X.shape[0], len(self.roots))).copy()
        for _ in range(self.depth):
            x = X[rows, self.feat[node]]
            go_left = np.where(np.isnan(x), self.default_left[node], x < self.cond[node])
            node = np.where(go_left, self.left[node], self.right[node])
        return self.base + self.leaf_value[node].sum(axis=1)

    def _predict_fast(self, X) -> np.ndarray:
        """Same result as predict() for NaN-free X, with ~half the numpy calls."""
        n, f = X.shape
        Xf = X.reshape(-1)
        feat, cond, children = self.feat_ip, self.cond_fast, self.children
        if n == 1:
            node = self.roots_ip
            for _ in range(self.depth):
                node = children[2 * node + (Xf[feat[node]] < cond[node])]
            return np.asarray([self.base + self.leaf_value[node].sum()])
        off = (np.arange(n, dtype=np.intp) * f)[:, None]
        node = np.broadcast_to(self.roots_ip, (n, len(self.roots_ip)))
        for _ in range(self.depth):
            node = children[2 * node + (Xf[off + feat[node]] < cond[node])]
        return self.base + self.leaf_value[node].sum(axis=1)
