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

    @classmethod
    def load(cls, path: str) -> "TreeModel":
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files})

    @property
    def n_trees(self) -> int:
        return len(self.roots)

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        rows = np.arange(X.shape[0])[:, None]
        node = np.broadcast_to(self.roots, (X.shape[0], len(self.roots))).copy()
        for _ in range(self.depth):
            x = X[rows, self.feat[node]]
            go_left = np.where(np.isnan(x), self.default_left[node], x < self.cond[node])
            node = np.where(go_left, self.left[node], self.right[node])
        return self.base + self.leaf_value[node].sum(axis=1)
