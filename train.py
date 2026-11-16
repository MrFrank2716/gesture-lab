#!/usr/bin/env python3
"""Train a first gesture classifier from Gesture Lab dataset files.

    pip install numpy pandas scikit-learn joblib
    python train.py sample-dataset.csv
    python train.py p01.csv p02.csv p03.csv --depth 4 --model gestures.joblib

Each dataset CSV (the "Save dataset CSV" button) holds many short clips, one labelled gesture
per clip. This script turns every clip into one row of summary features, then reports how well
three things tell the gestures apart:

  rules   the hand-tuned detector that ran in the page while recording (the baseline to beat)
  tree    a small decision tree, printed in full so you can read it. Trees this size are the
          kind that fit inside an IMU's built-in machine learning core.
  forest  a random forest, usually the most accurate of the three

With three or more participants the test is "leave one person out": train on everyone else,
test on a person the model has never seen. That is the honest number for a ride, where every
guest is a stranger. With fewer participants it falls back to 5-fold testing, which flatters
the model because the same person's clips are on both sides.
"""
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_text

CHANNELS = ["rot_x", "rot_y", "rot_z", "acc_x", "acc_y", "acc_z", "yaw_rate", "tilt_rate", "acc_h", "acc_v"]


def clip_features(c):
    """One clip (a DataFrame of samples) -> a dict of numbers that describe it."""
    t = c["t_ms"].to_numpy() / 1000.0
    dt = np.diff(t, prepend=t[0])
    f = {}
    for ch in CHANNELS:
        v = c[ch].to_numpy()
        f[ch + "_mean"], f[ch + "_std"] = v.mean(), v.std()
        f[ch + "_min"], f[ch + "_max"] = v.min(), v.max()
    turn = np.sqrt(c["rot_x"] ** 2 + c["rot_y"] ** 2 + c["rot_z"] ** 2).to_numpy()
    f["turn_peak"] = turn.max()                                   # deg/s
    f["turn_total"] = float((turn * dt).sum())                    # deg swept in any direction
    f["yaw_sweep"] = abs(float((c["yaw_rate"].to_numpy() * dt).sum()))   # deg about vertical
    f["tilt_sweep"] = float((c["tilt_rate"].to_numpy() * dt).sum())      # deg about horizontal
    f["yaw_share"] = f["yaw_sweep"] / (f["yaw_sweep"] + f["tilt_sweep"] + 1e-9)
    f["busy_s"] = float(dt[turn > 60].sum())                      # seconds spent moving
    return f


def load(paths):
    df = pd.concat([pd.read_csv(p, dtype={"rule_event": str}) for p in paths], ignore_index=True)
    rows, meta = [], []
    for cid, c in df.groupby("clip_id", sort=False):
        rows.append(clip_features(c))
        ev = c["rule_event"].dropna()
        meta.append({"clip_id": cid, "label": c["label"].iloc[0], "participant": str(c["participant"].iloc[0]),
                     "rule": ev.iloc[0] if len(ev) else "none"})
    return pd.DataFrame(rows), pd.DataFrame(meta)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="+", help="dataset CSV files saved from Gesture Lab")
    ap.add_argument("--depth", type=int, default=4, help="maximum depth of the readable tree (default 4)")
    ap.add_argument("--model", help="save the trained random forest to this .joblib file")
    a = ap.parse_args()

    X, meta = load(a.csv)
    y, people = meta["label"].to_numpy(), meta["participant"].to_numpy()
    labels = sorted(set(y))
    counts = meta["label"].value_counts()
    print(f"{len(meta)} clips, {len(labels)} gestures, {len(set(people))} participant(s)")
    print(counts.to_string(), "\n")
    if len(labels) < 2:
        sys.exit("Need at least two different gestures to train.")

    if len(set(people)) >= 3:
        cv, groups, how = LeaveOneGroupOut(), people, "leave one person out"
    else:
        k = min(5, int(counts.min()))
        if k < 2:
            sys.exit("Need at least two clips of every gesture.")
        cv, groups, how = StratifiedKFold(k, shuffle=True, random_state=0), None, f"{k}-fold (optimistic: fewer than 3 participants)"
    print("Test method:", how)

    # rule baseline: slide/chop clips count as right if the rules fired that gesture, others if the rules stayed quiet
    rule_ok = np.where(np.isin(y, ["slide", "chop"]), meta["rule"] == y, meta["rule"] == "none")
    print(f"  rules   {100 * rule_ok.mean():5.1f}% of clips handled correctly")

    tree = DecisionTreeClassifier(max_depth=a.depth, min_samples_leaf=3, random_state=0)
    forest = RandomForestClassifier(n_estimators=200, random_state=0)
    pred = {}
    for name, model in (("tree", tree), ("forest", forest)):
        pred[name] = cross_val_predict(model, X, y, cv=cv, groups=groups)
        print(f"  {name:7s} {100 * accuracy_score(y, pred[name]):5.1f}% accuracy")

    cm = pd.DataFrame(confusion_matrix(y, pred["forest"], labels=labels), index=labels, columns=labels)
    print("\nForest confusion table (rows: what it was, columns: what the model said)")
    print(cm.to_string())

    tree.fit(X, y)
    forest.fit(X, y)
    print(f"\nDecision tree trained on every clip (depth {a.depth}):")
    print(export_text(tree, feature_names=list(X.columns), decimals=1))
    top = pd.Series(forest.feature_importances_, index=X.columns).sort_values(ascending=False).head(8)
    print("Features the forest leans on most:")
    print(top.round(3).to_string())

    if a.model:
        import joblib
        joblib.dump({"model": forest, "features": list(X.columns), "labels": labels}, a.model)
        print("\nSaved", a.model)


if __name__ == "__main__":
    main()
