#!/usr/bin/env python3
"""Train and test a gesture classifier on Gesture Lab clips.

Usage:
    python train.py clips.json                      # one export
    python train.py p01.json p02.json p03.json      # several phones or sessions, merged
    python train.py clips.json --save model.joblib --plot confusion.png

Input is the JSON saved from the Collect tab ("Save JSON"). Each clip is one labelled
movement: a list of samples [t_ms, acc xyz, accg xyz, rot xyz] plus who made it, which
hand, posture and prop.

What it does:
  1. Turns every clip into a fixed-length feature vector (see `features`).
  2. Tests a random forest honestly. With two or more participants it leaves one whole
     participant out at a time, which is the number that matters for a ride: guests the
     model has never seen. With one participant it falls back to 5-fold cross-validation
     and says so, because that figure will be optimistic.
  3. Prints accuracy, a per-gesture report and a confusion table, and optionally saves
     the trained model and a confusion plot.

Needs: numpy, scikit-learn, matplotlib (only for --plot), joblib (only for --save)
    pip install numpy scikit-learn matplotlib joblib

The features are small on purpose (61 numbers from simple sums, minima and maxima), so
the same maths can run on a microcontroller. A random forest or a small decision tree
trained on them can be exported to C later; a 1D CNN on the raw samples is the next step
if this plateaus.
"""
import argparse
import json
import sys
from collections import Counter

import numpy as np


def channels(samples):
    """Grip-independent channels per sample, scaled to roughly -1..1.

    Uses the gravity direction (accg - acc) to split turn rate and acceleration into
    'about vertical' and 'about horizontal' parts, so it does not matter which way up
    the device is held. Same maths as the web app.
    """
    s = np.asarray(samples, float)
    a, ag, w = s[:, 1:4], s[:, 4:7], s[:, 7:10]
    g = ag - a
    gl = np.linalg.norm(g, axis=1)
    g_hat = np.zeros_like(g)
    last = np.array([0.0, 0.0, 1.0])
    for i in range(len(g)):
        if gl[i] > 1:
            last = g[i] / gl[i]
        g_hat[i] = last
    yaw = np.einsum("ij,ij->i", w, g_hat)
    tilt = np.linalg.norm(w - yaw[:, None] * g_hat, axis=1)
    a_v = np.einsum("ij,ij->i", a, g_hat)
    a_h = np.linalg.norm(a - a_v[:, None] * g_hat, axis=1)
    wm = np.linalg.norm(w, axis=1)
    return np.stack([yaw / 300, tilt / 300, a_h / 10, a_v / 10, wm / 300], axis=1), s[:, 0]


FEATURE_NAMES = [f"{c}_{k}" for c in ["yaw", "tilt", "acc_h", "acc_v", "turn"]
                 for k in ["mean", "std", "min", "max", "crossings"] + [f"env{i}" for i in range(6)]] + ["log_duration"]


def features(samples):
    """61 numbers per clip: per channel mean, std, min, max, zero crossings and a 6-step
    envelope, plus log duration. Summary statistics do not depend on timing or phase."""
    ch, t = channels(samples)
    n = len(ch)
    f = []
    for c in range(ch.shape[1]):
        v = ch[:, c]
        zc = int(np.sum((np.abs(v[1:]) > 0.15) & (v[1:] * v[:-1] < 0)))
        f += [v.mean(), v.std(), v.min(), v.max(), zc / 4]
        for k in range(6):
            part = np.abs(v[k * n // 6:(k + 1) * n // 6])
            f.append(0.5 * part.mean() if len(part) else 0.0)
    f.append(np.log(max(1.0, t[-1] - t[0]) / 500))
    return np.array(f)


def load(paths):
    clips = []
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        if d.get("format") != "gesture-lab-clips":
            sys.exit(f"{p} is not a Gesture Lab clip export.")
        clips += [c for c in d["clips"] if len(c["samples"]) >= 8]
    return clips


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("json", nargs="+")
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--save", help="write the model trained on all clips to this .joblib file")
    ap.add_argument("--plot", help="write a confusion matrix image to this file")
    args = ap.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_predict

    clips = load(args.json)
    y = np.array([c["label"] for c in clips])
    who = np.array([c.get("participant", "P00") for c in clips])
    X = np.array([features(c["samples"]) for c in clips])
    counts = Counter(y)
    print(f"{len(clips)} clips, {len(counts)} gestures, {len(set(who))} participant(s)")
    for k, v in sorted(counts.items()):
        print(f"  {k:<12}{v}")
    extra = Counter((c.get("hand"), c.get("posture"), c.get("prop"), c.get("source")) for c in clips)
    print("  hand / posture / prop / source:", ", ".join(f"{'/'.join(map(str, k))} x{v}" for k, v in extra.items()))
    if len(counts) < 2:
        sys.exit("Need at least two gestures to train.")
    small = [k for k, v in counts.items() if v < 5]
    if small:
        print(f"Warning: fewer than 5 clips for {', '.join(small)}. Results for those will be unreliable.")

    model = RandomForestClassifier(n_estimators=args.trees, random_state=0, class_weight="balanced")
    if len(set(who)) >= 2:
        cv, groups, how = LeaveOneGroupOut(), who, "leave one participant out"
    else:
        folds = max(2, min(5, min(counts.values())))
        cv, groups, how = StratifiedKFold(folds, shuffle=True, random_state=0), None, \
            f"{folds}-fold on a single participant (optimistic: test with more people)"
    pred = cross_val_predict(model, X, y, cv=cv, groups=groups)
    labels = sorted(counts)
    print(f"\nTested by: {how}")
    print(f"Accuracy: {100 * np.mean(pred == y):.1f}%\n")
    print(classification_report(y, pred, labels=labels, zero_division=0))
    cm = confusion_matrix(y, pred, labels=labels)
    width = max(len(l) for l in labels) + 2
    print("Confusion table. Rows = true gesture, columns = predicted.")
    print(" " * width + "".join(f"{l[:9]:>10}" for l in labels))
    for l, row in zip(labels, cm):
        print(f"{l:<{width}}" + "".join(f"{v:>10}" for v in row))

    model.fit(X, y)
    top = np.argsort(model.feature_importances_)[::-1][:8]
    print("\nMost useful features:", ", ".join(f"{FEATURE_NAMES[i]} ({model.feature_importances_[i]:.2f})" for i in top))

    if args.save:
        import joblib
        joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "labels": labels}, args.save)
        print(f"Model written to {args.save}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2.5, 1.0 * len(labels) + 2))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(f"{how}\naccuracy {100 * np.mean(pred == y):.1f}%", fontsize=10)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"Plot written to {args.plot}")


if __name__ == "__main__":
    main()
