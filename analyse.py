#!/usr/bin/env python3
"""Analyse a Gesture Lab recording.

Usage:
    python analyse.py recording.csv                 # plot + summary, default thresholds
    python analyse.py recording.csv --sweep 40      # try other thresholds offline
    python analyse.py recording.csv --no-plot

It rebuilds the gesture features from the raw accelerometer and gyroscope columns,
re-runs the same rule-based slide/chop detector as the web app, and compares the
result with what the app detected live and with the game prompts (ground truth).

Needs: numpy, pandas, matplotlib  (pip install numpy pandas matplotlib)

CSV columns (one row per sensor sample):
    t_ms                  time since the recording started
    acc_x/y/z             linear acceleration, gravity removed, m/s^2, device axes
    accg_x/y/z            acceleration including gravity, m/s^2
    rot_x/y/z             turn rate about the device x, y, z axes, deg/s
    yaw_rate, tilt_rate   features the app computed live (deg/s)
    acc_h, acc_v          sideways and vertical acceleration the app computed live
    label                 free-text label typed in the Record tab
    prompt                game prompt showing at that moment: slide, chop, hold or empty
    event                 gesture the live detector fired on that sample: slide, chop or empty
"""
import argparse
import sys

import numpy as np
import pandas as pd


def features(df):
    """Gravity-referenced features, the same maths as the web app."""
    a = df[["acc_x", "acc_y", "acc_z"]].to_numpy(float)
    ag = df[["accg_x", "accg_y", "accg_z"]].to_numpy(float)
    w = df[["rot_x", "rot_y", "rot_z"]].to_numpy(float)
    g = ag - a
    gl = np.linalg.norm(g, axis=1)
    g_hat = np.zeros_like(g)
    last = np.array([0.0, 0.0, 1.0])
    for i in range(len(g)):                      # hold the last good gravity direction
        if gl[i] > 1:
            last = g[i] / gl[i]
        g_hat[i] = last
    yaw = np.einsum("ij,ij->i", w, g_hat)        # turn rate about vertical, deg/s
    wh = w - yaw[:, None] * g_hat                # turn rate about horizontal axes
    a_v = np.einsum("ij,ij->i", a, g_hat)        # vertical acceleration
    a_h = np.linalg.norm(a - a_v[:, None] * g_hat, axis=1)
    return {"t": df["t_ms"].to_numpy(float), "w": w, "yaw": yaw, "wh": wh,
            "tilt": np.linalg.norm(wh, axis=1), "a_h": a_h, "a_v": a_v}


def detect(f, start=120.0, sweep=30.0, dom=0.65, acc=4.0, peak=250.0):
    """Offline copy of the app's rule-based detector. Returns a list of windows."""
    t, w, yaw, wh, a_h, a_v = f["t"], f["w"], f["yaw"], f["wh"], f["a_h"], f["a_v"]
    wm = np.linalg.norm(w, axis=1)
    out, active, quiet_until = [], False, -1e18
    for i in range(len(t)):
        if not active:
            if t[i] < quiet_until or wm[i] <= start:
                continue
            active, t0, last = True, t[i], t[i]
            yaw_int, tv, p_ah, p_av, fired, below, p_fast = 0.0, np.zeros(3), 0.0, 0.0, None, 0.0, 0.0
        dt = min(0.05, max(0.0, (t[i] - last) / 1000.0))
        last = t[i]
        yaw_int += yaw[i] * dt
        tv += wh[i] * dt
        p_ah, p_av = max(p_ah, a_h[i]), max(p_av, abs(a_v[i]))
        dur = t[i] - t0
        if dur <= 150:                           # fast onset: rejects slow vehicle spin
            p_fast = max(p_fast, wm[i])
        ys, ts = abs(yaw_int), float(np.linalg.norm(tv))
        tot = ys + ts or 1e-9
        if fired is None and dur >= 50 and p_fast >= peak:
            if ys >= sweep and ys / tot >= dom and p_ah >= acc:
                fired = ("slide", t[i], dur)
            elif ts >= sweep and ts / tot >= dom and p_av >= acc:
                fired = ("chop", t[i], dur)
        below = below + dt * 1000 if wm[i] < start * 0.5 else 0.0
        if below >= 100 or dur > 900:
            active, quiet_until = False, t[i] + 150
            out.append({"type": fired[0] if fired else "none", "t_start": t0,
                        "t_fire": fired[1] if fired else np.nan,
                        "decision_ms": fired[2] if fired else np.nan, "t_end": t[i],
                        "yaw_sweep": yaw_int, "tilt_sweep": ts, "yaw_share": ys / tot,
                        "peak_acc_h": p_ah, "peak_acc_v": p_av, "peak_fast": p_fast})
    return pd.DataFrame(out)


def trials(df):
    """Group consecutive rows with the same non-empty prompt into game trials."""
    p = df["prompt"].fillna("").astype(str).to_numpy()
    t = df["t_ms"].to_numpy(float)
    res, i = [], 0
    while i < len(p):
        if p[i]:
            j = i
            while j + 1 < len(p) and p[j + 1] == p[i]:
                j += 1
            res.append((p[i], t[i], t[j]))
            i = j + 1
        else:
            i += 1
    return res


def confusion(trial_list, ev_times, ev_types, slack_ms=60):
    kinds, cols = ["slide", "chop", "hold"], ["slide", "chop", "none"]
    m = pd.DataFrame(0, index=kinds, columns=cols)
    for kind, t0, t1 in trial_list:
        hit = [ty for tt, ty in zip(ev_times, ev_types) if t0 <= tt <= t1 + slack_ms]
        m.loc[kind, hit[0] if hit else "none"] += 1
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv")
    ap.add_argument("--start", type=float, default=120, help="turn rate that opens a window, deg/s")
    ap.add_argument("--sweep", type=float, default=30, help="minimum angle swept, deg")
    ap.add_argument("--dom", type=float, default=0.65, help="share of rotation on the dominant axis, 0-1")
    ap.add_argument("--acc", type=float, default=4, help="minimum acceleration peak, m/s^2")
    ap.add_argument("--peak", type=float, default=250, help="turn rate that must be reached within 150 ms, deg/s")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--save", help="write the plot to this image file instead of showing it")
    args = ap.parse_args()

    df = pd.read_csv(args.csv, keep_default_na=False)
    if len(df) < 5:
        sys.exit("Recording is too short.")
    f = features(df)
    dt = np.diff(f["t"])
    print(f"{args.csv}: {len(df)} samples, {f['t'][-1] / 1000:.1f} s, "
          f"{1000 / np.median(dt):.0f} Hz median (sample gap {dt.min():.0f} to {dt.max():.0f} ms)")

    live = df[df["event"] != ""]
    print(f"Live detector in the app: {live['event'].value_counts().to_dict()}")

    win = detect(f, args.start, args.sweep, args.dom, args.acc, args.peak)
    fired = win[win["type"] != "none"] if len(win) else win
    print(f"Offline detector (start={args.start}, sweep={args.sweep}, dom={args.dom}, acc={args.acc}, peak={args.peak}): "
          f"{fired['type'].value_counts().to_dict() if len(fired) else {}}; "
          f"{int((win['type'] == 'none').sum()) if len(win) else 0} movements rejected")
    if len(fired):
        print(f"Time from motion start to decision: median {fired['decision_ms'].median():.0f} ms, "
              f"max {fired['decision_ms'].max():.0f} ms")
        print("\nMovement windows:")
        print(win.round(1).to_string(index=False))

    tl = trials(df)
    if tl:
        print(f"\nGame trials: {len(tl)}. Rows = asked, columns = detected.")
        print("Live detector:")
        print(confusion(tl, live["t_ms"].to_numpy(float), live["event"].to_numpy()).to_string())
        print("Offline detector:")
        print(confusion(tl, fired["t_fire"].to_numpy(float) if len(fired) else [],
                        fired["type"].to_numpy() if len(fired) else []).to_string())

    if args.no_plot:
        return
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    col = {"slide": "#d98a17", "chop": "#1f9e86", "hold": "#5a6fd8"}
    ts = f["t"] / 1000
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    ax[0].plot(ts, f["yaw"], color=col["slide"], lw=1, label="about vertical (slide)")
    ax[0].plot(ts, f["tilt"], color=col["chop"], lw=1, label="about horizontal (chop)")
    ax[0].axhline(args.start, color="grey", ls=":", lw=0.8)
    ax[0].set_ylabel("turn rate, deg/s")
    ax[1].plot(ts, f["a_h"], color=col["slide"], lw=1, label="sideways")
    ax[1].plot(ts, f["a_v"], color=col["chop"], lw=1, label="vertical")
    ax[1].set_ylabel("acceleration, m/s²")
    for name, c in zip(["rot_x", "rot_y", "rot_z"], ["#c0392b", "#27ae60", "#2980b9"]):
        ax[2].plot(ts, df[name], color=c, lw=0.8, label=name)
    ax[2].set_ylabel("raw turn rate, deg/s")
    ax[2].set_xlabel("time, s")
    for a in ax:
        for kind, t0, t1 in tl:
            a.axvspan(t0 / 1000, t1 / 1000, color=col[kind], alpha=0.10, lw=0)
        for _, r in fired.iterrows():
            a.axvline(r["t_fire"] / 1000, color=col[r["type"]], ls="--", lw=1)
        a.legend(loc="upper right", fontsize=8, ncol=3)
        a.grid(alpha=0.25)
    ax[0].set_title("Shaded = game prompt (amber slide, green chop, blue hold). Dashed = offline detection.")
    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=130)
        print(f"\nPlot written to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
