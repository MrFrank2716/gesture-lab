# Gesture Lab

A phone stands in for the handheld. The page reads the phone's accelerometer and gyroscope and gives you:

- a live 3D view of a wand, blaster or ninja blade that follows the phone's orientation
- a phone link, so the 3D view and the tools sit on a laptop screen while the phone is in your hand
- guided capture of labelled gesture clips, saved as a dataset for training a model
- the rule-based slide and chop detector, with a target game, a live signal plot and threshold sliders

## Files

- `index.html`: the whole app in one file. Host this.
- `gesture-lab.html`: the same page without the outer HTML shell (the source; `sh build.sh` rebuilds `index.html` from it).
- `train.py`: trains a first classifier from dataset files and compares it with the rules.
- `sample-dataset.csv`: 36 clips made with the drag-pad simulator, so `train.py` can be tried straight away. Not real motion.
- `analyse.py`: plots a continuous recording and re-runs the rule detector offline with any thresholds.
- `sample-recording.csv`: a short synthetic continuous recording for `analyse.py`.

## Running it

Browsers only give motion data to pages served over HTTPS, so opening the file directly will
not work. Put `index.html` on any HTTPS host, for example GitHub Pages.

**Phone on its own.** Open the link on the phone and tap "Start sensors". iPhone asks for
motion permission at that tap.

**Phone as the sensor, laptop as the screen (recommended).** You cannot watch a phone you are
swinging, so:

1. Open the page on the laptop and press "Link a phone". It shows a five character code and a QR code.
2. Scan the QR code with the phone (or open the page, tap "This phone is the sensor" and type the code), then tap "Start sending".
3. Point the top of the phone at the laptop screen and press "Recentre" (or the R key).

The laptop now runs everything from the phone's motion: the 3D view, capture, the game, the plot.
On Android the phone buzzes when a gesture is recognised and on each capture cue.

The link uses [PeerJS](https://peerjs.com) (WebRTC). The two browsers find each other through
PeerJS's free public pairing server, then motion samples go directly between them. Only the
code and connection details pass through that server. To avoid it, run your own PeerServer and
set `window.GL_PEER = {host, port, path, secure}` before the page script.

**No sensor** (a laptop on its own, or the copy inside Claude): a drag pad appears and fakes
the motion. You can also drag the 3D view.

Vibration works on Android only; iPhone browsers do not expose it. Sound works on both.

## The 3D view

The item is drawn with its grip at your hand and its tip along the top edge of the phone.
Orientation comes from integrating the gyroscope, corrected by gravity so tilt cannot drift.
There is no compass in the loop (the planned IMU has none either), so heading drifts slowly;
"Recentre" resets it. The view shows orientation only. Position cannot be tracked from an IMU alone.

If the model is upside down, tick "Flip gravity" on the Tune tab (or on the phone's sending
screen). iPhones report acceleration with the opposite sign to Android and the page corrects
for that automatically, but this has not been checked on real phones yet.

## Collecting a training set

On the Collect tab pick a gesture (or add your own), set a participant code and press Capture.
For each repeat you hear a low tick, then a higher GO beep: make the gesture on the beep.
Every repeat becomes one 1.5 s clip that starts 0.25 s before the beep. "Save dataset CSV"
writes all clips to one file with one row per sample:

    clip_id, label, participant, session, item, source, rep, t_ms,
    acc_x/y/z      acceleration without gravity, m/s^2, device axes
    accg_x/y/z     acceleration with gravity, m/s^2
    rot_x/y/z      turn rate, deg/s
    yaw_rate, tilt_rate, acc_h, acc_v    the grip-independent features the rules use
    rule_event     what the rule detector fired on that sample, if anything

Clips live in the page until saved, so save before closing the tab. Rules of thumb for a first
model: 50 or more clips per gesture, several people, and plenty of "still" and "other" clips
so it learns what is not a gesture. Use codes, not names, for participants.

## Training a first model

    pip install numpy pandas scikit-learn joblib
    python train.py sample-dataset.csv
    python train.py p01.csv p02.csv p03.csv --depth 4 --model gestures.joblib

It turns each clip into summary features, then reports the rule detector, a small readable
decision tree and a random forest side by side. With three or more participants the test is
leave one person out, which is the honest number for a ride where every guest is new.

## How the detector works

Each sample is turned into grip-independent features using the gravity direction:
turn rate about vertical, turn rate about horizontal axes, sideways acceleration and vertical
acceleration. A window opens when total turn rate passes a start threshold. A slide is enough
angle swept about vertical, with that axis dominant and a sideways acceleration peak. A chop is
the same about a horizontal axis with a vertical acceleration peak. It fires as soon as the
rules are met, so the decision arrives during the movement.

Both also need a fast onset: turn rate must reach 250 deg/s within 150 ms of the window
opening. Trackless ride vehicles can spin at up to about 120 deg/s, and without this rule a
steady spin could pass as a slow slide.

Defaults: start 120 deg/s, sweep 30 deg, dominance 65%, acceleration 4 m/s^2, fast-onset peak 250 deg/s.

## Analysing a continuous recording

    pip install numpy pandas matplotlib
    python analyse.py sample-recording.csv
    python analyse.py my-take.csv --sweep 40 --dom 0.7

It prints sample rate, detections, time to decision, and a confusion table against the game
prompts, then plots the signals. In MATLAB, `readtable('my-take.csv')` loads the same file.
