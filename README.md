# Gesture Lab

A phone stands in for the handheld. The page reads the phone's accelerometer and gyroscope,
recognises a slide and a chop with simple rules, and gives you a target game, a live signal
plot, labelled CSV recording and threshold sliders.

## Files

- `index.html`: the whole app in one file. Host this.
- `gesture-lab.html`: the same page without the outer HTML shell (the source; `sh build.sh` rebuilds `index.html` from it).
- `analyse.py`: plots a recording and re-runs the detector offline with any thresholds.
- `sample-recording.csv`: a short synthetic recording so you can try `analyse.py` straight away.

## Running it on a phone

Browsers only give motion data to pages served over HTTPS, so opening the file directly will
not work. Put `index.html` on any HTTPS host, for example GitHub Pages, then open the link on
the phone and tap "Start sensors". iPhone asks for motion permission at that tap. Without a
sensor (a laptop, or the copy inside Claude) a drag pad appears and fakes the motion.

Vibration works on Android only; iPhone browsers do not expose it. Sound works on both.

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

## Analysing a recording

    pip install numpy pandas matplotlib
    python analyse.py sample-recording.csv
    python analyse.py my-take.csv --sweep 40 --dom 0.7

It prints sample rate, detections, time to decision, and a confusion table against the game
prompts, then plots the signals. In MATLAB, `readtable('my-take.csv')` loads the same file.
