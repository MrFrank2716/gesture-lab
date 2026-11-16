# Gesture Lab

A phone stands in for the handheld. The page reads the phone's accelerometer and gyroscope and gives you:

- **Prop:** a live 3D wand, blaster, katana or shuriken that follows the phone, with a motion trail.
- **Collect:** labelled gesture clips for training a machine-learning model later, plus a quick in-browser model to preview what the clips can do.
- **Game:** a 45 second target game driven by a rule-based slide and chop detector.
- **Signals:** a live plot and a continuous recording.
- **Tune:** thresholds for the rule-based detector.

## Files

- `index.html`: the whole app in one file. Host this.
- `gesture-lab.html`: the same page without the outer HTML shell (the source; `sh build.sh` rebuilds `index.html` from it).
- `train.py`: trains and tests a classifier on clips saved from the Collect tab.
- `analyse.py`: plots a continuous recording and re-runs the rule-based detector offline.
- `sample-clips.json`, `sample-recording.csv`: short synthetic files so both scripts can be tried straight away. They are not real gestures.

## Running it on a phone

Browsers only give motion data to pages served over HTTPS, so opening the file directly will
not work. Put `index.html` on any HTTPS host, for example GitHub Pages, then open the link on
the phone and tap "Start sensors". iPhone asks for motion permission at that tap. Without a
sensor (a laptop, or the copy inside Claude) a drag pad appears and fakes the motion.

Vibration works on Android only; iPhone browsers do not expose it. Sound works on both.

## The 3D prop

Hold the phone like the prop, top edge pointing away from you, and tap Recentre. Two tracking
modes:

- **Phone's own fusion** uses the orientation the phone already computes.
- **6-axis filter** integrates the gyroscope and corrects tilt from the accelerometer, which is
  what a handheld with a 6-axis IMU and no magnetometer would do. Its heading drifts slowly.
  Comparing the two shows how much a docking recalibration would need to correct.

If the prop rolls over or hangs upside down in the filter mode, flip the accelerometer sign on
the Tune tab. Phones disagree about that sign.

## Collecting a dataset

1. On the Collect tab enter a participant code (not a name), hand and posture.
2. Pick a gesture, tap "Start capturing", and repeat the gesture with a half-second pause between
   repetitions. Each movement is cut out automatically and saved as one clip. "Discard last"
   removes a bad repetition.
3. Pick "background" to save a clip every 1.5 s of anything that is not a gesture: sitting,
   walking, fidgeting, a bus or train ride. A model needs plenty of these to avoid false triggers.
4. "Save JSON" for Python, "Save CSV" for MATLAB. Clips also stay in the browser until deleted.

Aim for at least 20 clips per gesture per person, several people of different ages and both
hands, and as much background as gestures.

Each clip holds raw samples `[t_ms, acc_x, acc_y, acc_z, accg_x, accg_y, accg_z, rot_x, rot_y, rot_z]`
(acceleration without gravity and with gravity in m/s^2, turn rate in deg/s about the device axes)
plus label, participant, hand, posture, prop and whether it came from a real sensor (`imu`) or
the drag pad (`sim`).

## Training a model

    pip install numpy scikit-learn matplotlib joblib
    python train.py sample-clips.json
    python train.py p01.json p02.json --plot confusion.png --save model.joblib

It builds 61 simple features per clip, trains a random forest, and tests it by leaving one whole
participant out at a time when there are two or more participants. That is the honest number for
a ride, where every guest is new to the model.

## The rule-based detector

Each sample is turned into grip-independent features using the gravity direction: turn rate
about vertical, turn rate about horizontal axes, sideways acceleration and vertical
acceleration. A window opens when total turn rate passes a start threshold. A slide is enough
angle swept about vertical, with that axis dominant and a sideways acceleration peak. A chop is
the same about a horizontal axis with a vertical acceleration peak. It fires as soon as the
rules are met, so the decision arrives during the movement.

Both also need a fast onset: turn rate must reach 250 deg/s within 150 ms of the window
opening. Trackless ride vehicles can spin at up to about 120 deg/s, and without this rule a
steady spin could pass as a slow slide.

Defaults: start 120 deg/s, sweep 30 deg, dominance 65%, acceleration 4 m/s^2, fast-onset peak 250 deg/s.

    pip install numpy pandas matplotlib
    python analyse.py sample-recording.csv
    python analyse.py my-take.csv --sweep 40 --dom 0.7

In MATLAB, `readtable('my-take.csv')` loads the same files.
