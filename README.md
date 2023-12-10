## Body tracking on Oak-D Camera

![gif](utils/tracking_demo.gif)

Tracking 33 points of human body on an Oak-D camera (infrared mono rectified stream)

Based on Mediapipe pose: 

Sending x33 tracking points via OSC :
- /nose [x, y, z]
- /x [:33]
- /y [:33]

+ Optional send video via NDI

## Pre-requisites

Install requirements:
```
python3 -m pip install opencv-python --force-reinstall --no-cache-dir

[//]: # (python3 install_requirements.py)

python3 -m pip install mediapipe
python3 -m pip install python-osc
python3 -m pip install ndi-python
```

## Usage

```
python3 main_v5.py
```

## Landmarks

![utils/landmarks.png](utils/landmarks.png)