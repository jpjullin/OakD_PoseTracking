## Body tracking on Oak-D Camera

![gif](utils/tracking_demo.gif)

Tracking 33 points of human body on an Oak-D camera 

Tracking available on:
- IR mono right rectified stream
- Disparity

Based on Mediapipe pose: https://github.com/google/mediapipe.git

Sending x33 tracking points via OSC :
- /nose [x, y, z]
- /x [:33]
- /y [:33]

## Pre-requisites

Install requirements:
```
python3 -m pip install opencv-python --force-reinstall --no-cache-dir
python3 -m pip install depthai
python3 -m pip install mediapipe
python3 -m pip install python-osc
```

## Usage

```
python3 main.py
```

## Landmarks

![utils/landmarks.png](utils/landmarks.png)