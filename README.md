## OpenPose Tracking on IR Camera

![gif](https://i.imgur.com/Fmi57HJ.gif)

Tracking based on Mediapipe on IR rectified camera 
Send tracking points via OSC :
- /nose [x, y, z]
- /x
- /y

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

![Utilities/landmarks.png](Utilities/landmarks.png)