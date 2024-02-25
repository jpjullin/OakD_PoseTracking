## Body tracking on Oak-D Camera

Tracking 17 points of human body on an Oak-D camera

Based on Mediapipe pose: https://github.com/geaxgx/depthai_movenet

Sending x17 tracking points via OSC :
- /nose [x, y]
- /x [:17]
- /y [:17]

## Pre-requisites

Install requirements:
```
pip install -r requirements.txt
```

## Usage

```
python main.py
```