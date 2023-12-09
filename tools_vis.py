#!/usr/bin/env python3

import time
import cv2

pTime = 0


def show_frame(frame):
    cTime = time.time()
    global pTime
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(frame, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    return frame
